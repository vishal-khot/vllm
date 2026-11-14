# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment 2, step 1: capture real IVF keys and queries from a live serve.

One eager IVF serve per length (TP from runs.yaml) with ``key_capture`` on.
Each configured RULER prompt is sent alone with a one-token answer, so only its
prefill runs, and the hook saves ``<out>/keys/L<length>/<task>_<idx>/L<layer>.pt``.
A prompt whose layer files all exist is skipped, so a requeued job resumes.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.join(os.path.dirname(EXP_DIR), "benchmark")
sys.path.insert(0, os.path.join(BENCH_DIR, "slurm"))
sys.path.insert(0, BENCH_DIR)
sys.path.insert(0, EXP_DIR)

import lost_attention_run as lar  # noqa: E402
from bench_config import nothink_template  # noqa: E402
from exp_config import load, ruler_samples, serve_cmd, serve_env  # noqa: E402


def _done(label_dir: str, layers: list[int]) -> bool:
    return all(os.path.exists(os.path.join(label_dir, f"L{x}.pt")) for x in layers)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--runs-yaml", default=os.path.join(EXP_DIR, "runs.yaml"))
    p.add_argument("--port", type=int, required=True)
    args = p.parse_args()
    cfg = load(args.runs_yaml)
    cap = cfg["reorder"]["capture"]
    lar.THINKING = False
    lar._exit_on_sigterm()
    keys_root = os.path.join(args.out_dir, "keys")
    os.makedirs(keys_root, exist_ok=True)
    template = nothink_template(cfg["model_dir"], args.out_dir)

    for length in cap["lengths"]:
        cap_dir = os.path.join(keys_root, f"L{length}")
        samples = ruler_samples(cfg, length, cap["tasks"], [cap["sample_idx"]])
        todo = [s for s in samples
                if not _done(os.path.join(cap_dir, s["label"]), cap["layers"])]
        if not todo:
            lar._log(f"[capture L{length}] complete, skipping")
            continue
        grid = cfg["ivf_grid"][length]
        clusters, probes = grid["clusters"], grid["probes"][cap["probe_share"]]
        os.makedirs(cap_dir, exist_ok=True)
        env = serve_env(indexer="ivf", clusters=clusters, probes=probes)
        env["GLM5_KEY_CAPTURE_DIR"] = cap_dir
        env["GLM5_KEY_CAPTURE_LAYERS"] = ",".join(map(str, cap["layers"]))
        cmd = serve_cmd(cfg, length=length, tp=cap["tp"], port=args.port,
                        util=cap["gpu_memory_utilization"], indexer="ivf",
                        clusters=clusters, probes=probes, eager=True,
                        chat_template=template)
        lar._log(f"[capture L{length}] serve: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, env=env, start_new_session=True)
        hung = False
        try:
            try:
                lar._wait_ready(proc, args.port, 3600)
            except TimeoutError:
                hung = True
                raise
            for s in todo:
                label_dir = os.path.join(cap_dir, s["label"])
                lar._write_atomic(os.path.join(cap_dir, "current_label"), s["label"])
                body = {
                    "model": "glm",
                    **lar.ruler_messages(s["input"], s["answer_prefix"]),
                    "max_tokens": 1, "temperature": 0, "seed": cfg["fairness"]["seed"],
                }
                t0 = time.time()
                lar._post_with_retry(
                    f"http://127.0.0.1:{args.port}/v1/chat/completions",
                    body, 7200, 2)
                time.sleep(2)  # the hook writes from the worker process
                ok = _done(label_dir, cap["layers"])
                lar._log(f"[capture L{length}] {s['label']}: "
                         f"{'saved' if ok else 'MISSING'} in {time.time() - t0:.0f}s")
                if not ok:
                    raise RuntimeError(f"capture missing for {label_dir}")
                with open(os.path.join(label_dir, "meta.json"), "w") as f:
                    json.dump({k: s[k] for k in ("task", "idx", "L", "label")}, f)
        finally:
            lar._write_atomic(os.path.join(cap_dir, "current_label"), "")
            lar._stop_serve(proc, dump_stacks=hung)
    lar._log("[capture] all lengths complete")


if __name__ == "__main__":
    main()
