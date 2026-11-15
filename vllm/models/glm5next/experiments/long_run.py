# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment 5: a 2M-token context, past GLM-5.3's trained 1M limit.

RULER accuracy and per-request prefill time of kpool against IVF at the probe
shares in ``long.ivf_probe_indices``, on every RULER task at ``long.length``.
The model's ``max_position_embeddings`` (1,048,576 in config.json) is raised by
``--hf-overrides`` so the serve accepts the length; MLA has no rotary part and
KDA has no positions, but the indexer's rotary positions past 1M were never
trained, so accuracy here is out of distribution for both arms alike.

The samples are split into ``long.halves`` contiguous index ranges; one process
(``--half``) serves each arm in turn on its own GPUs and sends its range one
request at a time, so TTFT is the request's whole prefill. Outputs under
``--out-dir``:

    <cell>/serve_h<half>.log     the serve's log
    <cell>/results_h<half>.jsonl one record per request (checkpoint)
    summary.json                 every cell, merged over the halves

A request with a record is not sent again, so a requeued job resumes where it
stopped. The first request after each serve start is flagged ``cold`` and left
out of the timing summary (JIT and autotuning at a new length).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
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
from exp_config import (  # noqa: E402
    load,
    max_tokens,
    ruler_samples,
    serve_cmd,
    serve_env,
)


def cells(cfg: dict) -> list[dict]:
    """kpool, then IVF at each configured probe share."""
    p = cfg["long"]
    length = p["length"]
    grid = cfg["ivf_grid"][length]
    out = []
    if "kpool" in p["arms"]:
        out.append({"id": f"kpool_L{length}", "indexer": "kpool", "length": length,
                    "clusters": None, "probes": None})
    if "ivf" in p["arms"]:
        for i in p["ivf_probe_indices"]:
            probes = grid["probes"][i]
            out.append({"id": f"ivf_C{grid['clusters']}_P{probes}_L{length}",
                        "indexer": "ivf", "length": length,
                        "clusters": grid["clusters"], "probes": probes})
    return out


def half_indices(cfg: dict, half: int) -> list[int]:
    """Sample rows of ``half``: contiguous ranges of range(n)."""
    p = cfg["long"]
    n, halves = int(p["n"]), int(p["halves"])
    per = -(-n // halves)
    return list(range(half * per, min(n, (half + 1) * per)))


def samples(cfg: dict, half: int) -> list[dict]:
    """This half's samples, task-interleaved: row i of every task, then i+1."""
    p = cfg["long"]
    rcfg = {**cfg, "ruler_dir": p["ruler_dir"]}
    by_task = [ruler_samples(rcfg, p["length"], [t], half_indices(cfg, half))
               for t in p["tasks"]]
    return [s for row in zip(*by_task) for s in row]


def long_cmd(cfg: dict, cell: dict, port: int, chat_template: str | None) -> list[str]:
    """The matrix's serve flags, plus the raised position limit."""
    p = cfg["long"]
    cmd = serve_cmd(cfg, length=cell["length"], tp=p["tp"], port=port,
                    util=p["util"], indexer=cell["indexer"],
                    clusters=cell["clusters"], probes=cell["probes"],
                    chat_template=chat_template)
    pos = {"max_position_embeddings": int(p["max_position_embeddings"])}
    if "--hf-overrides" in cmd:
        i = cmd.index("--hf-overrides") + 1
        over = json.loads(cmd[i])
        over["text_config"].update(pos)
        cmd[i] = json.dumps(over)
    else:
        cmd += ["--hf-overrides", json.dumps({"text_config": pos})]
    return cmd


def _request(port: int, sample: dict, cfg: dict) -> dict:
    p = cfg["long"]
    body = {
        "model": "glm",
        **lar.ruler_messages(sample["input"], sample["answer_prefix"]),
        "max_tokens": max_tokens(cfg, sample["task"]),
        "temperature": 0,
        "seed": cfg["fairness"]["seed"],
    }
    t0 = time.time()
    reply = lar._post_with_retry(
        f"http://127.0.0.1:{port}/v1/chat/completions", body,
        p["request_timeout_s"], 1)
    return {
        "label": sample["label"], "task": sample["task"], "idx": sample["idx"],
        "gold": sample["gold"],
        **lar._score(sample["gold"], reply, sample["task"]),
        "ttft_s": round(reply.ttft_s, 3),
        "decode_s": round(reply.decode_s, 3),
        "total_s": round(time.time() - t0, 3),
        "prompt_tokens": reply.usage.get("prompt_tokens"),
        "out_tok": reply.usage.get("completion_tokens"),
    }


def run_cell(cfg: dict, cell: dict, args) -> None:
    p = cfg["long"]
    cell_dir = os.path.join(args.out_dir, cell["id"])
    os.makedirs(cell_dir, exist_ok=True)
    ckpt = lar._Results(os.path.join(cell_dir, f"results_h{args.half}.jsonl"),
                        key=lambda r: r["label"])
    todo = [s for s in samples(cfg, args.half) if s["label"] not in ckpt.done]
    if not todo:
        lar._log(f"[{cell['id']}] half {args.half}: complete, skipping")
        return
    for start in range(1, int(p["max_serve_starts"]) + 1):
        cmd = long_cmd(cfg, cell, args.port, args.chat_template)
        lar._log(f"[{cell['id']}] half {args.half} start {start}: {' '.join(cmd)}")
        t0 = time.time()
        log_path = os.path.join(cell_dir, f"serve_h{args.half}.log")
        with open(log_path, "a") as log:
            proc = subprocess.Popen(
                cmd, env=serve_env(indexer=cell["indexer"], clusters=cell["clusters"],
                                   probes=cell["probes"]),
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        cold = True
        try:
            lar._wait_ready(proc, args.port, p["startup_timeout_s"])
            lar._log(f"[{cell['id']}] ready in {time.time() - t0:.0f}s, "
                     f"{len(todo)} requests to send")
            while todo:
                s = todo[0]
                try:
                    rec = _request(args.port, s, cfg)
                except Exception as e:
                    if proc.poll() is not None:
                        raise RuntimeError(
                            f"serve died at {s['label']}: {e!r}") from e
                    # A live serve that failed one request: record it, go on.
                    rec = {"label": s["label"], "task": s["task"], "idx": s["idx"],
                           "gold": s["gold"], "recall": 0.0, "error": repr(e)}
                rec["cold"] = cold
                rec["start"] = start
                cold = False
                ckpt.add(rec)
                todo.pop(0)
                lar._log(f"[{cell['id']}] {rec['label']}: recall {rec['recall']:.2f} "
                         f"ttft {rec.get('ttft_s')}s prompt {rec.get('prompt_tokens')}"
                         f"{' ERROR ' + rec['error'] if 'error' in rec else ''}")
            return
        except Exception as e:  # restart the serve once more, then give up
            lar._log(f"[{cell['id']}] half {args.half} start {start} FAILED: {e!r}")
        finally:
            lar._stop_serve(proc, dump_stacks=proc.poll() is None and bool(todo))
    lar._log(f"[{cell['id']}] half {args.half}: {len(todo)} requests not done")


def summarize(cfg: dict, out_dir: str) -> dict:
    """Per cell over every half: recall by task and overall, prefill times."""
    out = {}
    for cell in cells(cfg):
        recs = []
        for path in sorted(glob.glob(os.path.join(out_dir, cell["id"],
                                                  "results_h*.jsonl"))):
            recs += lar._Results(path, key=lambda r: r["label"]).done.values()
        if not recs:
            continue
        by_task: dict[str, list[float]] = {}
        for r in recs:
            by_task.setdefault(r["task"], []).append(r["recall"])
        warm = [r["ttft_s"] for r in recs if "ttft_s" in r and not r.get("cold")]
        task_means = {t: round(statistics.fmean(v), 4) for t, v in by_task.items()}
        out[cell["id"]] = {
            **cell,
            "n": len(recs),
            "errors": sum("error" in r for r in recs),
            "recall": round(statistics.fmean(task_means.values()), 4),
            "recall_by_task": task_means,
            "n_by_task": {t: len(v) for t, v in by_task.items()},
            "ttft_s_mean": round(statistics.fmean(warm), 2) if warm else None,
            "ttft_s_median": round(statistics.median(warm), 2) if warm else None,
            "prompt_tokens_max": max((r.get("prompt_tokens") or 0) for r in recs),
        }
    lar._write_atomic(os.path.join(out_dir, "summary.json"), json.dumps(out, indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--runs-yaml", default=os.path.join(EXP_DIR, "runs.yaml"))
    ap.add_argument("--port", type=int)
    ap.add_argument("--half", type=int, default=0)
    ap.add_argument("--only", help="comma list of cell ids")
    ap.add_argument("--dry-run", action="store_true",
                    help="print cells, sample counts and serve commands; start nothing")
    ap.add_argument("--summarize", action="store_true",
                    help="only merge the halves' results into summary.json")
    args = ap.parse_args()
    cfg = load(args.runs_yaml)
    todo = cells(cfg)
    if args.only:
        want = set(args.only.split(","))
        todo = [c for c in todo if c["id"] in want]
    if args.summarize:
        print(json.dumps(summarize(cfg, args.out_dir), indent=2))
        return
    if args.dry_run:
        for h in range(int(cfg["long"]["halves"])):
            print(f"half {h}: sample rows {half_indices(cfg, h)}")
        for c in todo:
            print(c["id"], " ".join(long_cmd(cfg, c, args.port or 0, None)))
        return
    assert 0 <= args.half < int(cfg["long"]["halves"]) and args.port
    lar.THINKING = False
    lar._exit_on_sigterm()
    os.makedirs(args.out_dir, exist_ok=True)
    args.chat_template = nothink_template(cfg["model_dir"], args.out_dir)
    for c in todo:
        try:
            run_cell(cfg, c, args)
        except Exception as e:  # one failed cell must not stop the others
            lar._log(f"[{c['id']}] FAILED: {e!r}")
        summarize(cfg, args.out_dir)
    lar._log(f"[long] half {args.half} done -> {args.out_dir}/summary.json")


if __name__ == "__main__":
    main()
