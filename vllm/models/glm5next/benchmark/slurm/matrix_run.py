# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A/B benchmark matrix on Slurm (no Modal), driven by runs.yaml.

The Slurm counterpart of ``../matrix_run.py`` (which only runs on Modal). It
parses the run matrix with ``bench_config``, serves each run once, runs its
datasets through that serve, and stores results **per run**, keyed by
``run.id`` (unique by construction) so no two runs collide:

    <out>/<run.id>/
        accuracy.json       per-task recall + mean (ruler)
        latency.json        single-stream prefill/decode (concurrency == 1)
        throughput.json     aggregate tok/s        (concurrency > 1)
        timing/             indexer_timing_pid*_r*.json (eager runs only)
        lost_attention/     lost_attention_pid*.json    (base ivf only)
        verdict.json        per-run roll-up
    <out>/summary.json      roll-up over every run

Model + RULER staging and the venv are done by ``_matrix_body.sh``; this script
assumes the model and ``<task>/validation_<len>.jsonl`` are already local and a
vLLM+fork env is active. regression/swebench have no in-repo scorer yet, so
those datasets are recorded as ``scorer_not_implemented`` rather than skipped.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import shutil
import sys
import time

BENCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLURM_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BENCH_DIR)
sys.path.insert(0, SLURM_DIR)
from bench_config import load_matrix  # noqa: E402
from tasks import TASKS, recall  # noqa: E402

# Reuse the proven serve/HTTP/summary helpers from the fidelity runner.
from lost_attention_run import (  # noqa: E402
    _check_ivf_counts,
    _get,
    _post_stream,
    _start_server,
    _summarize_lost,
    _summarize_timing,
    _wait_ready,
)

_HOUR = 3600


def _serve_env(run, timing_dir: str, lost_dir: str) -> dict:
    """Serve-process env: arm env from bench_config, instrument on only for
    eager runs, lost-attention dir only when the run measures it."""
    env = dict(os.environ)
    env["PYTHONPATH"] = BENCH_DIR + (
        ":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env.update(run.env)  # VLLM_ENABLE_DSA_IVF_INDEXER, measure flag
    if run.indexer == "kpool":
        # The stock arm runs entirely inside the stock venv, which has no IVF.
        env.pop("VLLM_ENABLE_DSA_IVF_INDEXER", None)
        stock_bin = os.path.dirname(_stock_vllm())
        env["VIRTUAL_ENV"] = os.path.dirname(stock_bin)
        env["PATH"] = stock_bin + os.pathsep + env.get("PATH", "")
    if run.eager:
        env["GLM5_INSTRUMENT"] = "1"
        env["GLM5_TIMING_DIR"] = timing_dir
        env["GLM5_MEM_PROBE"] = "1"
        env["GLM5_RESIDENT_PROBE"] = "1"
    if run.measure_lost_attention:
        env["VLLM_DSA_IVF_LOST_ATTENTION_DIR"] = lost_dir
    return env


def _stock_vllm() -> str:
    """The unmodified vLLM release's `vllm` (GLM5_STOCK_VLLM), for kpool."""
    path = os.environ.get("GLM5_STOCK_VLLM")
    if not path or not os.access(path, os.X_OK):
        raise RuntimeError(
            "kpool runs serve stock vLLM: set GLM5_STOCK_VLLM to the stock "
            f"venv's bin/vllm (got {path!r}); see _build_venv.sh"
        )
    return path


def _serve_cmd(run, model: str, tp: int, port: int, gpu_mem_util: float) -> list:
    """`vllm serve` for a run: fixed flags plus the run's own serve_args
    (max-model-len, seed, eager, prefix-caching, hf-overrides). kpool runs
    serve the stock release; IVF runs serve the fork."""
    vllm_bin = (
        _stock_vllm()
        if run.indexer == "kpool"
        else shutil.which("vllm") or "vllm"
    )
    return [
        vllm_bin, "serve", model,
        "--served-model-name", "glm",
        "--tensor-parallel-size", str(tp),
        "--gpu-memory-utilization", str(gpu_mem_util),
        "--port", str(port),
    ] + run.serve_args()


def _one_request(port: int, meta: dict, request_timeout: float) -> dict:
    ttft, dec_s, text, usage = _post_stream(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        {
            "model": "glm",
            "messages": [{"role": "user", "content": meta["input"]}],
            "max_tokens": TASKS[meta["task"]]["max_tokens"],
            "temperature": 0,
        },
        timeout=request_timeout,
    )
    otk = usage.get("completion_tokens")
    return {
        "task": meta["task"], "L": meta["L"], "idx": meta["idx"],
        "recall": recall(meta["gold"], text),
        "prefill_ms": round(ttft * 1000, 1),
        "decode_ms": round(dec_s * 1000, 1),
        "decode_s": dec_s, "out_tok": otk,
    }


def _run_block(
    port: int, metas: list, request_timeout: float, concurrency: int,
    block_path: str, force: bool = False,
) -> tuple[list, bool]:
    """Run one (length, task) block and checkpoint its results to disk.

    Returns ``(results, cached)``. If ``block_path`` already holds a complete
    block and ``force`` is False, it is loaded and no requests are issued, so a
    requeued run skips blocks a previous attempt finished. The write is atomic
    (tmp + rename) so a preemption mid-write leaves the old block intact.
    """
    if not force and os.path.exists(block_path):
        try:
            with open(block_path) as f:
                cached = json.load(f)
            if len(cached) == len(metas):
                return cached, True
        except Exception:
            pass  # truncated/unreadable -> re-run the block
    if concurrency <= 1:
        res = [_one_request(port, m, request_timeout) for m in metas]
    else:
        with cf.ThreadPoolExecutor(max_workers=concurrency) as pool:
            res = list(pool.map(
                lambda m: _one_request(port, m, request_timeout), metas))
    tmp = block_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f)
    os.replace(tmp, block_path)
    return res, False


def run_one(run, args) -> dict:
    """Serve one run, run its datasets, store results under its own id."""
    base_out = os.path.abspath(os.path.join(args.out_dir, run.id))
    partial_dir = os.path.join(base_out, "partial")
    # Resume a preempted eager run from its per-block checkpoints: keep partial/
    # if it exists (and not --force), else start clean. The per-rank instrument
    # dumps (timing/, lost_attention/) are always cleared so a prior attempt's
    # dumps don't blend into this one -- they are re-measured over whatever
    # blocks this attempt actually serves.
    resuming = os.path.isdir(partial_dir) and not args.force
    if os.path.isdir(base_out) and not resuming:
        shutil.rmtree(base_out)
    timing_dir = os.path.join(base_out, "timing")
    lost_dir = os.path.join(base_out, "lost_attention")
    for d in (timing_dir, lost_dir):
        if os.path.isdir(d):
            shutil.rmtree(d)
    for d in (base_out, timing_dir, lost_dir, partial_dir):
        os.makedirs(d, exist_ok=True)

    # A local model path must survive the server's cwd change.
    model = os.path.abspath(args.model) if os.path.exists(args.model) else args.model
    cmd = _serve_cmd(run, model, args.tp, args.port, args.gpu_mem_util)
    print(f"[{run.id}] serve: {' '.join(cmd)}", flush=True)
    # Start the server outside both clones: a cwd on sys.path (python -c
    # children) would otherwise import that clone's vllm/ package.
    proc, ivf_counts = _start_server(
        cmd, _serve_env(run, timing_dir, lost_dir), base_out
    )

    accuracy: dict = {}
    latency = None
    throughput = None
    skipped: dict = {}
    try:
        t0 = time.time()
        _wait_ready(proc, args.port, args.startup_timeout)
        print(f"[{run.id}] ready in {time.time() - t0:.0f}s", flush=True)
        if run.indexer == "ivf":
            _check_ivf_counts(
                ivf_counts,
                run.hf_overrides["index_num_clusters"],
                run.hf_overrides["index_num_probes"],
            )

        for ds in run.datasets:
            if ds.name != "ruler":
                skipped[ds.name] = "scorer_not_implemented"
                continue

            # Eager runs are the instrumented accuracy/fidelity points: every
            # sample. Graph runs are throughput/latency: a per-task cap is fine.
            cap = None if run.eager else (ds.n or 1)

            def _metas(length, task):
                path = os.path.join(
                    args.ruler_dir, task, f"validation_{length}.jsonl")
                if not os.path.exists(path):
                    raise SystemExit(f"missing ruler data: {path}")
                rows = [json.loads(line) for line in open(path)][:cap]
                return [{"task": task, "L": length, "idx": ex["index"],
                         "gold": ex["outputs"], "input": ex["input"]}
                        for ex in rows]

            wall0 = time.time()
            if run.concurrency <= 1:
                # Checkpoint per (length, task) block so a preemption keeps the
                # blocks already done; a requeue re-runs only the missing ones.
                # This is what lets the long eager base runs survive lowprio's
                # ~hourly preemption, which the per-run-only resume could not.
                ds_partial = os.path.join(partial_dir, ds.name)
                os.makedirs(ds_partial, exist_ok=True)
                results = []
                ran_any = False
                blocks = [(l, t) for l in ds.lengths for t in ds.tasks]
                for i, (length, task) in enumerate(blocks):
                    metas = _metas(length, task)
                    bpath = os.path.join(ds_partial, f"L{length}__{task}.json")
                    # Re-run the final block fresh even if cached, so the serve
                    # always processes >=1 block and this attempt's instrument /
                    # lost-attention dumps are non-empty.
                    force_block = (i == len(blocks) - 1) and not ran_any
                    block, cached = _run_block(
                        args.port, metas, args.request_timeout,
                        run.concurrency, bpath, force=force_block)
                    ran_any = ran_any or not cached
                    results.extend(block)
                    print(f"[{run.id}] block L{length} {task}: "
                          f"{'cached' if cached else 'ran'} ({len(block)})",
                          flush=True)
            else:
                metas = [m for length in ds.lengths for task in ds.tasks
                         for m in _metas(length, task)]
                with cf.ThreadPoolExecutor(max_workers=run.concurrency) as pool:
                    results = list(pool.map(
                        lambda m: _one_request(args.port, m, args.request_timeout),
                        metas,
                    ))
            wall = time.time() - wall0

            per_task = {}
            for t in ds.tasks:
                rs = [r["recall"] for r in results if r["task"] == t]
                if rs:
                    per_task[t] = round(sum(rs) / len(rs), 4)
            mean = (round(sum(per_task.values()) / len(per_task), 4)
                    if per_task else None)
            accuracy[ds.name] = {
                "per_task": per_task, "mean": mean, "n_req": len(results)}

            if run.concurrency <= 1:
                pf = [r["prefill_ms"] for r in results]
                dm = [r["decode_ms"] for r in results]
                dts = [(r["out_tok"] - 1) / r["decode_s"]
                       for r in results if r["out_tok"] and r["decode_s"] > 0]
                latency = {
                    "prefill_ttft_ms": round(sum(pf) / len(pf), 1) if pf else None,
                    "decode_ms": round(sum(dm) / len(dm), 1) if dm else None,
                    "decode_tok_s": round(sum(dts) / len(dts), 2) if dts else None,
                }
            else:
                out_tok = sum(r["out_tok"] or 0 for r in results)
                throughput = {
                    "concurrency": run.concurrency, "requests": len(results),
                    "out_tok": out_tok, "wall_s": round(wall, 2),
                    "aggregate_tok_s": round(out_tok / wall, 2) if wall > 0 else None,
                }
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=90)
        except Exception:
            proc.kill()
        time.sleep(5)  # let the instrument + ivf_fidelity atexit dumps settle

    verdict = {m: "expected_but_missing" for m in run.metrics()}
    verdict["accuracy"] = "PASS" if accuracy else "MISSING"
    if "latency_ttft" in verdict:
        verdict["latency_ttft"] = "PASS" if latency else "MISSING"
    if "throughput" in verdict:
        verdict["throughput"] = "PASS" if throughput else "MISSING"
    timing = _summarize_timing(timing_dir) if run.eager else None
    if run.eager:
        verdict["per_layer_indexer_ms"] = (
            "PASS" if timing["per_layer_ms"] else "MISSING")
        verdict["peak_indexer_mib"] = (
            "PASS" if timing["peak_indexer_mib"] > 0 else "MISSING")
        verdict["resident_state_mib"] = (
            "PASS" if timing["resident_state_mib"] > 0 else "MISSING")
        if "prefill_cluster_assign_split" in verdict:
            split = any("cluster" in k or "assign" in k
                        for k in timing["per_layer_ms"])
            verdict["prefill_cluster_assign_split"] = "PASS" if split else "MISSING"
    lost = _summarize_lost(lost_dir) if run.measure_lost_attention else None
    if "lost_attention" in verdict:
        verdict["lost_attention"] = "PASS" if lost else "MISSING"

    with open(os.path.join(base_out, "accuracy.json"), "w") as f:
        json.dump(accuracy, f, indent=2)
    if latency is not None:
        with open(os.path.join(base_out, "latency.json"), "w") as f:
            json.dump(latency, f, indent=2)
    if throughput is not None:
        with open(os.path.join(base_out, "throughput.json"), "w") as f:
            json.dump(throughput, f, indent=2)
    summary = {
        "id": run.id, "indexer": run.indexer, "exec": run.exec_mode,
        "concurrency": run.concurrency, "length": run.length,
        "max_model_len": run.max_model_len, "hf_overrides": run.hf_overrides,
        "accuracy": accuracy, "latency": latency, "throughput": throughput,
        "memory_timing": timing, "lost_attention": lost,
        "skipped": skipped, "verdict": verdict,
    }
    with open(os.path.join(base_out, "verdict.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{run.id}] done -> {base_out}  verdict={verdict}", flush=True)
    return summary


def main() -> None:
    default_yaml = os.path.join(BENCH_DIR, "runs.yaml")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="local path to GLM-5.3-Flash")
    p.add_argument("--ruler-dir", required=True,
                   help="<task>/validation_<len>.jsonl root")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--runs-yaml", default=default_yaml)
    p.add_argument("--only", help="comma list of run ids to run (default: all)")
    p.add_argument("--force", action="store_true",
                   help="re-run every run even if its verdict.json exists "
                        "(default: resume — skip already-finished runs)")
    p.add_argument("--tp", type=int, required=True)
    p.add_argument("--gpu-mem-util", type=float, default=0.9)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--startup-timeout", type=float, default=45 * 60)
    p.add_argument("--request-timeout", type=float, default=2 * _HOUR)
    args = p.parse_args()

    matrix = load_matrix(args.runs_yaml)
    runs = matrix.runs
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        bad = want - {r.id for r in runs}
        if bad:
            raise SystemExit(f"unknown run id(s): {sorted(bad)}")
        runs = [r for r in runs if r.id in want]

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"==== A/B matrix: {len(runs)} run(s) tp={args.tp} "
          f"(from {args.runs_yaml}) ====", flush=True)
    summary = {}
    for run in runs:
        # Resume: a run is finished iff it wrote a readable verdict.json (the
        # last write in run_one), so requeue after preemption re-runs only the
        # runs that did not complete. --force overrides.
        done = os.path.join(args.out_dir, run.id, "verdict.json")
        prior = None
        if not args.force and os.path.exists(done):
            try:
                with open(done) as f:
                    prior = json.load(f)["verdict"]
            except Exception:
                prior = None  # truncated by a mid-write preemption -> re-run
        if prior is not None:
            summary[run.id] = prior
            print(f"[{run.id}] resume: already complete, skipping", flush=True)
        else:
            summary[run.id] = run_one(run, args)["verdict"]
        with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
    print(f"\nwrote {os.path.join(args.out_dir, 'summary.json')}", flush=True)


if __name__ == "__main__":
    main()
