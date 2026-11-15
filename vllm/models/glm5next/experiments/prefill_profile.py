# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment 4: where the time of one served prefill goes.

Per cell (arm x length; IVF at one probe share of ivf_grid):

1. serve with the matrix's flags (graph mode, TP=4) plus vLLM's torch profiler
   (``--profiler-config``, CPU + CUDA activities, no stacks);
2. one short warmup request (JIT kernels, autotuning), then ``timed_requests``
   full-length requests one at a time, unprofiled: their TTFT is the prefill
   time;
3. ``/start_profile``, one more full-length request, ``/stop_profile``: each TP
   rank writes a Chrome trace of every kernel of that prefill;
4. ``profile_breakdown.py`` splits rank 0's kernel time by model component.

Requests ask for one token, so TTFT is the whole request. Outputs under
``--out-dir``:

    <cell>/serve.log         the serve's own log
    <cell>/trace/            per-rank traces and kernel tables from the profiler
    <cell>/breakdown.json    rank 0's time by component (profile_breakdown.py)
    <cell>/result.json       TTFTs and the breakdown's headline; marks the cell done
    summary.json             every cell

A cell with result.json is skipped, so a requeued job resumes at the cell in
flight.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.join(os.path.dirname(EXP_DIR), "benchmark")
sys.path.insert(0, os.path.join(BENCH_DIR, "slurm"))
sys.path.insert(0, BENCH_DIR)
sys.path.insert(0, EXP_DIR)

import lost_attention_run as lar  # noqa: E402
import profile_breakdown  # noqa: E402
from bench_config import nothink_template  # noqa: E402
from exp_config import load, ruler_samples, serve_cmd, serve_env  # noqa: E402


def cells(cfg: dict) -> list[dict]:
    """kpool then IVF at each length, shortest length first."""
    p = cfg["profile"]
    out = []
    for length in p["lengths"]:
        grid = cfg["ivf_grid"][length]
        probes = grid["probes"][p["ivf_probe_index"]]
        if "kpool" in p["arms"]:
            out.append({"id": f"kpool_L{length}", "indexer": "kpool", "length": length,
                        "clusters": None, "probes": None})
        if "ivf" in p["arms"]:
            out.append({"id": f"ivf_C{grid['clusters']}_P{probes}_L{length}",
                        "indexer": "ivf", "length": length,
                        "clusters": grid["clusters"], "probes": probes})
    return out


def _profiler_config(trace_dir: str) -> str:
    return json.dumps({
        "profiler": "torch",
        "torch_profiler_dir": trace_dir,
        "torch_profiler_with_stack": False,
        "torch_profiler_record_shapes": False,
        "torch_profiler_use_gzip": True,
        "torch_profiler_dump_cuda_time_total": True,
        "ignore_frontend": True,
    })


def _post(port: int, path: str, timeout: float) -> None:
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=b"{}",
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if r.status != 200:
            raise RuntimeError(f"{path} returned {r.status}")


def _request(port: int, sample: dict, cfg: dict, timeout: float) -> dict:
    body = {"model": "glm",
            **lar.ruler_messages(sample["input"], sample["answer_prefix"]),
            "max_tokens": 1, "temperature": 0, "seed": cfg["fairness"]["seed"]}
    t0 = time.time()
    reply = lar._post_with_retry(
        f"http://127.0.0.1:{port}/v1/chat/completions", body, timeout, 1)
    return {"label": sample["label"], "ttft_s": round(reply.ttft_s, 3),
            "total_s": round(time.time() - t0, 3),
            "prompt_tokens": reply.usage.get("prompt_tokens")}


def _wait_traces(trace_dir: str, n: int, timeout_s: float) -> list[str]:
    """The ``n`` rank traces, once their sizes stop changing."""
    deadline = time.time() + timeout_s
    last: dict[str, int] = {}
    while time.time() < deadline:
        files = sorted(glob.glob(os.path.join(trace_dir, "*.pt.trace.json*")))
        sizes = {f: os.path.getsize(f) for f in files}
        if len(files) >= n and sizes == last:
            return files
        last = sizes
        time.sleep(30)
    raise TimeoutError(f"{len(last)} of {n} traces in {trace_dir} after {timeout_s:.0f}s")


def run_cell(cfg: dict, cell: dict, args) -> dict:
    p = cfg["profile"]
    cell_dir = os.path.join(args.out_dir, cell["id"])
    trace_dir = os.path.join(cell_dir, "trace")
    # A half-written trace dir from a preempted try is not this try's profile.
    shutil.rmtree(trace_dir, ignore_errors=True)
    os.makedirs(trace_dir)
    cmd = serve_cmd(cfg, length=cell["length"], tp=p["tp"], port=args.port,
                    util=p["util"], indexer=cell["indexer"], clusters=cell["clusters"],
                    probes=cell["probes"], chat_template=args.chat_template)
    cmd += ["--profiler-config", _profiler_config(trace_dir)]
    lar._log(f"[{cell['id']}] {' '.join(cmd)}")
    rec: dict = {**cell, "util": p["util"], "tp": p["tp"]}
    t0 = time.time()
    with open(os.path.join(cell_dir, "serve.log"), "a") as log:
        proc = subprocess.Popen(
            cmd, env=serve_env(indexer=cell["indexer"], clusters=cell["clusters"],
                               probes=cell["probes"]),
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        lar._wait_ready(proc, args.port, p["startup_timeout_s"])
        rec["startup_s"] = round(time.time() - t0, 1)
        n = p["timed_requests"] + 1
        samples = ruler_samples(cfg, cell["length"], [p["task"]], list(range(n)))
        warm = ruler_samples(cfg, p["warmup_length"], [p["task"]], [n])[0]
        rec["warmup"] = _request(args.port, warm, cfg, p["request_timeout_s"])
        rec["timed"] = []
        for s in samples[:-1]:
            rec["timed"].append(_request(args.port, s, cfg, p["request_timeout_s"]))
            lar._log(f"[{cell['id']}] timed {rec['timed'][-1]}")
        _post(args.port, "/start_profile", 600)
        rec["profiled"] = _request(args.port, samples[-1], cfg, p["request_timeout_s"])
        t1 = time.time()
        _post(args.port, "/stop_profile", p["trace_timeout_s"])
        traces = _wait_traces(trace_dir, p["tp"], p["trace_timeout_s"])
        rec["trace_write_s"] = round(time.time() - t1, 1)
        rec["traces"] = [os.path.basename(f) for f in traces]
        lar._log(f"[{cell['id']}] profiled {rec['profiled']}, "
                 f"{len(traces)} traces in {rec['trace_write_s']}s")
    finally:
        lar._stop_serve(proc, dump_stacks=False)
    ttfts = [r["ttft_s"] for r in rec["timed"]]
    rec["ttft_s_mean"] = round(sum(ttfts) / len(ttfts), 3)
    try:
        rank0 = profile_breakdown.pick_rank0(traces)
        bd = profile_breakdown.analyze(rank0)
        lar._write_atomic(os.path.join(cell_dir, "breakdown.json"),
                          json.dumps(bd, indent=2))
        rec["breakdown"] = {"trace": os.path.basename(rank0),
                            "kernel_ms": bd["kernel_ms"], "span_ms": bd["span_ms"],
                            "by_category": bd["by_category"]}
    except Exception as e:  # the trace is kept; the analysis can be rerun
        rec["breakdown_error"] = repr(e)
        lar._log(f"[{cell['id']}] breakdown failed: {e!r}")
    lar._write_atomic(os.path.join(cell_dir, "result.json"), json.dumps(rec, indent=2))
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--runs-yaml", default=os.path.join(EXP_DIR, "runs.yaml"))
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--only", help="comma list of cell ids")
    ap.add_argument("--dry-run", action="store_true",
                    help="print cells and serve commands; start nothing")
    args = ap.parse_args()
    cfg = load(args.runs_yaml)
    todo = cells(cfg)
    if args.only:
        want = set(args.only.split(","))
        todo = [c for c in todo if c["id"] in want]
    if args.dry_run:
        for c in todo:
            cmd = serve_cmd(cfg, length=c["length"], tp=cfg["profile"]["tp"],
                            port=args.port, util=cfg["profile"]["util"],
                            indexer=c["indexer"], clusters=c["clusters"],
                            probes=c["probes"])
            print(c["id"], " ".join(cmd + ["--profiler-config",
                                           _profiler_config("/abs/trace")]))
        return
    lar.THINKING = False
    lar._exit_on_sigterm()
    os.makedirs(args.out_dir, exist_ok=True)
    args.chat_template = nothink_template(cfg["model_dir"], args.out_dir)
    summary_path = os.path.join(args.out_dir, "summary.json")
    summary = {}
    for c in todo:
        done = os.path.join(args.out_dir, c["id"], "result.json")
        if os.path.exists(done):
            with open(done) as f:
                summary[c["id"]] = json.load(f)
            lar._log(f"[{c['id']}] already complete, skipping")
            continue
        try:
            summary[c["id"]] = run_cell(cfg, c, args)
        except Exception as e:  # one failed cell must not stop the others
            lar._log(f"[{c['id']}] FAILED: {e!r}")
            summary[c["id"]] = {**c, "error": repr(e)}
        lar._write_atomic(summary_path, json.dumps(summary, indent=2))
    lar._log(f"[profile] all cells complete -> {summary_path}")


if __name__ == "__main__":
    main()
