# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment 1: the highest gpu_memory_utilization each arm survives at load.

For each arm (kpool, IVF at the configured probe shares) and length, bisect
``--gpu-memory-utilization`` over [util_lo, util_hi] in util_step steps,
assuming a setting that passes also passes every lower one. At each setting:

1. serve (graph mode, the matrix's flags) and read the KV-cache size it reports;
2. send floor(KV tokens / max_model_len) full-length RULER requests at once;
3. pass iff every request finishes. Out of memory, a dead engine, a failed
   request or a hang (stacks dumped first) is a fail.

Outputs under ``--out-dir``:

    <cell>/config.json    settings the points are valid for (resume guard)
    <cell>/points.jsonl   one record per setting tried
    <cell>/serve_u<util>_a<attempt>.log   the serve's own log
    <cell>/verdict.json   highest passing util, one step lower as the safe
                          value, KV tokens and requests at each
    summary.json          every cell

Every point is appended as soon as it is measured, so a requeued job resumes
and re-runs at most the point in flight.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import subprocess
import sys
import threading
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
    max_model_len,
    max_tokens,
    round_robin_samples,
    serve_cmd,
    serve_env,
)

_KV_RE = re.compile(r"GPU KV cache size: ([\d,]+) tokens")
_CONC_RE = re.compile(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x")
_OOM_RE = re.compile(r"CUDA out of memory|OutOfMemoryError|out of memory", re.I)
_DEAD_RE = re.compile(r"EngineDeadError|WorkerProc hit an exception|"
                      r"EngineCore encountered a fatal error")


def cells(cfg: dict) -> list[dict]:
    """Arm x length, shortest length first, kpool first within a length."""
    c = cfg["capacity"]
    out = []
    for length in c["lengths"]:
        grid = cfg["ivf_grid"][length]
        out.append({"id": f"kpool_L{length}", "indexer": "kpool", "length": length,
                    "clusters": None, "probes": None})
        for p in grid["probes"][: c["ivf_probe_shares"]]:
            out.append({"id": f"ivf_C{grid['clusters']}_P{p}_L{length}",
                        "indexer": "ivf", "length": length,
                        "clusters": grid["clusters"], "probes": p})
    return out


def util_grid(c: dict) -> list[float]:
    n = round((c["util_hi"] - c["util_lo"]) / c["util_step"])
    return [round(c["util_lo"] + i * c["util_step"], 4) for i in range(n + 1)]


def _cell_config(cfg: dict, cell: dict) -> str:
    c = cfg["capacity"]
    return json.dumps({
        "cell": cell, "tp": c["tp"], "tasks": c["tasks"],
        "max_model_len": max_model_len(cfg, cell["length"]),
        "seed": cfg["fairness"]["seed"], "exec": "graph", "thinking": False,
    }, sort_keys=True)


def _grep(path: str, regex: re.Pattern):
    try:
        with open(path, errors="replace") as f:
            hits = regex.findall(f.read())
    except OSError:
        return []
    return hits


class _GpuPeak:
    """Max memory.used per visible GPU, polled while the stress runs."""

    def __init__(self) -> None:
        self.peak: list[int] = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=30).stdout
                used = [int(x) for x in out.split()]
                self.peak = ([max(a, b) for a, b in zip(self.peak, used)]
                             if len(self.peak) == len(used) else used)
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            self._stop.wait(5)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=40)


def _request(port: int, sample: dict, cfg: dict, timeout: float) -> dict:
    body = {
        "model": "glm",
        **lar.ruler_messages(sample["input"], sample["answer_prefix"]),
        "max_tokens": max_tokens(cfg, sample["task"]),
        "temperature": 0,
        "seed": cfg["fairness"]["seed"],
    }
    t0 = time.time()
    reply = lar._post_with_retry(
        f"http://127.0.0.1:{port}/v1/chat/completions", body, timeout, 1)
    return {"label": sample["label"], "ttft_s": round(reply.ttft_s, 2),
            "total_s": round(time.time() - t0, 2),
            "out_tok": reply.usage.get("completion_tokens"),
            "finish_reason": reply.finish_reason}


def run_point(cfg: dict, cell: dict, util: float, attempt: int, cell_dir: str,
              args) -> dict:
    """One serve at ``util`` and one burst of full-length requests."""
    c = cfg["capacity"]
    mml = max_model_len(cfg, cell["length"])
    log_path = os.path.join(cell_dir, f"serve_u{util:.2f}_a{attempt}.log")
    cmd = serve_cmd(cfg, length=cell["length"], tp=c["tp"], port=args.port,
                    util=util, indexer=cell["indexer"], clusters=cell["clusters"],
                    probes=cell["probes"], chat_template=args.chat_template)
    lar._log(f"[{cell['id']}] util {util:.2f} attempt {attempt}: {' '.join(cmd)}")
    rec: dict = {"util": util, "attempt": attempt, "max_model_len": mml}
    t0 = time.time()
    with open(log_path, "a") as log:
        proc = subprocess.Popen(
            cmd, env=serve_env(indexer=cell["indexer"], clusters=cell["clusters"],
                               probes=cell["probes"]),
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    hung = False
    try:
        try:
            lar._wait_ready(proc, args.port, c["startup_timeout_s"])
        except (RuntimeError, TimeoutError) as e:
            hung = isinstance(e, TimeoutError)
            oom = bool(_grep(log_path, _OOM_RE))
            rec.update(passed=False, phase="startup",
                       reason="oom" if oom else ("hang" if hung else "startup_failed"),
                       error=repr(e))
            return rec
        rec["startup_s"] = round(time.time() - t0, 1)
        kv = _grep(log_path, _KV_RE)
        if not kv:
            rec.update(passed=False, phase="startup", reason="no_kv_size_in_log")
            return rec
        rec["kv_tokens"] = int(kv[-1].replace(",", ""))
        conc = _grep(log_path, _CONC_RE)
        if conc:
            rec["server_max_concurrency"] = float(conc[-1][1])
        n = max(1, rec["kv_tokens"] // mml)
        rec["requests"] = n
        samples = round_robin_samples(cfg, cell["length"], c["tasks"], n)
        lar._log(f"[{cell['id']}] util {util:.2f}: {rec['kv_tokens']:,} KV tokens "
                 f"-> {n} x {cell['length']} at once")
        t1 = time.time()
        pool = cf.ThreadPoolExecutor(max_workers=n)
        with _GpuPeak() as peak:
            futs = [pool.submit(_request, args.port, s, cfg, c["request_timeout_s"])
                    for s in samples]
            done, errors = [], []
            try:
                for fut in cf.as_completed(futs, timeout=c["request_timeout_s"] + 60):
                    try:
                        done.append(fut.result())
                    except Exception as e:
                        errors.append(repr(e))
            except cf.TimeoutError:
                hung = True
            finally:
                pool.shutdown(wait=False, cancel_futures=True)
        rec["stress_s"] = round(time.time() - t1, 1)
        rec["peak_gpu_mib"] = peak.peak
        rec["finished"] = len(done)
        rec["results"] = done
        alive = proc.poll() is None
        oom = bool(_grep(log_path, _OOM_RE))
        dead = bool(_grep(log_path, _DEAD_RE)) or not alive
        passed = len(done) == n and not errors and not hung and alive and not oom
        reason = None
        if not passed:
            reason = ("oom" if oom else "engine_dead" if dead
                      else "hang" if hung else "request_failed")
        rec.update(passed=passed, phase="stress", reason=reason,
                   errors=errors[:5])
        return rec
    finally:
        lar._stop_serve(proc, dump_stacks=hung)


class _Points:
    def __init__(self, path: str):
        self.path = path
        self.by_util: dict[float, list[dict]] = {}
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    self.by_util.setdefault(r["util"], []).append(r)

    def add(self, rec: dict) -> None:
        self.by_util.setdefault(rec["util"], []).append(rec)
        with open(self.path, "a") as f:
            f.write(f"\n{json.dumps(rec)}\n")


def run_cell(cfg: dict, cell: dict, args) -> dict:
    c = cfg["capacity"]
    cell_dir = os.path.join(args.out_dir, cell["id"])
    os.makedirs(cell_dir, exist_ok=True)
    config_path = os.path.join(cell_dir, "config.json")
    text = _cell_config(cfg, cell)
    if os.path.exists(config_path):
        with open(config_path) as f:
            if f.read() != text:
                raise SystemExit(f"{config_path} is from other settings; "
                                 "move it away rather than mix results")
    else:
        lar._write_atomic(config_path, text)
    points = _Points(os.path.join(cell_dir, "points.jsonl"))

    def passes(util: float) -> bool:
        recs = points.by_util.get(util, [])
        while True:
            if any(r["passed"] for r in recs):
                return True
            # A memory failure is the answer; anything else is retried.
            if any(r.get("reason") == "oom" for r in recs) or (
                    len(recs) >= c["max_attempts_per_point"]):
                return False
            rec = run_point(cfg, cell, util, len(recs) + 1, cell_dir, args)
            points.add(rec)
            lar._log(f"[{cell['id']}] util {util:.2f}: "
                     f"{'PASS' if rec['passed'] else 'FAIL ' + str(rec['reason'])}")
            recs = points.by_util[util]

    grid = util_grid(c)
    lo, hi = 0, len(grid) - 1
    if passes(grid[hi]):
        best = hi
    elif not passes(grid[lo]):
        best = None
    else:
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if passes(grid[mid]):
                lo = mid
            else:
                hi = mid
        best = lo

    def summary_at(i):
        if i is None or i < 0:
            return None
        recs = [r for r in points.by_util.get(grid[i], []) if r["passed"]]
        r = recs[-1] if recs else None
        return None if r is None else {
            "util": grid[i], "kv_tokens": r["kv_tokens"], "requests": r["requests"],
            "server_max_concurrency": r.get("server_max_concurrency"),
            "peak_gpu_mib": r.get("peak_gpu_mib")}

    verdict = {
        **cell, "max_model_len": max_model_len(cfg, cell["length"]),
        "max_pass": summary_at(best),
        # The safe value is one step lower; it was not necessarily served.
        "safe_util": None if best is None else grid[max(best - 1, 0)],
        "first_fail": None if best is None or best == len(grid) - 1 else {
            "util": grid[best + 1],
            "reason": [r.get("reason") for r in points.by_util[grid[best + 1]]]},
        "points_tried": sorted(points.by_util),
    }
    lar._write_atomic(os.path.join(cell_dir, "verdict.json"),
                      json.dumps(verdict, indent=2))
    return verdict


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--runs-yaml", default=os.path.join(EXP_DIR, "runs.yaml"))
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--only", help="comma list of cell ids")
    p.add_argument("--dry-run", action="store_true",
                   help="print cells, grid and serve commands; start nothing")
    args = p.parse_args()
    cfg = load(args.runs_yaml)
    todo = cells(cfg)
    if args.only:
        want = set(args.only.split(","))
        todo = [x for x in todo if x["id"] in want]
    if args.dry_run:
        print("util grid:", util_grid(cfg["capacity"]))
        for cell in todo:
            print(cell["id"], " ".join(serve_cmd(
                cfg, length=cell["length"], tp=cfg["capacity"]["tp"],
                port=args.port, util=0.9, indexer=cell["indexer"],
                clusters=cell["clusters"], probes=cell["probes"])))
        return
    lar.THINKING = False
    lar._exit_on_sigterm()
    os.makedirs(args.out_dir, exist_ok=True)
    args.chat_template = nothink_template(cfg["model_dir"], args.out_dir)
    summary_path = os.path.join(args.out_dir, "summary.json")
    summary = {}
    for cell in todo:
        done = os.path.join(args.out_dir, cell["id"], "verdict.json")
        if os.path.exists(done):
            with open(done) as f:
                summary[cell["id"]] = json.load(f)
            lar._log(f"[{cell['id']}] already complete, skipping")
        else:
            summary[cell["id"]] = run_cell(cfg, cell, args)
        lar._write_atomic(summary_path, json.dumps(summary, indent=2))
    lar._log(f"[capacity] all cells complete -> {summary_path}")


if __name__ == "__main__":
    main()
