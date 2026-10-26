# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure IVF lost-attention per RULER length bucket (no Modal).

Serves the IVF arm of GLM-5.3-Flash with
``VLLM_DSA_IVF_MEASURE_LOST_ATTENTION=1`` and runs ``--n`` samples of one RULER
task through it, **one length bucket at a time**. Each bucket gets its own fresh
serve and its own ``VLLM_DSA_IVF_LOST_ATTENTION_DIR`` so the per-(layer, phase)
recall/mass stats are isolated per bucket rather than blended across lengths
(the recorder accumulates globally within a process).

It assumes the model and the RULER jsonl are already staged on local disk and a
working vLLM+fork env is active; ``_run_body.sh`` does that staging.

Per cell it writes ``<out>/L<len>/C<clusters>_P<probes>/``:
    verdict.json         per-task recall + lost-attention + memory/timing summary
    lost_attention/      the raw per-rank lost_attention_pid*.json dumps
    timing/              the raw per-rank indexer_timing_pid*.json dumps

and a top-level ``<out>/summary.json`` over all cells.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

import yaml

BENCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BENCH_DIR)
from tasks import TASKS, recall  # noqa: E402

# Serve must cover the longest bucket's context plus the answer budget.
_LEN_MARGIN = 4096


def _get(url: str, timeout: float) -> int:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status


def _post_stream(url: str, body: dict, timeout: float):
    """POST a streamed chat completion; return (ttft_s, decode_s, text, usage)."""
    body = {**body, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    ttft = None
    t_last = t0
    content = reasoning = ""
    usage: dict = {}
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices", []):
                d = ch.get("delta") or {}
                piece = (d.get("content") or "") + (d.get("reasoning_content") or "")
                if piece:
                    now = time.time()
                    if ttft is None:
                        ttft = now - t0
                    t_last = now
                content += d.get("content") or ""
                reasoning += d.get("reasoning_content") or ""
    decode_s = max(t_last - t0 - (ttft or 0), 0.0)
    return ttft or 0.0, decode_s, content + " " + reasoning, usage


def _serve_env(lost_dir: str, timing_dir: str) -> dict:
    """IVF arm with lost-attention measurement and the memory/timing instrument
    on (the serve is eager, so the instrument can time and probe per layer)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = BENCH_DIR + (
        ":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["VLLM_ENABLE_DSA_IVF_INDEXER"] = "1"
    env["VLLM_DSA_IVF_MEASURE_LOST_ATTENTION"] = "1"
    env["VLLM_DSA_IVF_LOST_ATTENTION_DIR"] = lost_dir
    env["GLM5_INSTRUMENT"] = "1"
    env["GLM5_TIMING_DIR"] = timing_dir
    env["GLM5_MEM_PROBE"] = "1"
    env["GLM5_RESIDENT_PROBE"] = "1"
    return env


def _summarize_timing(timing_dir: str) -> dict:
    """Peak/resident indexer memory and per-(layer, phase) mean indexer time,
    taken across the per-rank instrument dumps."""
    peak = resident = 0.0
    layers: dict[str, float] = {}
    for fn in glob.glob(os.path.join(timing_dir, "indexer_timing_pid*_r*.json")):
        try:
            rep = json.load(open(fn))
        except Exception:
            continue
        for layer, phases in rep.get("layers", {}).items():
            for phase, s in phases.items():
                peak = max(peak, s.get("peak_mib", 0.0))
                resident = max(resident, s.get("resident_mib", 0.0))
                key = f"{layer}|{phase}"
                layers[key] = max(layers.get(key, 0.0), s.get("mean_ms", 0.0))
    return {
        "peak_indexer_mib": round(peak, 2),
        "resident_state_mib": round(resident, 2),
        "per_layer_ms": {k: round(v, 4) for k, v in sorted(layers.items())},
    }


def _serve(
    model: str,
    tp: int,
    max_len: int,
    port: int,
    clusters: int,
    probes: int,
    gpu_mem_util: float,
) -> list[str]:
    return [
        shutil.which("vllm") or "vllm", "serve", model,
        "--served-model-name", "glm",
        "--tensor-parallel-size", str(tp),
        "--trust-remote-code",
        "--max-model-len", str(max_len),
        "--gpu-memory-utilization", str(gpu_mem_util),
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--port", str(port),
        "--hf-overrides", json.dumps(
            {"index_num_clusters": clusters, "index_num_probes": probes}
        ),
    ]


def _wait_ready(proc: subprocess.Popen, port: int, timeout_s: float) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(f"vllm exited during startup (code {proc.returncode})")
        try:
            if _get(f"http://127.0.0.1:{port}/health", timeout=5) == 200:
                return
        except Exception:
            pass
        time.sleep(10)
    raise TimeoutError(f"serve not ready within {timeout_s:.0f}s")


def _summarize_lost(lost_dir: str) -> dict:
    """Mean key_lost / mass_lost per (layer, phase) over the per-rank dumps."""
    agg: dict[str, dict] = {}
    for fn in glob.glob(os.path.join(lost_dir, "lost_attention_pid*.json")):
        try:
            d = json.load(open(fn))
        except Exception:
            continue
        for key, s in d.items():
            a = agg.setdefault(key, {"rows": 0, "key_lost": 0.0, "mass_lost": 0.0})
            rows = s.get("rows", 0)
            a["rows"] += rows
            a["key_lost"] += s.get("key_lost_mean", 0.0) * rows
            a["mass_lost"] += s.get("mass_lost_mean", 0.0) * rows
    return {
        key: {
            "rows": a["rows"],
            "key_lost_mean": a["key_lost"] / a["rows"] if a["rows"] else 0.0,
            "mass_lost_mean": a["mass_lost"] / a["rows"] if a["rows"] else 0.0,
        }
        for key, a in agg.items()
    }


def _run_task(
    port: int, task: str, rows: list, request_timeout: float,
    concurrency: int, max_tokens: int, seed: int | None, greedy: bool,
) -> list:
    """Stream ``task``'s samples through the live serve; return per-sample recs.

    ``max_tokens``, ``seed`` and ``greedy`` come from the ``fairness`` block of
    ``runs.yaml`` (the single source of truth both benchmark arms share), not
    from the per-task caps in ``tasks.py``, so the fidelity sweep uses the same
    generation budget and decoding settings as the A/B matrix.

    With ``concurrency`` > 1 the samples are issued through a thread pool so the
    server's continuous batching runs them together. This is metric-neutral:
    recall is per-request and the per-(layer, phase) key/mass_lost stats are
    aggregated per row, so batch composition does not change them. Every sample
    here shares one task, hence identical length and clusters/probes. It does
    blur the per-request ``prefill_ms``/``decode_ms`` (requests share a step);
    the authoritative latency/timing come from the A/B matrix, not this sweep.
    """
    url = f"http://127.0.0.1:{port}/v1/chat/completions"

    def _one(ex: dict) -> dict:
        body = {
            "model": "glm",
            "messages": [{"role": "user", "content": ex["input"]}],
            "max_tokens": max_tokens,
            "temperature": 0 if greedy else 1.0,
        }
        if seed is not None:
            body["seed"] = seed
        ttft, dec_s, text, usage = _post_stream(
            url, body, timeout=request_timeout,
        )
        return {
            "idx": ex["index"],
            "gold": ex["outputs"],
            "recall": recall(ex["outputs"], text),
            "prefill_ms": round(ttft * 1000, 1),
            "decode_ms": round(dec_s * 1000, 1),
            "in_tok": usage.get("prompt_tokens"),
            "out_tok": usage.get("completion_tokens"),
        }

    if concurrency <= 1:
        return [_one(ex) for ex in rows]
    with cf.ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(_one, rows))


def run_cell(
    length: int, clusters: int, probes: int, pct: float,
    tasks: list[str], n: int, args,
) -> dict:
    """Serve one (length, probe) cell once, run every task through it.

    Tasks are a free axis (same clusters/probes), so they share the serve; the
    cell has its own lost-attention dir, so its per-(layer, phase) stats are
    isolated from other cells. Lost attention is aggregated over all the cell's
    tasks; recall is reported per task.
    """
    tag = f"L{length} C{clusters} P{probes}({pct:g}%)"
    cell_out = os.path.join(args.out_dir, f"L{length}", f"C{clusters}_P{probes}")
    # Drop any partial dir from a preempted attempt so stale per-rank dumps
    # (globbed by the summarizers) don't bleed into this cell's numbers.
    if os.path.isdir(cell_out):
        shutil.rmtree(cell_out)
    lost_dir = os.path.join(cell_out, "lost_attention")
    timing_dir = os.path.join(cell_out, "timing")
    os.makedirs(lost_dir, exist_ok=True)
    os.makedirs(timing_dir, exist_ok=True)

    max_len = length + args.max_tokens + _LEN_MARGIN
    cmd = _serve(
        args.model, args.tp, max_len, args.port, clusters, probes, args.gpu_mem_util
    )
    print(f"[{tag}] serve: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, env=_serve_env(lost_dir, timing_dir))
    per_task: dict[str, dict] = {}
    try:
        t0 = time.time()
        _wait_ready(proc, args.port, args.startup_timeout)
        print(f"[{tag}] ready in {time.time() - t0:.0f}s", flush=True)

        cap = None if n in ("all", None, 0) else int(n)
        for task in tasks:
            jsonl = os.path.join(args.ruler_dir, task, f"validation_{length}.jsonl")
            rows = [json.loads(line) for line in open(jsonl)][:cap]
            if not rows:
                raise RuntimeError(f"no samples in {jsonl}")
            samples = _run_task(
                args.port, task, rows, args.request_timeout, args.concurrency,
                args.max_tokens, args.seed, args.greedy,
            )
            rmean = sum(s["recall"] for s in samples) / len(samples)
            per_task[task] = {"recall_mean": rmean, "samples": samples}
            print(f"[{tag}] {task:16s} recall_mean={rmean:.3f} n={len(samples)}",
                  flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=90)
        except Exception:
            proc.kill()
        time.sleep(5)  # let the recorder's atexit dump settle

    lost = _summarize_lost(lost_dir)
    timing = _summarize_timing(timing_dir)
    recalls = [t["recall_mean"] for t in per_task.values()]
    verdict = {
        "length": length,
        "clusters": clusters,
        "probes": probes,
        "probe_pct": pct,
        "n": n,
        "recall_mean": sum(recalls) / len(recalls) if recalls else 0.0,
        "tasks": per_task,
        "lost_attention": lost,
        "memory_timing": timing,
    }
    with open(os.path.join(cell_out, "verdict.json"), "w") as f:
        json.dump(verdict, f, indent=2)
    return verdict


def _load_sweep(path: str) -> dict:
    """Read the IVF ``sweep.cells`` block (n, tasks, grid) from runs.yaml.

    ``tasks: all`` resolves to the full ``ruler`` dataset task list, so the
    task set stays defined in one place.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cells = next(
        ((r.get("sweep") or {}).get("cells") for r in cfg.get("runs", [])
         if (r.get("sweep") or {}).get("cells")),
        None,
    )
    if not cells:
        raise SystemExit(f"{path}: no ivf sweep.cells block")
    tasks = cells.get("tasks", "all")
    if tasks in ("all", None):
        tasks = cfg["datasets"]["ruler"]["tasks"]
    nval = cells.get("n", 5)
    n = nval if nval in ("all", None) else int(nval)
    return {"n": n, "tasks": list(tasks), "grid": cells["grid"]}


def _load_fairness(path: str) -> dict:
    """Read the shared ``fairness`` block from runs.yaml.

    These request-level knobs (generation budget, seed, decoding) are the same
    for both benchmark arms, so the fidelity sweep derives them here instead of
    hardcoding them or reading the per-task caps in ``tasks.py``.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("fairness", {}) or {}


def main() -> None:
    default_yaml = os.path.join(BENCH_DIR, "runs.yaml")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="local path to GLM-5.3-Flash")
    p.add_argument("--ruler-dir", required=True,
                   help="<task>/validation_<len>.jsonl root")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--runs-yaml", default=default_yaml,
                   help="source of the explicit ivf sweep.cells grid")
    p.add_argument("--task", help="restrict to one task (default: all ruler tasks)")
    p.add_argument("--n", type=int, help="override samples per task per cell")
    p.add_argument("--max-tokens", type=int, default=None,
                   help="generation cap per request; default from runs.yaml "
                        "fairness.max_tokens (overrides tasks.py per-task caps)")
    p.add_argument("--concurrency", type=int, default=1,
                   help="requests in flight per task (ThreadPool); the server "
                        "batches them. >1 speeds the sweep and leaves recall and "
                        "key/mass_lost unchanged, but blurs per-request latency.")
    p.add_argument("--lengths", help="comma list to restrict to a subset of cells")
    p.add_argument("--force", action="store_true",
                   help="re-run every cell even if its verdict.json exists "
                        "(default: resume — skip already-finished cells)")
    p.add_argument("--tp", type=int, required=True)
    p.add_argument("--gpu-mem-util", type=float, default=0.9)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--startup-timeout", type=float, default=45 * 60)
    p.add_argument("--request-timeout", type=float, default=3600)
    args = p.parse_args()

    sweep = _load_sweep(args.runs_yaml)
    fairness = _load_fairness(args.runs_yaml)
    if args.max_tokens is None:
        args.max_tokens = int(fairness.get("max_tokens", 2048))
    seed = fairness.get("seed")
    args.seed = int(seed) if seed is not None else None
    args.greedy = bool(fairness.get("greedy", True))
    tasks = [args.task] if args.task else sweep["tasks"]
    n = args.n or sweep["n"]
    bad = [t for t in tasks if t not in TASKS]
    if bad:
        raise SystemExit(f"unknown task(s): {bad}")
    grid = sweep["grid"]
    if args.lengths:
        keep = {int(x) for x in args.lengths.split(",") if x}
        grid = [c for c in grid if int(c["length"]) in keep]

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"==== IVF lost-attention: tasks={len(tasks)} n={n} tp={args.tp} "
          f"concurrency={args.concurrency} max_tokens={args.max_tokens} "
          f"seed={args.seed} greedy={args.greedy} "
          f"cells={sum(len(c['probes']) for c in grid)} (from {args.runs_yaml}) ====",
          flush=True)

    summary = []
    for cell in grid:
        length = int(cell["length"])
        clusters = int(cell["clusters"])
        for probes in cell["probes"]:
            probes = int(probes)
            pct = round(100.0 * probes / clusters, 2)
            # Resume: a cell is finished iff it wrote a readable verdict.json
            # (run_cell's last write), so requeue after preemption re-runs only
            # the cells that did not complete. --force overrides.
            done = os.path.join(
                args.out_dir, f"L{length}", f"C{clusters}_P{probes}",
                "verdict.json")
            v = None
            if not args.force and os.path.exists(done):
                try:
                    with open(done) as f:
                        v = json.load(f)
                    print(f"[L{length} C{clusters} P{probes}({pct:g}%)] "
                          "resume: already complete, skipping", flush=True)
                except Exception:
                    v = None  # truncated by a mid-write preemption -> re-run
            if v is None:
                v = run_cell(length, clusters, probes, pct, tasks, n, args)
            summary.append(v)
            print(f"[L{length} C{clusters} P{probes}({pct:g}%)] "
                  f"recall_mean={v['recall_mean']:.3f} cells_done={len(summary)}",
                  flush=True)
            for key, s in sorted(v["lost_attention"].items()):
                print(f"    {key:16s} key_lost={s['key_lost_mean']:.4f} "
                      f"mass_lost={s['mass_lost_mean']:.4f} rows={s['rows']}",
                      flush=True)

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {os.path.join(args.out_dir, 'summary.json')}", flush=True)


if __name__ == "__main__":
    main()
