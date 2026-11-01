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
    verdict.json                    per-task recall + lost-attention summary
    config.json                     the settings the checkpoints are valid for
    partial/results.jsonl           per-request checkpoint
    partial/lost_attention/attempt<k>/   per-rank dumps of each serve

and a top-level ``<out>/summary.json`` over all cells.

A preempted job is requeued and resumes its cell: only unfinished requests are
re-sent and the lost-attention stats of every attempt are merged. Requests in
flight at preemption are measured again, so their rows count twice.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import http.client
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from typing import NamedTuple

import yaml

BENCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGGING_CONFIG = os.path.join(BENCH_DIR, "slurm", "fidelity_logging.json")
sys.path.insert(0, BENCH_DIR)
from tasks import TASKS, recall  # noqa: E402


class StreamError(OSError):
    """The stream ended without a complete answer, e.g. the engine died."""


_TRANSIENT = (OSError, http.client.HTTPException, ValueError)


def _log(msg: str) -> None:
    """Print to the job log; a failing log (shared storage) must not kill us."""
    try:
        print(msg, flush=True)
    except OSError:
        pass


def _key(rec: dict) -> str:
    return f"{rec['L']}|{rec['task']}|{rec['idx']}"


def _write_atomic(path: str, text: str) -> None:
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


class _Results:
    """Per-request checkpoint: an append-only jsonl, reloaded on resume.

    Every record is written between newlines, so a line torn by a preemption or a
    full disk is skipped on load and can't corrupt the next one. Records that
    fail to write stay queued and go out with the next one, so a storage blip
    costs no results while the run stays up.
    """

    def __init__(self, path: str):
        self.path = path
        self.done: dict[str, dict] = {}
        self._unsynced: list[dict] = []
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    self.done[_key(rec)] = rec
        self.attempt = 1 + max(
            (r["attempt"] for r in self.done.values()), default=0)

    def add(self, rec: dict) -> None:
        rec["attempt"] = self.attempt
        self.done[_key(rec)] = rec
        self._unsynced.append(rec)
        self.sync()

    def sync(self) -> None:
        text = "".join(f"\n{json.dumps(r)}\n" for r in self._unsynced)
        try:
            with open(self.path, "a") as f:
                f.write(text)
        except OSError as e:
            _log(f"checkpoint write failed, will retry: {e!r}")
            return
        self._unsynced.clear()


def _exit_on_sigterm() -> None:
    """Stop at once on preemption.

    The serve dies from the same signal, so without this its failed requests
    look like a run failure and the runner would move on to the next one.
    """
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))


class _Reply(NamedTuple):
    ttft_s: float
    decode_s: float
    answer: str
    reasoning: str
    usage: dict
    finish_reason: str | None


def _split_answer(
    content: str, reasoning: str, finish_reason: str | None,
) -> tuple[str, str]:
    """Return (final answer, reasoning) of a completion.

    The chat template opens ``<think>`` and we serve without a reasoning parser,
    so GLM-5.3's stream is ``reasoning</think>answer``. A stream cut by
    ``max_tokens`` before ``</think>`` has reasoning and no answer.
    """
    if reasoning:
        return content, reasoning
    if "</think>" in content:
        thought, answer = content.rsplit("</think>", 1)
        return answer, thought
    if finish_reason == "length":
        return "", content
    return content, ""


def _score(gold, reply: _Reply) -> dict:
    """Scoring fields of a request record.

    ``recall`` reads the final answer only, as RULER scores it. Reasoning restates
    the context, so counting it would credit a model that never answered.
    ``answer`` is kept so the scoring can be re-run.
    """
    return {
        "recall": recall(gold, reply.answer),
        "recall_with_reasoning": recall(gold, f"{reply.answer} {reply.reasoning}"),
        "finish_reason": reply.finish_reason,
        "reasoning_chars": len(reply.reasoning),
        "answer": reply.answer,
    }


def _get(url: str, timeout: float) -> int:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status


def _post_stream(url: str, body: dict, timeout: float):
    """POST a streamed chat completion and return its ``_Reply``."""
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
    finish_reason = None
    finished = False
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                finished = True
                break
            obj = json.loads(data)
            if obj.get("error"):
                raise StreamError(f"error in stream: {obj['error']}")
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices", []):
                finish_reason = ch.get("finish_reason") or finish_reason
                d = ch.get("delta") or {}
                piece = (d.get("content") or "") + (d.get("reasoning_content") or "")
                if piece:
                    now = time.time()
                    if ttft is None:
                        ttft = now - t0
                    t_last = now
                content += d.get("content") or ""
                reasoning += d.get("reasoning_content") or ""
    # A cut-off stream is not an answer; scoring it would record a false miss.
    if not finished or "completion_tokens" not in usage:
        raise StreamError("stream ended before the final usage chunk")
    decode_s = max(t_last - t0 - (ttft or 0), 0.0)
    answer, thought = _split_answer(content, reasoning, finish_reason)
    return _Reply(ttft or 0.0, decode_s, answer, thought, usage, finish_reason)


def _post_with_retry(url: str, body: dict, timeout: float, attempts: int):
    """``_post_stream`` with backoff; re-raises once ``attempts`` are used up."""
    for attempt in range(1, attempts + 1):
        try:
            return _post_stream(url, body, timeout)
        except _TRANSIENT:
            if attempt == attempts:
                raise
            time.sleep(2 * attempt)
    raise ValueError(f"attempts must be >= 1, got {attempts}")


def _serve_env(lost_dir: str) -> dict:
    """IVF arm with lost-attention measurement on. Indexer memory and time come
    from the A/B matrix: batching here would mix requests in both numbers."""
    env = dict(os.environ)
    env["VLLM_ENABLE_DSA_IVF_INDEXER"] = "1"
    env["VLLM_DSA_IVF_MEASURE_LOST_ATTENTION"] = "1"
    env["VLLM_DSA_IVF_LOST_ATTENTION_DIR"] = lost_dir
    # The recorder logs every forward; the job log can't take that for 25 cells.
    env["VLLM_LOGGING_CONFIG_PATH"] = LOGGING_CONFIG
    return env


def _hist_quantile(hist: dict[int, int], base: float, q: float) -> float:
    """``q`` quantile of a log-spaced histogram: the geometric middle of its bin."""
    target = q * sum(hist.values())
    seen = 0
    for b in sorted(hist):
        seen += hist[b]
        if seen >= target:
            return base ** (b + 0.5)
    return 0.0


def _summarize_timing(timing_dir: str) -> dict:
    """Indexer memory and per-(layer, phase) time over the per-rank dumps,
    including those in subdirectories (one per attempt).

    Within an attempt a mean is the slowest rank's and resident state is summed
    over layers and ranks. Across attempts means are weighted by calls and memory
    takes the maximum, since each serve allocates the same state again. Medians
    and p99 pool the calls of every rank and attempt.
    """
    peak = resident = 0.0
    totals: dict[str, float] = {}
    means: dict[str, dict[str, tuple[int, float]]] = {}
    hists: dict[str, dict[int, int]] = {}
    base = 0.0
    pattern = os.path.join(timing_dir, "**", "indexer_timing_pid*_r*.json")
    for fn in glob.glob(pattern, recursive=True):
        try:
            with open(fn) as f:
                rep = json.load(f)
        except Exception:
            continue
        attempt = os.path.dirname(fn)
        base = rep.get("hist_base", base)
        for layer, phases in rep.get("layers", {}).items():
            layer_resident = 0.0
            for phase, s in phases.items():
                peak = max(peak, s.get("peak_mib", 0.0))
                layer_resident = max(layer_resident, s.get("resident_mib", 0.0))
                key = f"{layer}|{phase}"
                seen = means.setdefault(attempt, {}).get(key, (0, 0.0))
                if s.get("mean_ms", 0.0) >= seen[1]:
                    means[attempt][key] = (s.get("calls", 0), s.get("mean_ms", 0.0))
                hist = hists.setdefault(key, {})
                for b, count in s.get("hist", {}).items():
                    hist[int(b)] = hist.get(int(b), 0) + count
            resident = max(resident, layer_resident)
            totals[attempt] = totals.get(attempt, 0.0) + layer_resident
    calls: dict[str, int] = {}
    ms_sum: dict[str, float] = {}
    for by_key in means.values():
        for key, (n, mean) in by_key.items():
            calls[key] = calls.get(key, 0) + n
            ms_sum[key] = ms_sum.get(key, 0.0) + n * mean
    return {
        "peak_indexer_mib": round(peak, 2),
        "resident_state_mib": round(resident, 2),
        "resident_total_mib": round(max(totals.values(), default=0.0), 2),
        "per_layer_ms": {
            k: round(ms_sum[k] / calls[k], 4) for k in sorted(calls) if calls[k]
        },
        "per_layer_calls": dict(sorted(calls.items())),
        "per_layer_p50_ms": {
            k: round(_hist_quantile(h, base, 0.5), 4)
            for k, h in sorted(hists.items()) if h
        },
        "per_layer_p99_ms": {
            k: round(_hist_quantile(h, base, 0.99), 4)
            for k, h in sorted(hists.items()) if h
        },
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
        # GLM-5.3 is multimodal: the indexer reads the nested text_config, so
        # the override must target text_config. A flat key lands on the top
        # glm5_next config and is silently ignored (clusters/probes -> 64/8).
        "--hf-overrides", json.dumps(
            {"text_config": {
                "index_num_clusters": clusters, "index_num_probes": probes,
            }}
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
    """Mean key_lost / mass_lost per (layer, phase) over the per-rank dumps,
    including those in subdirectories (one per attempt).

    ``mass_lost_total`` is 1 - summed kept mass / summed exact mass. Prefer it to
    ``mass_lost_mean``, the per-row mean, which scores a row with no attention
    mass to lose (exact mass 0) as 100% lost.
    """
    agg: dict[str, dict] = {}
    pattern = os.path.join(lost_dir, "**", "lost_attention_pid*.json")
    for fn in glob.glob(pattern, recursive=True):
        try:
            d = json.load(open(fn))
        except Exception:
            continue
        for key, s in d.items():
            a = agg.setdefault(key, {
                "rows": 0, "key_lost": 0.0, "mass_lost": 0.0, "real": 0.0, "exact": 0.0,
            })
            rows = s.get("rows", 0)
            a["rows"] += rows
            a["key_lost"] += s.get("key_lost_mean", 0.0) * rows
            a["mass_lost"] += s.get("mass_lost_mean", 0.0) * rows
            a["real"] += s.get("real_mass_mean", 0.0) * rows
            a["exact"] += s.get("exact_mass_mean", 0.0) * rows
    return {
        key: {
            "rows": a["rows"],
            "key_lost_mean": a["key_lost"] / a["rows"] if a["rows"] else 0.0,
            "mass_lost_mean": a["mass_lost"] / a["rows"] if a["rows"] else 0.0,
            "mass_lost_total": 1.0 - a["real"] / a["exact"] if a["exact"] else 0.0,
        }
        for key, a in agg.items()
    }


def _copy_dumps(dump_dir: str, kept_dir: str) -> None:
    """Copy the per-rank dumps off tmpfs; a storage blip costs only this copy."""
    for fn in glob.glob(os.path.join(dump_dir, "*.json")):
        dst = os.path.join(kept_dir, os.path.basename(fn))
        try:
            shutil.copyfile(fn, f"{dst}.tmp")
            os.replace(f"{dst}.tmp", dst)
        except OSError as e:
            _log(f"dump copy failed, will retry: {e!r}")


def _keep_lost(
    lost_dir: str, kept_dir: str, every_s: float, stop: threading.Event,
) -> None:
    while not stop.wait(every_s):
        _copy_dumps(lost_dir, kept_dir)


def _run_tasks(
    tag: str, port: int, length: int, todo: list[tuple[str, dict]],
    request_timeout: float, concurrency: int, max_tokens: int, seed: int | None,
    greedy: bool, request_attempts: int, ckpt: _Results,
) -> None:
    """Stream ``todo`` (task, sample) through the live serve, checkpointing each.

    All tasks share one pool, so samples that run to ``max_tokens`` overlap
    instead of each holding the GPUs for a whole task. Decode runs at about 6
    tok/s per request here, so one such sample takes about 45 minutes.

    ``max_tokens``, ``seed`` and ``greedy`` come from the ``fairness`` block of
    ``runs.yaml`` (the single source of truth both benchmark arms share), not
    from the per-task caps in ``tasks.py``, so the fidelity sweep uses the same
    generation budget and decoding settings as the A/B matrix.

    With ``concurrency`` > 1 the server's continuous batching runs the samples
    together. This is metric-neutral: recall is per-request and the per-(layer,
    phase) key/mass_lost stats are aggregated per row, so batch composition does
    not change them. Every sample here shares one length and clusters/probes. It
    does blur the per-request ``prefill_ms``/``decode_ms`` (requests share a
    step); the authoritative latency/timing come from the A/B matrix, not this
    sweep.

    The pool is not used as a context manager: on failure or SIGTERM that would
    wait for every queued request.
    """
    url = f"http://127.0.0.1:{port}/v1/chat/completions"

    def _one(task: str, ex: dict) -> dict:
        body = {
            "model": "glm",
            "messages": [{"role": "user", "content": ex["input"]}],
            "max_tokens": max_tokens,
            "temperature": 0 if greedy else 1.0,
        }
        if seed is not None:
            body["seed"] = seed
        reply = _post_with_retry(url, body, request_timeout, request_attempts)
        return {
            "task": task,
            "L": length,
            "idx": ex["index"],
            "gold": ex["outputs"],
            **_score(ex["outputs"], reply),
            "prefill_ms": round(reply.ttft_s * 1000, 1),
            "decode_ms": round(reply.decode_s * 1000, 1),
            "in_tok": reply.usage.get("prompt_tokens"),
            "out_tok": reply.usage.get("completion_tokens"),
        }

    pool = cf.ThreadPoolExecutor(max_workers=max(1, concurrency))
    futures = [pool.submit(_one, task, ex) for task, ex in todo]
    try:
        for done, fut in enumerate(cf.as_completed(futures), 1):
            ckpt.add(fut.result())
            if done % 20 == 0 or done == len(todo):
                _log(f"[{tag}] {done}/{len(todo)} requests "
                     f"({len(ckpt.done)} checkpointed)")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _serve_cell(
    tag: str, length: int, clusters: int, probes: int,
    todo: list[tuple[str, dict]], ckpt: _Results, kept_dir: str, dump_root: str,
    args,
) -> None:
    """One serve of a cell: send ``todo``, keeping its dumps in ``kept_dir``."""
    lost_dir = os.path.join(dump_root, "lost_attention")
    if os.path.isdir(lost_dir):
        shutil.rmtree(lost_dir)
    os.makedirs(lost_dir, exist_ok=True)

    max_len = length + args.max_tokens + args.length_margin
    cmd = _serve(
        args.model, args.tp, max_len, args.port, clusters, probes, args.gpu_mem_util
    )
    print(f"[{tag}] serve: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, env=_serve_env(lost_dir))
    stop = threading.Event()
    threading.Thread(
        target=_keep_lost, args=(lost_dir, kept_dir, args.lost_save_s, stop),
        daemon=True,
    ).start()
    try:
        t0 = time.time()
        _wait_ready(proc, args.port, args.startup_timeout)
        print(f"[{tag}] ready in {time.time() - t0:.0f}s", flush=True)
        _run_tasks(
            tag, args.port, length, todo, args.request_timeout, args.concurrency,
            args.max_tokens, args.seed, args.greedy, args.request_attempts, ckpt,
        )
    finally:
        stop.set()
        try:
            proc.terminate()
            try:
                proc.wait(timeout=90)
            except Exception:
                proc.kill()
            time.sleep(5)  # let the recorder's atexit dump settle
        finally:  # SIGTERM can land in the wait above
            _copy_dumps(lost_dir, kept_dir)


def _cell_config(
    length: int, clusters: int, probes: int, tasks: list[str], n, args,
) -> str:
    """The settings a cell's checkpoints are valid for."""
    return json.dumps({
        "length": length, "clusters": clusters, "probes": probes,
        "tasks": tasks, "n": n, "max_tokens": args.max_tokens,
        "seed": args.seed, "greedy": args.greedy, "scoring": "final_answer",
    }, sort_keys=True)


def _config_matches(cell_out: str, config: str) -> bool:
    try:
        with open(os.path.join(cell_out, "config.json")) as f:
            return f.read() == config
    except OSError:
        return False


def run_cell(
    length: int, clusters: int, probes: int, pct: float,
    tasks: list[str], n: int, args,
) -> dict:
    """Serve one (length, probe) cell once, run every task through it.

    Tasks are a free axis (same clusters/probes), so they share the serve; the
    cell has its own lost-attention dir, so its per-(layer, phase) stats are
    isolated from other cells. Lost attention is aggregated over all the cell's
    tasks; recall is reported per task. Resumes from the cell's checkpoints unless
    --force or runs.yaml changed for this cell.
    """
    tag = f"L{length} C{clusters} P{probes}({pct:g}%)"
    cell_out = os.path.join(args.out_dir, f"L{length}", f"C{clusters}_P{probes}")
    config = _cell_config(length, clusters, probes, tasks, n, args)
    if os.path.isdir(cell_out) and (
        args.force or not _config_matches(cell_out, config)
    ):
        shutil.rmtree(cell_out)
    kept_root = os.path.join(cell_out, "partial", "lost_attention")
    os.makedirs(kept_root, exist_ok=True)
    _write_atomic(os.path.join(cell_out, "config.json"), config)

    cap = None if n in ("all", None, 0) else int(n)
    rows_by_task: dict[str, list] = {}
    for task in tasks:
        jsonl = os.path.join(args.ruler_dir, task, f"validation_{length}.jsonl")
        rows_by_task[task] = [json.loads(line) for line in open(jsonl)][:cap]
        if not rows_by_task[task]:
            raise RuntimeError(f"no samples in {jsonl}")

    def _rec_key(task: str, ex: dict) -> str:
        return _key({"L": length, "task": task, "idx": ex["index"]})

    ckpt = _Results(os.path.join(cell_out, "partial", "results.jsonl"))
    todo = [(task, ex) for task, rows in rows_by_task.items() for ex in rows
            if _rec_key(task, ex) not in ckpt.done]
    if ckpt.done:
        _log(f"[{tag}] resume: {len(ckpt.done)} checkpointed, {len(todo)} to send")

    # The engine's per-rank instrumentation dumps go to node-local tmpfs, not
    # the shared Weka quota: a transient group-quota ENOSPC must never crash the
    # engine mid-cell (it did once, aborting a whole sweep). The runner copies
    # them to the cell's partial/ dir, where a failed write costs only a copy.
    local_base = os.environ.get("GLM5_LOCAL_DUMP_BASE", "/dev/shm")
    if not (os.path.isdir(local_base) and os.access(local_base, os.W_OK)):
        local_base = cell_out
    dump_root = os.path.join(
        local_base, f"glm5dump_{os.getpid()}_L{length}_P{probes}")
    if todo:
        # Numbered by directory: an attempt can end before finishing a request.
        attempt = 1 + len(os.listdir(kept_root))
        kept_dir = os.path.join(kept_root, f"attempt{attempt}")
        os.makedirs(kept_dir, exist_ok=True)
        _serve_cell(
            tag, length, clusters, probes, todo, ckpt, kept_dir, dump_root, args)
    ckpt.sync()
    shutil.rmtree(dump_root, ignore_errors=True)

    per_task: dict[str, dict] = {}
    for task, rows in rows_by_task.items():
        samples = [ckpt.done[_rec_key(task, ex)] for ex in rows]
        rmean = sum(s["recall"] for s in samples) / len(samples)
        per_task[task] = {
            "recall_mean": rmean,
            "capped": sum(s["finish_reason"] == "length" for s in samples),
            "samples": samples,
        }
        print(f"[{tag}] {task:16s} recall_mean={rmean:.3f} n={len(samples)}",
              flush=True)
    lost = _summarize_lost(kept_root)
    recalls = [t["recall_mean"] for t in per_task.values()]
    verdict = {
        "length": length,
        "clusters": clusters,
        "probes": probes,
        "probe_pct": pct,
        "n": n,
        "attempts": len(os.listdir(kept_root)),
        "recall_mean": sum(recalls) / len(recalls) if recalls else 0.0,
        "tasks": per_task,
        "lost_attention": lost,
    }
    _write_atomic(os.path.join(cell_out, "verdict.json"),
                  json.dumps(verdict, indent=2))
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
    nval = cells["n"]
    n = nval if nval in ("all", None) else int(nval)
    return {
        "n": n, "tasks": list(tasks), "grid": cells["grid"],
        "concurrency": int(cells["concurrency"]),
    }


def _load_shared(path: str) -> tuple[dict, dict]:
    """Read the ``fairness`` and ``resilience`` blocks from runs.yaml.

    Both benchmark arms and this sweep share them, so nothing here is hardcoded
    or taken from the per-task caps in ``tasks.py``.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg["fairness"], cfg["resilience"]


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
    p.add_argument("--concurrency", type=int,
                   help="override requests in flight per task (default: "
                        "sweep.cells.concurrency in runs.yaml). >1 leaves recall "
                        "and key/mass_lost unchanged but blurs per-request latency.")
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
    fairness, resilience = _load_shared(args.runs_yaml)
    if args.max_tokens is None:
        args.max_tokens = int(fairness["max_tokens"])
    args.length_margin = int(fairness["length_margin"])
    args.seed = int(fairness["seed"])
    args.greedy = bool(fairness["greedy"])
    args.concurrency = args.concurrency or sweep["concurrency"]
    args.request_attempts = int(resilience["request_attempts"])
    args.lost_save_s = float(resilience["lost_attention_save_s"])
    _exit_on_sigterm()
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

    pending = [
        (int(cell["length"]), int(cell["clusters"]), int(probes))
        for cell in grid for probes in cell["probes"]
    ]
    summary = []
    passes = int(resilience["run_passes"])
    for pass_no in range(1, passes + 1):
        failed = []
        for length, clusters, probes in pending:
            pct = round(100.0 * probes / clusters, 2)
            tag = f"L{length} C{clusters} P{probes}({pct:g}%)"
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
                    _log(f"[{tag}] resume: already complete, skipping")
                except Exception:
                    v = None  # truncated by a mid-write preemption -> re-run
            if v is None:
                try:
                    v = run_cell(length, clusters, probes, pct, tasks, n, args)
                except Exception as e:
                    # A dead serve or a storage failure must not abort the sweep.
                    # The cell wrote no verdict.json, so a later pass retries it.
                    _log(f"[{tag}] FAILED: {e!r} (pass {pass_no}/{passes})")
                    failed.append((length, clusters, probes))
                    continue
            summary.append(v)
            _log(f"[{tag}] recall_mean={v['recall_mean']:.3f} "
                 f"cells_done={len(summary)}")
            for key, s in sorted(v["lost_attention"].items()):
                _log(f"    {key:16s} key_lost={s['key_lost_mean']:.4f} "
                     f"mass_lost={s['mass_lost_total']:.4f} rows={s['rows']}")
        pending = failed
        if not pending:
            break
        if pass_no < passes:
            _log(f"{len(pending)} cell(s) failed; retrying after "
                 f"{resilience['pass_backoff_s']}s")
            time.sleep(float(resilience["pass_backoff_s"]))

    summary.sort(key=lambda v: (v["length"], v["probes"]))
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    _log(f"\nwrote {os.path.join(args.out_dir, 'summary.json')}")
    if pending:
        raise SystemExit(f"cells still failing after {passes} passes: {pending}")


if __name__ == "__main__":
    main()
