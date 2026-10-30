# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A/B benchmark matrix on Slurm (no Modal), driven by runs.yaml.

The Slurm counterpart of ``../matrix_run.py`` (which only runs on Modal). It
parses the run matrix with ``bench_config``, serves each run once, runs its
datasets through that serve, and stores results **per run**, keyed by
``run.id`` (unique by construction) so no two runs collide:

    <out>/<run.id>/
        config.json         the run's resolved settings, guards resume
        partial/<ds>/results.jsonl   one line per finished request
        accuracy.json       per-task recall + mean (ruler)
        latency.json        single-stream prefill/decode (concurrency == 1)
        throughput.json     aggregate tok/s        (concurrency > 1)
        verdict.json        per-run roll-up
    <out>/summary.json      roll-up over every run

A preempted job is requeued and resumes from results.jsonl, so only unfinished
requests are re-sent. A run that fails (dead serve, storage error) is skipped and
retried on a later pass, per ``resilience`` in runs.yaml; the job exits non-zero
if any run is still failing after the last pass. Instrument and lost-attention
stats cover only the requests served by the final attempt; ``verdict.json``
records how many as ``instrumented_requests``.

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
import subprocess
import sys
import time

BENCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLURM_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BENCH_DIR)
sys.path.insert(0, SLURM_DIR)
from bench_config import load_matrix  # noqa: E402
from tasks import recall  # noqa: E402

# Reuse the proven serve/HTTP/summary helpers from the fidelity runner.
from lost_attention_run import (  # noqa: E402
    _exit_on_sigterm,
    _get,
    _log,
    _post_with_retry,
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
    if run.eager:
        env["GLM5_INSTRUMENT"] = "1"
        env["GLM5_TIMING_DIR"] = timing_dir
        env["GLM5_MEM_PROBE"] = "1"
        env["GLM5_RESIDENT_PROBE"] = "1"
    if run.measure_lost_attention:
        env["VLLM_DSA_IVF_LOST_ATTENTION_DIR"] = lost_dir
    return env


def _serve_cmd(run, model: str, tp: int, port: int, gpu_mem_util: float) -> list:
    """`vllm serve` for a run: fixed flags plus the run's own serve_args
    (max-model-len, seed, eager, prefix-caching, hf-overrides)."""
    return [
        shutil.which("vllm") or "vllm", "serve", model,
        "--served-model-name", "glm",
        "--tensor-parallel-size", str(tp),
        "--gpu-memory-utilization", str(gpu_mem_util),
        "--port", str(port),
    ] + run.serve_args()


def _one_request(port: int, meta: dict, run, args) -> dict:
    body = {
        "model": "glm",
        "messages": [{"role": "user", "content": meta["input"]}],
        "max_tokens": run.max_tokens,
        "temperature": 0 if run.greedy else 1.0,
        "seed": run.seed,
    }
    ttft, dec_s, text, usage = _post_with_retry(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        body, args.request_timeout, args.request_attempts,
    )
    otk = usage.get("completion_tokens")
    return {
        "task": meta["task"], "L": meta["L"], "idx": meta["idx"],
        "recall": recall(meta["gold"], text),
        "prefill_ms": round(ttft * 1000, 1),
        "decode_ms": round(dec_s * 1000, 1),
        "decode_s": dec_s, "out_tok": otk,
    }


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


def _run_requests(port: int, metas: list, run, args, ckpt: _Results) -> None:
    """Send ``metas`` at the run's concurrency, checkpointing each as it finishes.

    The pool is not used as a context manager: on failure or SIGTERM that would
    wait for every queued request.
    """
    t0 = time.time()
    pool = cf.ThreadPoolExecutor(max_workers=run.concurrency)
    futures = [pool.submit(_one_request, port, m, run, args) for m in metas]
    try:
        for i, fut in enumerate(cf.as_completed(futures), 1):
            rec = fut.result()
            rec["t_done"] = round(time.time() - t0, 3)
            ckpt.add(rec)
            if i % 20 == 0 or i == len(metas):
                _log(f"[{run.id}] attempt {ckpt.attempt}: {i}/{len(metas)} "
                     f"requests ({len(ckpt.done)} checkpointed)")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _throughput(results: list, concurrency: int) -> dict:
    """Aggregate tok/s, with wall time summed over the attempts that served them.

    An attempt's wall time ends at its last finished request, so time lost to a
    preemption or to the queue is not counted, and neither are its in-flight
    tokens.
    """
    walls: dict[int, float] = {}
    for r in results:
        walls[r["attempt"]] = max(walls.get(r["attempt"], 0.0), r["t_done"])
    wall = sum(walls.values())
    out_tok = sum(r["out_tok"] or 0 for r in results)
    return {
        "concurrency": concurrency, "requests": len(results), "out_tok": out_tok,
        "wall_s": round(wall, 2), "attempts": len(walls),
        "aggregate_tok_s": round(out_tok / wall, 2) if wall > 0 else None,
    }


def _config_text(run) -> str:
    """The settings a checkpoint is valid for."""
    return json.dumps({
        "serve": run.serve_key(), "max_tokens": run.max_tokens,
        "seed": run.seed, "greedy": run.greedy,
        "datasets": [[d.name, d.n, d.lengths, d.tasks] for d in run.datasets],
    }, sort_keys=True)


def _config_matches(run, base_out: str) -> bool:
    try:
        with open(os.path.join(base_out, "config.json")) as f:
            return f.read() == _config_text(run)
    except OSError:
        return False


def run_one(run, args) -> dict:
    """Serve one run, run its datasets, store results under its own id."""
    base_out = os.path.join(args.out_dir, run.id)
    partial_dir = os.path.join(base_out, "partial")
    # Resume from the checkpoints unless --force or runs.yaml changed for this
    # run. The per-rank instrument dumps always start empty and cover only this
    # attempt. They go to node-local tmpfs so a shared-quota ENOSPC can't crash
    # the engine, and are discarded once summarized into verdict.json.
    resuming = not args.force and _config_matches(run, base_out)
    if os.path.isdir(base_out) and not resuming:
        _log(f"[{run.id}] no valid checkpoint for this config, starting clean")
        shutil.rmtree(base_out)
    os.makedirs(partial_dir, exist_ok=True)
    _write_atomic(os.path.join(base_out, "config.json"), _config_text(run))
    local_base = os.environ.get("GLM5_LOCAL_DUMP_BASE", "/dev/shm")
    if not (os.path.isdir(local_base) and os.access(local_base, os.W_OK)):
        local_base = base_out
    dump_root = os.path.join(local_base, f"glm5dump_{os.getpid()}_{run.id}")
    timing_dir = os.path.join(dump_root, "timing")
    lost_dir = os.path.join(dump_root, "lost_attention")
    for d in (timing_dir, lost_dir):
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    cmd = _serve_cmd(run, args.model, args.tp, args.port, args.gpu_mem_util)
    print(f"[{run.id}] serve: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, env=_serve_env(run, timing_dir, lost_dir))

    accuracy: dict = {}
    latency = None
    throughput = None
    skipped: dict = {}
    instrumented = 0
    try:
        t0 = time.time()
        _wait_ready(proc, args.port, args.startup_timeout)
        print(f"[{run.id}] ready in {time.time() - t0:.0f}s", flush=True)

        for ds in run.datasets:
            if ds.name != "ruler":
                skipped[ds.name] = "scorer_not_implemented"
                continue

            cap = ds.n or 1

            def _metas(length, task):
                path = os.path.join(
                    args.ruler_dir, task, f"validation_{length}.jsonl")
                if not os.path.exists(path):
                    raise SystemExit(f"missing ruler data: {path}")
                rows = [json.loads(line) for line in open(path)][:cap]
                return [{"task": task, "L": length, "idx": ex["index"],
                         "gold": ex["outputs"], "input": ex["input"]}
                        for ex in rows]

            metas = [m for length in ds.lengths for task in ds.tasks
                     for m in _metas(length, task)]
            ds_partial = os.path.join(partial_dir, ds.name)
            os.makedirs(ds_partial, exist_ok=True)
            ckpt = _Results(os.path.join(ds_partial, "results.jsonl"))
            todo = [m for m in metas if _key(m) not in ckpt.done]
            if run.eager and not todo:
                # Everything was checkpointed but the verdict was not written.
                # Repeat the last task so this attempt's dumps aren't empty.
                todo = metas[-cap:]
            if ckpt.done:
                _log(f"[{run.id}] resume: {len(ckpt.done)}/{len(metas)} "
                     f"checkpointed, attempt {ckpt.attempt}")
            _run_requests(args.port, todo, run, args, ckpt)
            instrumented += len(todo)
            ckpt.sync()
            results = [ckpt.done[_key(m)] for m in metas]

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
                throughput = _throughput(results, run.concurrency)
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
    shutil.rmtree(dump_root, ignore_errors=True)
    if "lost_attention" in verdict:
        verdict["lost_attention"] = "PASS" if lost else "MISSING"

    _write_atomic(os.path.join(base_out, "accuracy.json"),
                  json.dumps(accuracy, indent=2))
    if latency is not None:
        _write_atomic(os.path.join(base_out, "latency.json"),
                      json.dumps(latency, indent=2))
    if throughput is not None:
        _write_atomic(os.path.join(base_out, "throughput.json"),
                      json.dumps(throughput, indent=2))
    summary = {
        "id": run.id, "indexer": run.indexer, "exec": run.exec_mode,
        "concurrency": run.concurrency, "length": run.length,
        "max_model_len": run.max_model_len, "hf_overrides": run.hf_overrides,
        "accuracy": accuracy, "latency": latency, "throughput": throughput,
        "memory_timing": timing, "lost_attention": lost,
        "instrumented_requests": instrumented if run.eager else None,
        "skipped": skipped, "verdict": verdict,
    }
    _write_atomic(os.path.join(base_out, "verdict.json"),
                  json.dumps(summary, indent=2))
    _log(f"[{run.id}] done -> {base_out}  verdict={verdict}")
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

    resilience = matrix.resilience
    args.request_attempts = int(resilience["request_attempts"])
    passes = int(resilience["run_passes"])
    _exit_on_sigterm()

    os.makedirs(args.out_dir, exist_ok=True)
    _log(f"==== A/B matrix: {len(runs)} run(s) tp={args.tp} "
         f"(from {args.runs_yaml}) ====")
    summary = {}
    pending = runs
    failed: dict[str, str] = {}
    for pass_no in range(1, passes + 1):
        failed = {}
        for run in pending:
            # Resume: a run is finished iff it wrote a readable verdict.json (the
            # last write in run_one) for the current config, so a requeue re-runs
            # only the runs that did not complete. --force overrides.
            base_out = os.path.join(args.out_dir, run.id)
            done = os.path.join(base_out, "verdict.json")
            prior = None
            if not args.force and _config_matches(run, base_out):
                try:
                    with open(done) as f:
                        prior = json.load(f)["verdict"]
                except (OSError, ValueError, KeyError):
                    prior = None  # missing or truncated -> run it
            if prior is not None:
                summary[run.id] = prior
                _log(f"[{run.id}] resume: already complete, skipping")
            else:
                try:
                    summary[run.id] = run_one(run, args)["verdict"]
                except Exception as e:
                    # A dead serve or a storage failure must not abort the matrix.
                    # Its requests stay checkpointed for the next pass.
                    _log(f"[{run.id}] FAILED: {e!r} (pass {pass_no}/{passes})")
                    failed[run.id] = repr(e)
                    summary[run.id] = {"failed": repr(e)}
            try:
                _write_atomic(os.path.join(args.out_dir, "summary.json"),
                              json.dumps(summary, indent=2))
            except OSError as e:
                _log(f"summary write failed: {e!r}")
        pending = [r for r in pending if r.id in failed]
        if not pending:
            break
        if pass_no < passes:
            _log(f"{len(pending)} run(s) failed; retrying after "
                 f"{resilience['pass_backoff_s']}s")
            time.sleep(float(resilience["pass_backoff_s"]))

    _log(f"\nwrote {os.path.join(args.out_dir, 'summary.json')}")
    if failed:
        raise SystemExit(f"runs still failing after {passes} passes: {failed}")


if __name__ == "__main__":
    main()
