# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A/B benchmark matrix driver on Modal (GLM-5.3-Flash, 4x H200 TP4).

Drives the run matrix in ``runs.yaml`` (parsed by ``bench_config``): one Modal
invocation per run, each serving its arm once and running its datasets through
that serve. Results are stored **per run**, keyed by ``run.id`` (unique by
construction), so no two runs can collide:

    /outputs/matrix/<run.id>/
        accuracy.json       per-task recall + mean (ruler)
        latency.json        single-stream prefill/decode (concurrency == 1)
        throughput.json     aggregate tok/s        (concurrency > 1)
        timing/             indexer_timing_pid*_r*.json (eager runs only)
        lost_attention/     lost_attention_pid*.json    (base ivf only)
        verdict.json        per-run roll-up (accuracy + latency/throughput)

The local entrypoint prints a matrix-wide roll-up of every run's verdict after
the runs finish. ``smoke_single.py`` remains the one-sample pipeline check; this
is the full
matrix. regression/swebench have no in-repo scorer yet, so those datasets are
recorded as ``scorer_not_implemented`` rather than silently skipped.

    modal run matrix_run.py                 # every run in runs.yaml
    modal run matrix_run.py --only ivf,kpool
    modal run matrix_run.py --only ivf_L65536_c16
"""

import os

import modal

app = modal.App("glm5-ivf-matrix")

FORK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
BENCH_DIR = "/fork/vllm/models/glm5next/benchmark"

# Same overlay-on-nightly image as the smoke test (see smoke_single.py).
image = (
    modal.Image.from_registry("vllm/vllm-openai:nightly", add_python="3.12")
    .entrypoint([])
    .add_local_dir(
        FORK,
        "/fork",
        ignore=[
            ".git",
            "**/__pycache__",
            "**/*.pyc",
            ".venv",
            "build",
            "dist",
            "*.egg-info",
        ],
        copy=True,
    )
    .run_commands(
        'VLLM_PY=$(head -1 "$(which vllm)" | sed "s/^#!//") && '
        'VLLM_DIR=$("$VLLM_PY" -c '
        '"import vllm, os; print(os.path.dirname(vllm.__file__))") && '
        'echo "overlay -> $VLLM_DIR" && cp -rf /fork/vllm/* "$VLLM_DIR/"'
    )
)

model_vol = modal.Volume.from_name("glm5-model")
ruler_vol = modal.Volume.from_name("glm5-ruler")
out_vol = modal.Volume.from_name("glm5-matrix-results", create_if_missing=True)

MODEL = "/model/GLM-5.3-Flash"
PORT = 8000
HOUR = 3600
GPU = "H200:4"
TP = 4


def _serve_env(run, timing_dir: str, lost_dir: str) -> dict:
    """Serve-process env for a run: arm env from bench_config, instrument on
    only for eager runs, lost-attention dir only when the run measures it."""
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


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/model": model_vol, "/ruler": ruler_vol, "/outputs": out_vol},
    timeout=6 * HOUR,
    memory=320 * 1024,
)
def run_one(run_id: str):
    """Serve one run, run its datasets, store results under its own id."""
    import concurrent.futures as cf
    import glob
    import json
    import shutil
    import subprocess
    import sys
    import time
    import urllib.request

    sys.path.insert(0, BENCH_DIR)
    from bench_config import load_matrix  # noqa: E402
    from tasks import TASKS, recall  # noqa: E402

    matrix = load_matrix(os.path.join(BENCH_DIR, "runs.yaml"))
    run = next((r for r in matrix.runs if r.id == run_id), None)
    if run is None:
        raise SystemExit(f"run id {run_id!r} not in runs.yaml")

    base_out = f"/outputs/matrix/{run.id}"
    timing_dir = f"{base_out}/timing"
    lost_dir = f"{base_out}/lost_attention"
    for d in (base_out, timing_dir, lost_dir):
        os.makedirs(d, exist_ok=True)

    def _get(url, timeout):
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status

    def _post_stream(url, body, timeout):
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
        usage = {}
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
                    piece = (d.get("content") or "") + (
                        d.get("reasoning_content") or "")
                    if piece:
                        now = time.time()
                        if ttft is None:
                            ttft = now - t0
                        t_last = now
                    content += d.get("content") or ""
                    reasoning += d.get("reasoning_content") or ""
        decode_s = max(t_last - t0 - (ttft or 0), 0.0)
        return ttft or 0.0, decode_s, content + " " + reasoning, usage

    # ---- serve ----
    vllm_bin = shutil.which("vllm") or "/usr/local/bin/vllm"
    cmd = [
        vllm_bin, "serve", MODEL, "--served-model-name", "glm",
        "--tensor-parallel-size", str(TP), "--gpu-memory-utilization", "0.9",
        "--port", str(PORT),
    ] + run.serve_args()
    env = _serve_env(run, timing_dir, lost_dir)
    print(f"[{run.id}] serve: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, env=env)
    t0 = time.time()
    while time.time() - t0 < 45 * 60:
        if proc.poll() is not None:
            raise RuntimeError(f"vllm exited during startup (code {proc.returncode})")
        try:
            if _get(f"http://127.0.0.1:{PORT}/health", timeout=5) == 200:
                break
        except Exception:
            pass
        time.sleep(10)
    print(f"[{run.id}] ready in {time.time() - t0:.0f}s", flush=True)

    url = f"http://127.0.0.1:{PORT}/v1/chat/completions"

    def _one_request(meta):
        body = {
            "model": "glm",
            "messages": [{"role": "user", "content": meta["input"]}],
            "max_tokens": TASKS[meta["task"]]["max_tokens"],
            "temperature": 0,
        }
        ttft, dec_s, text, usage = _post_stream(url, body, timeout=2 * HOUR)
        otk = usage.get("completion_tokens")
        return {
            "task": meta["task"], "L": meta["L"], "idx": meta["idx"],
            "recall": recall(meta["gold"], text),
            "prefill_ms": round(ttft * 1000, 1),
            "decode_ms": round(dec_s * 1000, 1),
            "decode_s": dec_s, "out_tok": otk,
        }

    # ---- datasets ----
    accuracy = {}        # dataset -> {per_task, mean}
    latency = None
    throughput = None
    skipped = {}
    for ds in run.datasets:
        if ds.name != "ruler":
            skipped[ds.name] = "scorer_not_implemented"
            continue

        # Eager runs are the instrumented accuracy/fidelity points: use every
        # sample. Graph runs are throughput/latency: a per-task cap is enough.
        cap = None if run.eager else (ds.n or 1)
        metas = []
        for length in ds.lengths:
            for task in ds.tasks:
                path = f"/ruler/{task}/validation_{length}.jsonl"
                if not os.path.exists(path):
                    raise SystemExit(f"missing ruler data: {path}")
                rows = [json.loads(line) for line in open(path)]
                for ex in rows[:cap]:
                    metas.append({
                        "task": task, "L": length, "idx": ex["index"],
                        "gold": ex["outputs"], "input": ex["input"],
                    })

        wall0 = time.time()
        if run.concurrency <= 1:
            results = [_one_request(m) for m in metas]
        else:
            with cf.ThreadPoolExecutor(max_workers=run.concurrency) as pool:
                results = list(pool.map(_one_request, metas))
        wall = time.time() - wall0

        per_task = {}
        for t in ds.tasks:
            rs = [r["recall"] for r in results if r["task"] == t]
            if rs:
                per_task[t] = round(sum(rs) / len(rs), 4)
        mean = round(sum(per_task.values()) / len(per_task), 4) if per_task else None
        accuracy[ds.name] = {"per_task": per_task, "mean": mean, "n_req": len(results)}

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
                "concurrency": run.concurrency,
                "requests": len(results),
                "out_tok": out_tok,
                "wall_s": round(wall, 2),
                "aggregate_tok_s": round(out_tok / wall, 2) if wall > 0 else None,
            }

    # ---- stop serve so instrument + ivf_fidelity flush their atexit dumps ----
    proc.terminate()
    try:
        proc.wait(timeout=90)
    except Exception:
        proc.kill()
    time.sleep(5)
    out_vol.commit()

    # ---- serve-level metrics from the dumps ----
    def _timing_present():
        per_layer = peak = resident = split = False
        for fn in glob.glob(f"{timing_dir}/indexer_timing_pid*_r*.json"):
            try:
                rep = json.load(open(fn))
            except Exception:
                continue
            for _l, phases in rep.get("layers", {}).items():
                for _p, s in phases.items():
                    per_layer |= s.get("mean_ms", 0) > 0
                    peak |= s.get("peak_mib", 0) > 0
                    resident |= s.get("resident_mib", 0) > 0
                    if "cluster" in _p or "assign" in _p:
                        split |= s.get("mean_ms", 0) > 0
        return per_layer, peak, resident, split

    def _lost_present():
        for fn in glob.glob(f"{lost_dir}/lost_attention_pid*.json"):
            try:
                d = json.load(open(fn))
            except Exception:
                continue
            for _k, s in d.items():
                if "key_lost_mean" in s and "mass_lost_mean" in s:
                    return True
        return False

    verdict = {m: "expected_but_missing" for m in run.metrics()}
    verdict["accuracy"] = "PASS" if accuracy else "MISSING"
    if "latency_ttft" in verdict:
        verdict["latency_ttft"] = "PASS" if latency else "MISSING"
    if "throughput" in verdict:
        verdict["throughput"] = "PASS" if throughput else "MISSING"
    if run.eager:
        per_layer, peak, resident, split = _timing_present()
        verdict["per_layer_indexer_ms"] = "PASS" if per_layer else "MISSING"
        verdict["peak_indexer_mib"] = "PASS" if peak else "MISSING"
        verdict["resident_state_mib"] = "PASS" if resident else "MISSING"
        if "prefill_cluster_assign_split" in verdict:
            verdict["prefill_cluster_assign_split"] = "PASS" if split else "MISSING"
    if "lost_attention" in verdict:
        verdict["lost_attention"] = "PASS" if _lost_present() else "MISSING"

    with open(f"{base_out}/accuracy.json", "w") as f:
        json.dump(accuracy, f, indent=2)
    if latency is not None:
        with open(f"{base_out}/latency.json", "w") as f:
            json.dump(latency, f, indent=2)
    if throughput is not None:
        with open(f"{base_out}/throughput.json", "w") as f:
            json.dump(throughput, f, indent=2)
    summary = {
        "id": run.id, "indexer": run.indexer, "exec": run.exec_mode,
        "concurrency": run.concurrency, "length": run.length,
        "max_model_len": run.max_model_len, "hf_overrides": run.hf_overrides,
        "accuracy": accuracy, "latency": latency, "throughput": throughput,
        "skipped": skipped, "verdict": verdict,
    }
    with open(f"{base_out}/verdict.json", "w") as f:
        json.dump(summary, f, indent=2)
    out_vol.commit()
    print(f"[{run.id}] done -> {base_out}", flush=True)
    return summary


@app.local_entrypoint()
def main(only: str = ""):
    import json
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from bench_config import load_matrix

    matrix = load_matrix(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "runs.yaml"))
    ids = [r.id for r in matrix.runs]
    if only:
        want = {s.strip() for s in only.split(",") if s.strip()}
        bad = want - set(ids)
        if bad:
            raise SystemExit(f"unknown run id(s): {sorted(bad)}")
        ids = [i for i in ids if i in want]

    print(f"\n==== A/B matrix: {len(ids)} run(s) ====")
    summaries = []
    for s in run_one.map(ids, order_outputs=True):
        summaries.append(s)
        acc = s["accuracy"].get("ruler", {}).get("mean")
        extra = (f"tok/s={s['throughput']['aggregate_tok_s']}"
                 if s["throughput"] else
                 (f"ttft={s['latency']['prefill_ttft_ms']}ms"
                  if s["latency"] else "-"))
        fails = [m for m, st in s["verdict"].items() if st not in ("PASS", "n/a")]
        print(f"  {s['id']:22s} ruler_recall={acc} {extra}"
              + (f"  FAILS={fails}" if fails else ""))

    # a matrix-wide roll-up lives beside the per-run dirs.
    print(json.dumps({s["id"]: s["verdict"] for s in summaries}, indent=2))
