# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-sample A/B smoke test on Modal (GLM-5.3-Flash, 4x H200 TP4).

Serves one arm at a time from an image built off THIS fork, runs one RULER
sample through it, and verifies every metric family was actually recorded:

    accuracy            client recall of the gold needle
    latency             streamed prefill (TTFT) + decode ms
    per_layer_indexer_ms   instrument timing file, >0 on the DSA layers
    peak_indexer_mib    instrument mem probe
    resident_state_mib  instrument resident probe (k/cluster/centroid caches)
    lost_attention      ivf_fidelity dump (ivf arm only)

It is deliberately tiny (one task, one length, n=1) so it is cheap and its only
job is to confirm the full pipeline wires up and each metric lands, before the
real matrix is run.

    modal run smoke_single.py                 # both arms
    modal run smoke_single.py --arm ivf       # one arm
"""

import os

import modal

app = modal.App("glm5-ivf-smoke")

# Build the image from this fork so the IVF + lost-attention code is present.
# VLLM_USE_PRECOMPILED pulls the matching prebuilt .so (IVF itself is pure
# Python/Triton, no CUDA rebuild); the fork source provides the Python.
FORK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
BENCH_DIR = "/fork/vllm/models/glm5next/benchmark"

image = (
    modal.Image.from_registry("vllm/vllm-openai:nightly", add_python="3.12")
    .entrypoint([])
    .pip_install("uv")
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
        "cd /fork && VLLM_USE_PRECOMPILED=1 uv pip install --system -e . "
        "--torch-backend=auto"
    )
)

model_vol = modal.Volume.from_name("glm5-model")
ruler_vol = modal.Volume.from_name("glm5-ruler")
out_vol = modal.Volume.from_name("glm5-smoke-results", create_if_missing=True)

MODEL = "/model/GLM-5.3-Flash"
PORT = 8000
HOUR = 3600

# The single sample.
TASK = "niah_single_2"
LENGTH = 8192
N = 1
MAX_TOKENS = 1024

# IVF knobs (match the runs.yaml base operating point).
IVF_CLUSTERS = 256
IVF_PROBES = 32


def _arm_env(arm: str, timing_dir: str, lost_dir: str) -> dict:
    """Serve-process env for one arm: instrument always on, IVF on for ivf."""
    env = dict(os.environ)
    env["PYTHONPATH"] = BENCH_DIR + (
        ":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["GLM5_INSTRUMENT"] = "1"
    env["GLM5_TIMING_DIR"] = timing_dir
    env["GLM5_MEM_PROBE"] = "1"
    env["GLM5_RESIDENT_PROBE"] = "1"
    if arm == "ivf":
        env["VLLM_ENABLE_DSA_IVF_INDEXER"] = "1"
        env["VLLM_DSA_IVF_MEASURE_LOST_ATTENTION"] = "1"
        env["VLLM_DSA_IVF_LOST_ATTENTION_DIR"] = lost_dir
    else:
        env["VLLM_ENABLE_DSA_IVF_INDEXER"] = "0"
    return env


@app.function(
    image=image,
    gpu="H200:4",
    volumes={"/model": model_vol, "/ruler": ruler_vol, "/outputs": out_vol},
    timeout=3 * HOUR,
    memory=220 * 1024,
)
def run_arm(arm: str = "kpool"):
    """Serve one arm, run the single sample, verify every metric recorded."""
    import json
    import shutil
    import subprocess
    import sys
    import time
    import urllib.request

    assert arm in ("kpool", "ivf")
    sys.path.insert(0, BENCH_DIR)
    from tasks import recall  # noqa: E402

    base_out = f"/outputs/smoke/{arm}"
    timing_dir = f"{base_out}/timing"
    lost_dir = f"{base_out}/lost_attention"
    os.makedirs(timing_dir, exist_ok=True)
    os.makedirs(lost_dir, exist_ok=True)

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

    # ---- serve (eager so the instrument can time per layer) ----
    vllm_bin = shutil.which("vllm") or "/usr/local/bin/vllm"
    cmd = [
        vllm_bin, "serve", MODEL, "--served-model-name", "glm",
        "--tensor-parallel-size", "4", "--trust-remote-code",
        "--max-model-len", "34000", "--gpu-memory-utilization", "0.9",
        "--enforce-eager", "--no-enable-prefix-caching", "--port", str(PORT),
    ]
    if arm == "ivf":
        cmd += ["--hf-overrides", json.dumps(
            {"index_num_clusters": IVF_CLUSTERS, "index_num_probes": IVF_PROBES})]
    env = _arm_env(arm, timing_dir, lost_dir)
    print(f"[{arm}] serve: {' '.join(cmd)}", flush=True)
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
    print(f"[{arm}] ready in {time.time() - t0:.0f}s", flush=True)

    # ---- the single sample ----
    rows = [json.loads(line) for line in open(f"/ruler/{TASK}/validation_{LENGTH}.jsonl")]
    ex = rows[0]
    body = {
        "model": "glm",
        "messages": [{"role": "user", "content": ex["input"]}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
    }
    ttft, dec_s, text, usage = _post_stream(
        f"http://127.0.0.1:{PORT}/v1/chat/completions", body, timeout=2400)
    otk = usage.get("completion_tokens")
    rec = {
        "task": TASK, "L": LENGTH, "idx": ex["index"], "gold": ex["outputs"],
        "recall": recall(ex["outputs"], text),
        "prefill_ms": round(ttft * 1000, 1),
        "decode_ms": round(dec_s * 1000, 1),
        "decode_tok_s": round((otk - 1) / dec_s, 2) if otk and dec_s > 0 else None,
        "in_tok": usage.get("prompt_tokens"), "out_tok": otk,
    }
    print(f"[{arm}] recall={rec['recall']} prefill_ms={rec['prefill_ms']} "
          f"decode_ms={rec['decode_ms']} out_tok={otk}", flush=True)

    # stop the serve so the instrument + ivf_fidelity flush their atexit dumps
    proc.terminate()
    try:
        proc.wait(timeout=60)
    except Exception:
        proc.kill()
    time.sleep(5)
    out_vol.commit()

    verdict = _verify(arm, rec, timing_dir, lost_dir)
    with open(f"{base_out}/verdict.json", "w") as f:
        json.dump({"rec": rec, "verdict": verdict}, f, indent=2)
    out_vol.commit()
    return {"arm": arm, "rec": rec, "verdict": verdict}


def _verify(arm: str, rec: dict, timing_dir: str, lost_dir: str) -> dict:
    """Check each metric family actually landed; return name -> status string."""
    import glob
    import json

    v = {}
    v["accuracy"] = "PASS" if rec.get("recall") is not None else "MISSING"
    v["latency"] = (
        "PASS" if rec.get("prefill_ms") and rec.get("decode_ms") is not None
        else "MISSING"
    )

    tfiles = glob.glob(f"{timing_dir}/indexer_timing_pid*_r*.json")
    per_layer = peak = resident = False
    for fn in tfiles:
        try:
            rep = json.load(open(fn))
        except Exception:
            continue
        for _layer, phases in rep.get("layers", {}).items():
            for _phase, s in phases.items():
                if s.get("mean_ms", 0) > 0:
                    per_layer = True
                if s.get("peak_mib", 0) > 0:
                    peak = True
                if s.get("resident_mib", 0) > 0:
                    resident = True
    v["per_layer_indexer_ms"] = (
        "PASS" if per_layer else f"MISSING (no timing in {len(tfiles)} files)"
    )
    v["peak_indexer_mib"] = "PASS" if peak else "MISSING"
    v["resident_state_mib"] = "PASS" if resident else "MISSING"

    if arm == "ivf":
        lf = glob.glob(f"{lost_dir}/lost_attention_pid*.json")
        ok = False
        for fn in lf:
            try:
                d = json.load(open(fn))
            except Exception:
                continue
            for _k, s in d.items():
                if "key_lost_mean" in s and "mass_lost_mean" in s:
                    ok = True
        v["lost_attention"] = "PASS" if ok else f"MISSING ({len(lf)} files)"
    else:
        v["lost_attention"] = "n/a"
    return v


@app.local_entrypoint()
def main(arm: str = "both"):
    arms = ["kpool", "ivf"] if arm == "both" else [arm]
    print(f"\n==== single-sample smoke: {TASK}@{LENGTH} n={N}, arms={arms} ====")
    for a in arms:
        r = run_arm.remote(a)
        print(f"\n----- {a} -----")
        print(f"  recall={r['rec']['recall']}  prefill_ms={r['rec']['prefill_ms']}"
              f"  decode_ms={r['rec']['decode_ms']}")
        for metric, status in r["verdict"].items():
            print(f"  {metric:22s} {status}")
