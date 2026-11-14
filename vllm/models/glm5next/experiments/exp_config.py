# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Config, serve command and RULER samples shared by the experiment runners."""

from __future__ import annotations

import json
import os
import shutil

import yaml

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.join(os.path.dirname(EXP_DIR), "benchmark")


def load(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["ivf_grid"] = {int(k): v for k, v in cfg["ivf_grid"].items()}
    for length in cfg["capacity"]["lengths"] + cfg["reorder"]["capture"]["lengths"]:
        if length not in cfg["ivf_grid"]:
            raise ValueError(f"length {length} has no ivf_grid entry")
    return cfg


def max_model_len(cfg: dict, length: int) -> int:
    return length + int(cfg["fairness"]["length_margin"])


def max_tokens(cfg: dict, task: str) -> int:
    return int(cfg["fairness"]["max_tokens_by_task"][task])


def serve_env(*, indexer: str, clusters: int | None = None,
              probes: int | None = None) -> dict:
    """The serve's env: this dir's sitecustomize first, then ../benchmark for
    expect_ivf; the arm flag; the clusters/probes guard; stack dumps."""
    env = dict(os.environ)
    env["PYTHONPATH"] = ":".join(
        [EXP_DIR, BENCH_DIR] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env["VLLM_ENABLE_DSA_IVF_INDEXER"] = "1" if indexer == "ivf" else "0"
    env["GLM5_STACK_DUMP"] = "1"
    if indexer == "ivf":
        env["GLM5_EXPECT_IVF"] = f"{clusters},{probes}"
    for k in ("GLM5_KEY_CAPTURE_DIR", "GLM5_KEY_CAPTURE_LAYERS"):
        env.pop(k, None)
    return env


def serve_cmd(cfg: dict, *, length: int, tp: int, port: int, util: float,
              indexer: str, clusters: int | None = None, probes: int | None = None,
              eager: bool = False, chat_template: str | None = None) -> list[str]:
    """`vllm serve` with the matrix's flags (see benchmark/bench_config.py)."""
    cmd = [
        shutil.which("vllm") or "vllm", "serve", cfg["model_dir"],
        "--served-model-name", "glm",
        "--tensor-parallel-size", str(tp),
        "--gpu-memory-utilization", f"{util:.2f}",
        "--port", str(port),
        "--max-model-len", str(max_model_len(cfg, length)),
        "--seed", str(cfg["fairness"]["seed"]),
        "--trust-remote-code",
        "--no-enable-prefix-caching",
    ]
    if eager:
        cmd.append("--enforce-eager")
    if indexer == "ivf":
        # The indexer knobs live on the nested text_config (GLM-5.3 is multimodal).
        cmd += ["--hf-overrides", json.dumps({"text_config": {
            "index_num_clusters": clusters, "index_num_probes": probes}})]
    if chat_template:
        cmd += ["--chat-template", chat_template]
    return cmd


def ruler_samples(cfg: dict, length: int, tasks: list[str],
                  indices: list[int]) -> list[dict]:
    """RULER samples ``indices`` (row numbers) of each task at ``length``."""
    out = []
    for task in tasks:
        path = os.path.join(cfg["ruler_dir"], task, f"validation_{length}.jsonl")
        with open(path) as f:
            rows = [json.loads(line) for _, line in zip(range(max(indices) + 1), f)]
        for i in indices:
            ex = rows[i]
            out.append({"task": task, "idx": i, "L": length,
                        "label": f"{task}_{i}", "gold": ex["outputs"],
                        "input": ex["input"],
                        "answer_prefix": ex.get("answer_prefix", "")})
    return out


def round_robin_samples(cfg: dict, length: int, tasks: list[str], n: int) -> list[dict]:
    """``n`` distinct samples cycling over ``tasks``: sample 0 of each, then 1..."""
    per_task = -(-n // len(tasks))
    by_task = [ruler_samples(cfg, length, [t], list(range(per_task))) for t in tasks]
    order = [s for i in range(per_task) for s in (b[i] for b in by_task)]
    return order[:n]
