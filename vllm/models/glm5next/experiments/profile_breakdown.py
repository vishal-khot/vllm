# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split one rank's torch-profiler trace of a prefill into model components.

Each GPU kernel is tied to the CPU op that launched it (the trace's
``External id``) and to that op's innermost ``vllm::`` custom-op ancestor
(attention, MoE, the indexer, ... ; compiled regions have none). The trace is
reduced to totals per (custom op, kernel name), which ``categorize`` maps to
components with the ``RULES`` regexes, first match wins. Rerun on a saved
breakdown.json to re-categorize without the trace:

    python profile_breakdown.py <trace.json.gz | breakdown.json> [--out x.json]

Times are kernel durations summed (ms); kernels on side streams can overlap,
so ``kernel_ms`` may exceed ``span_ms`` (first kernel start to last end).
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
from collections import defaultdict

# (component, regex over "<custom op>|<kernel name>", lowercased).
RULES = [
    ("allreduce", r"allreduce|all_reduce|nccl|cross_device|reduce_scatter|all_gather|"
                  r"lamport|oneshot|twoshot"),
    ("moe", r"moe|expert|m_grouped|grouped_topk|topk_softmax|router"),
    ("dsa_indexer", r"indexer|ivf|kpool|kmeans|gscore|mqa_logits|top_k|topk|radix"),
    ("kda_attention", r"kda|gdn|gated_delta|linear_attention|fla_|chunk_|"
                      r"fused_recurrent|causal_conv|conv1d"),
    ("mla_attention", r"mla|attention|flash_attn|fmha|sparse_attn|sparse_prefill"),
    ("dense_gemm", r"gemm|cutlass|xmma|cublas|nvjet|wgmma|matmul|aten::mm|aten::addmm"),
    ("norm_elementwise", r"triton_|norm|rope|rotary|elementwise|vectorized|act_and_mul|"
                         r"silu|quant|copy|fill|cat|index|reduce|memcpy|memset"),
]
# Sub-steps of the indexer, by kernel name (IVF and kpool).
INDEXER_RULES = [
    ("kmeans_build", r"kmeans|append|seed|centroid|cluster_id"),
    ("run_layout", r"run_scan|run_scatter"),
    ("probe_plan", r"probe|group_count|group_plan|group_fill|collect|plan"),
    ("score", r"grouped_score|gscore|score|mqa_logits|logits|cutlass|wgmma"),
    ("topk", r"top_k|topk|radix|select|emit"),
    ("key_write", r"compress|write|cache"),
]


def _classify(rules, text: str) -> str:
    for name, rx in rules:
        if re.search(rx, text):
            return name
    return "other"


def pick_rank0(traces: list[str]) -> str:
    """The TP-rank-0 trace among the per-rank files (vLLM names them by rank)."""
    for pat in (r"tp_?0(?!\d)", r"rank_?0(?!\d)", r"(?<!\d)0(?!\d)"):
        hits = [t for t in traces if re.search(pat, os.path.basename(t))]
        if len(hits) == 1:
            return hits[0]
    return sorted(traces)[0]


def _load(path: str) -> dict:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)


def _owners(events: list[dict]) -> dict[int, tuple[str, str]]:
    """External id -> (innermost vllm:: ancestor or "", the op's own name)."""
    by_thread: dict[tuple, list[dict]] = defaultdict(list)
    for e in events:
        if e.get("ph") == "X" and e.get("cat") in ("cpu_op", "user_annotation"):
            by_thread[(e.get("pid"), e.get("tid"))].append(e)
    out: dict[int, tuple[str, str]] = {}
    for ops in by_thread.values():
        ops.sort(key=lambda e: (e["ts"], -e.get("dur", 0)))
        stack: list[tuple[float, str]] = []  # (end, custom op name or "")
        for e in ops:
            ts, end = e["ts"], e["ts"] + e.get("dur", 0)
            while stack and stack[-1][0] <= ts:
                stack.pop()
            name = e.get("name", "")
            custom = next((n for _, n in reversed(stack) if n), "")
            if name.startswith("vllm::"):
                custom = name
            stack.append((end, name if name.startswith("vllm::") else ""))
            ext = (e.get("args") or {}).get("External id")
            if ext is not None and e.get("cat") == "cpu_op":
                out[ext] = (custom, name)
    return out


def reduce_trace(path: str, data: dict | None = None) -> dict:
    """Totals per (custom op, kernel) and the stream / span figures."""
    events = (data if data is not None else _load(path)).get("traceEvents", [])
    owners = _owners(events)
    pair_us: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0])
    stream_us: dict[str, float] = defaultdict(float)
    t_lo, t_hi = float("inf"), 0.0
    for e in events:
        if e.get("ph") != "X" or e.get("cat") not in ("kernel", "gpu_memcpy",
                                                       "gpu_memset"):
            continue
        args = e.get("args") or {}
        custom, _ = owners.get(args.get("External id"), ("", ""))
        acc = pair_us[(custom, e.get("name", ""))]
        acc[0] += e.get("dur", 0)
        acc[1] += 1
        stream_us[str(args.get("stream"))] += e.get("dur", 0)
        t_lo = min(t_lo, e["ts"])
        t_hi = max(t_hi, e["ts"] + e.get("dur", 0))
    pairs = sorted(([c, k, round(us / 1e3, 4), n] for (c, k), (us, n) in pair_us.items()),
                   key=lambda r: -r[2])
    return {"trace": os.path.basename(path),
            "span_ms": round((t_hi - t_lo) / 1e3, 2) if pairs else 0.0,
            "stream_ms": {s: round(us / 1e3, 2) for s, us in
                          sorted(stream_us.items(), key=lambda kv: -kv[1])},
            "pairs": pairs}


def categorize(red: dict) -> dict:
    total = sum(r[2] for r in red["pairs"]) or 1.0
    cat_ms: dict[str, float] = defaultdict(float)
    sub_ms: dict[str, float] = defaultdict(float)
    owner_ms: dict[str, float] = defaultdict(float)
    top: dict[str, list] = defaultdict(list)
    for custom, kernel, ms, n in red["pairs"]:
        text = f"{custom}|{kernel}".lower()
        cat = _classify(RULES, text)
        cat_ms[cat] += ms
        owner_ms[custom or "(compiled / no custom op)"] += ms
        if cat == "dsa_indexer":
            sub_ms[_classify(INDEXER_RULES, kernel.lower())] += ms
        if len(top[cat]) < 12:
            top[cat].append([kernel[:160], custom, ms, n])

    def table(d):
        return {k: {"ms": round(v, 2), "pct": round(100 * v / total, 2)}
                for k, v in sorted(d.items(), key=lambda kv: -kv[1])}

    return {**{k: red[k] for k in ("trace", "span_ms", "stream_ms")},
            "kernel_ms": round(total, 2),
            "by_category": table(cat_ms),
            "indexer_steps": table(sub_ms),
            "by_custom_op": table(owner_ms),
            "top_kernels": dict(top),
            "pairs": red["pairs"]}


def analyze(path: str) -> dict:
    return categorize(reduce_trace(path))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", help="a rank trace (.json[.gz]) or a breakdown.json")
    ap.add_argument("--out")
    args = ap.parse_args()
    data = _load(args.path)
    bd = categorize(data if "pairs" in data else reduce_trace(args.path, data))
    text = json.dumps(bd, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    head = {k: bd[k] for k in ("kernel_ms", "span_ms", "by_category", "indexer_steps")}
    print(json.dumps(head, indent=2))


if __name__ == "__main__":
    main()
