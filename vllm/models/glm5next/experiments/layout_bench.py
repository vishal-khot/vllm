# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment 3: does the cluster-contiguous key layout make scoring faster?

Experiment 2 compared v0.32's prefill path (copy keys into cluster order, then
one grouped GEMM) with its decode path (per-row gather and scoring): two
layouts and two scoring schemes at once. This holds the scoring fixed and
changes only where the keys are read from. Inputs are experiment 2's captured
keys and queries (read only, from ``--keys-dir``); one process per GPU.

Part A, scoring (per input x probe share x rows), each hot and with L2 flushed:
  b_cute     v0.32 as served (CuTe grouped GEMM on the contiguous copy)
  b_triton   v0.32's Triton grouped GEMM on the contiguous copy
  b_inplace  the same Triton kernel, keys read in place from the paged cache
             (``layout_ops.grouped_score_inplace_kernel``)
  and the builds' copy step: ``scatter_full`` (v0.32: positions, keys and
  scales into cluster order) vs ``scatter_pos`` (positions only, all the
  in-place scoring needs). Same-kernel verdict:
  contiguous = scatter_full + b_triton  vs  in place = scatter_pos + b_inplace.
  ``match_*`` are top-k agreements (in place vs Triton, Triton vs CuTe).

Part B, raw reads (per input): every key once, contiguous from the copy,
  gathered in cluster order from the cache, and in token order from the cache.

Part C, chunked prefill (per input x probe share): the prompt is fed in
  ``chunk_tokens`` chunks as served (append/warm builds, causal rows); per
  chunk the build's copy and the three scorings are timed and summed, so the
  copy is charged every time the prefix is re-copied.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP_DIR)
import reorder_bench as rb  # noqa: E402
from exp_config import load  # noqa: E402

REC = rb.REC
MODE = {"score": "cute", "scatter": "full"}
CUR = {"key": None}
FLUSH: dict = {}


def _install() -> None:
    """Route v0.32's grouped scoring and scatter through mode switches, then
    let reorder_bench add its timers around them."""
    from vllm.models.glm5next.nvidia import ivf_sparse_indexer as ivf

    import layout_ops

    orig_score = ivf._score_grouped_gemm
    inplace = layout_ops.make_inplace_scorer(lambda: CUR["key"])

    def score(**kw):
        m = MODE["score"]
        if m == "inplace":
            return inplace(**kw)
        saved = ivf.TUNING.gemm_impl
        ivf.TUNING.gemm_impl = "cute" if m == "cute" else "triton"
        try:
            return orig_score(**kw)
        finally:
            ivf.TUNING.gemm_impl = saved

    orig_scatter = ivf.ivf_run_scatter_kernel

    class _Scatter:
        def __getitem__(self, grid):
            k = (orig_scatter if MODE["scatter"] == "full"
                 else layout_ops.run_scatter_positions_kernel)
            return k[grid]

    ivf._score_grouped_gemm = score
    ivf.ivf_run_scatter_kernel = _Scatter()
    rb._install_timers()


def _flush() -> None:
    FLUSH["buf"].zero_()


def _measure(fn, warmup: int, reps: int, cold: bool) -> dict[str, list[float]]:
    """rb._measure, with L2 flushed (outside the timed region) when ``cold``."""
    samples: dict[str, list[float]] = {}
    for i in range(warmup + reps):
        if cold:
            _flush()
        REC.on = True
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        REC.on = False
        got = REC.collect()
        got["total"] = s.elapsed_time(e)
        if i >= warmup:
            for t, v in got.items():
                samples.setdefault(t, []).append(v)
    return samples


def _med(samples: dict, tag: str) -> float | None:
    v = samples.get(tag)
    return round(statistics.median(v), 4) if v else None


def _select(bench, q, w, out, query_pos, n_keys):
    from vllm.models.glm5next.nvidia import ivf_sparse_indexer as ivf

    rows = q.shape[0]
    ivf.select_ivf_topk(
        key=bench.key1, state=bench.state1, q_fp8=q, weights=w,
        row_batch=torch.zeros(rows, dtype=torch.int32, device=q.device),
        query_pos=query_pos, config=bench.config, out=out, topk_backend="auto",
        layout=bench.layout, max_width=n_keys)


def _build(bench, n_keys: int, n_prev: int, final: bool) -> None:
    from vllm.models.glm5next.nvidia import ivf_sparse_indexer as ivf

    bench.layout = ivf.build_ivf_index(
        key=bench.key1, state=bench.state1, batch_rows_cpu=[0],
        n_keys_cpu=[n_keys], num_batch_rows=1, config=bench.config,
        n_prev_keys_cpu=[n_prev], final_cpu=[final])


def _scores(bench, q, w, query_pos, n_keys, warm, reps, cold) -> tuple[dict, dict]:
    """Median ms of the grouped scoring per mode, and each mode's top-k."""
    ms, outs = {}, {}
    for m in ("cute", "triton", "inplace"):
        MODE["score"] = m
        out = torch.full((q.shape[0], bench.topk), -1, dtype=torch.int32,
                         device="cuda")
        s = _measure(lambda: _select(bench, q, w, out, query_pos, n_keys),
                     warm, reps, cold)
        ms[m] = _med(s, "b_score")
        outs[m] = out
    MODE["score"] = "cute"
    return ms, outs


def _scatter_ms(bench, warm, reps, cold) -> dict:
    """Median ms of the full-context build's scan and scatter, per scatter mode;
    ends on a full build so the layout holds keys again."""
    got = {}
    for m in ("pos", "full"):
        MODE["scatter"] = m
        s = _measure(bench.build, warm, reps, cold)
        got[m] = {"scan": _med(s, "a_scan"), "scatter": _med(s, "a_scatter"),
                  "build": _med(s, "total")}
    MODE["scatter"] = "full"
    return got


# ---------------------------------------------------------------------- parts


def part_a(cfg, inp, writer, done) -> None:
    L = cfg["layout"]
    grid = cfg["ivf_grid"][inp["length"]]
    clusters = grid["clusters"]
    for share in L["probe_shares"]:
        probes = grid["probes"][share]
        bench = None
        for rows in L["rows"]:
            key = f"A|{inp['label']}|{inp['layer']}|{probes}|{rows}"
            if key in done:
                continue
            t0 = time.time()
            if bench is None:
                bench = rb._Bench(cfg, inp, clusters, probes)
                bench.build()
                CUR["key"] = bench.key1
            q, w = bench.queries(rows)
            qp = torch.full((rows,), bench.n - 1, dtype=torch.int32, device="cuda")
            rec = {"key": key, "part": "A", "label": inp["label"],
                   "layer": inp["layer"], "length": inp["length"],
                   "n_keys": bench.n, "clusters": clusters, "probes": probes,
                   "probe_share": round(probes / clusters, 4), "rows": rows}
            for cold in (False, True):
                tag = "cold" if cold else "hot"
                ms, outs = _scores(bench, q, w, qp, bench.n, L["warmup"],
                                   L["reps"], cold)
                for m, v in ms.items():
                    rec[f"b_{m}_{tag}_ms"] = v
                if not cold:
                    rec["match_inplace_triton"] = rb._agreement(
                        outs["inplace"], outs["triton"])
                    rec["match_triton_cute"] = rb._agreement(
                        outs["triton"], outs["cute"])
                outs = None
                sc = _scatter_ms(bench, L["warmup"], L["reps"], cold)
                rec[f"scan_{tag}_ms"] = sc["full"]["scan"]
                rec[f"scatter_full_{tag}_ms"] = sc["full"]["scatter"]
                rec[f"scatter_pos_{tag}_ms"] = sc["pos"]["scatter"]
                contig = rec[f"scatter_full_{tag}_ms"] + rec[f"b_triton_{tag}_ms"]
                inpl = rec[f"scatter_pos_{tag}_ms"] + rec[f"b_inplace_{tag}_ms"]
                rec[f"contiguous_{tag}_ms"] = round(contig, 4)
                rec[f"inplace_{tag}_ms"] = round(inpl, 4)
                rec[f"contiguous_speedup_{tag}"] = round(inpl / contig, 4)
            rec["wall_s"] = round(time.time() - t0, 1)
            writer("A", rec)
            done.add(key)
        bench = None
        CUR["key"] = None
        torch.cuda.empty_cache()


def part_b(cfg, inp, writer, done) -> None:
    import layout_ops

    L = cfg["layout"]
    key = f"B|{inp['label']}|{inp['layer']}"
    if key in done:
        return
    grid = cfg["ivf_grid"][inp["length"]]
    bench = rb._Bench(cfg, inp, grid["clusters"], grid["probes"][0])
    bench.build()
    n = bench.n
    lay = bench.layout
    out = torch.empty(n // 64 + 1, dtype=torch.int32, device="cuda")
    srcs = {0: lay.run_keys.view(torch.uint8), 1: bench.kv, 2: bench.kv}
    names = {0: "contiguous_copy", 1: "cache_cluster_order", 2: "cache_token_order"}
    rec = {"key": key, "part": "B", "label": inp["label"], "layer": inp["layer"],
           "length": inp["length"], "n_keys": n, "bytes": n * 128}
    for cold in (False, True):
        tag = "cold" if cold else "hot"
        for mode, src in srcs.items():
            def fn(mode=mode, src=src):
                layout_ops.read_keys(mode, src, lay.run_pos, bench.bt1[0],
                                     bench.key1.page_stride, bench.key1.page, n, out)
            s = _measure(fn, L["warmup"], L["read_reps"], cold)
            t = _med(s, "total")
            rec[f"{names[mode]}_{tag}_ms"] = t
            rec[f"{names[mode]}_{tag}_gbps"] = round(n * 128 / (t * 1e6), 1)
    writer("B", rec)
    done.add(key)
    torch.cuda.empty_cache()


def part_c(cfg, inp, writer, done) -> None:
    L = cfg["layout"]
    grid = cfg["ivf_grid"][inp["length"]]
    clusters = grid["clusters"]
    kpool = cfg["reorder"]["replay"]["kpool"]
    chunk = L["chunk_tokens"] // kpool
    for share in L["chunk_probe_shares"]:
        probes = grid["probes"][share]
        key = f"C|{inp['label']}|{inp['layer']}|{probes}"
        if key in done:
            continue
        t0 = time.time()
        n = inp["n_keys"]
        bounds = list(range(chunk, n, chunk)) + [n]
        sums = {"chunks": len(bounds)}

        def add(k, v):
            if v is not None:
                sums[k] = sums.get(k, 0.0) + v

        # Builds are timed once per chunk (a build changes the IVF state, so
        # it can't be repeated); each pass starts from an empty state, and an
        # untimed pass per scatter mode compiles every kernel shape first.
        for timed in (False, True):
            for m in ("pos", "full"):
                MODE["scatter"] = m
                bench = rb._Bench(cfg, inp, clusters, probes)
                CUR["key"] = bench.key1
                prev = 0
                for i, n_i in enumerate(bounds):
                    s = _measure(lambda: _build(bench, n_i, prev, i == len(bounds) - 1),
                                 0, 1, False)
                    if timed:
                        add(f"scatter_{m}_ms", _med(s, "a_scatter"))
                        add(f"scan_{m}_ms", _med(s, "a_scan"))
                        add(f"build_{m}_ms", _med(s, "total"))
                    if timed and m == "full":
                        rows = (n_i - prev) * kpool
                        q, w = bench.queries(rows)
                        vis = torch.arange(prev * kpool + 1, n_i * kpool + 1,
                                           dtype=torch.int32, device="cuda")
                        qp = vis // kpool - 1
                        ms, outs = _scores(bench, q, w, qp, n_i, 1, L["chunk_reps"],
                                           False)
                        for mm, v in ms.items():
                            add(f"b_{mm}_ms", v)
                        if i == len(bounds) - 1:
                            sums["match_inplace_triton_last"] = rb._agreement(
                                outs["inplace"], outs["triton"])
                        outs = None
                    prev = n_i
                bench = None
                CUR["key"] = None
                torch.cuda.empty_cache()
        MODE["scatter"] = "full"
        sums = {k: round(v, 4) if isinstance(v, float) else v for k, v in sums.items()}
        contig = sums["scatter_full_ms"] + sums["b_triton_ms"]
        inpl = sums["scatter_pos_ms"] + sums["b_inplace_ms"]
        rec = {"key": key, "part": "C", "label": inp["label"], "layer": inp["layer"],
               "length": inp["length"], "n_keys": n, "clusters": clusters,
               "probes": probes, "probe_share": round(probes / clusters, 4),
               "chunk_tokens": L["chunk_tokens"], **sums,
               "contiguous_ms": round(contig, 4), "inplace_ms": round(inpl, 4),
               "contiguous_speedup": round(inpl / contig, 4),
               "wall_s": round(time.time() - t0, 1)}
        writer("C", rec)
        done.add(key)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--keys-dir", required=True, help="experiment 2's <out>/keys")
    p.add_argument("--runs-yaml", default=os.path.join(EXP_DIR, "runs.yaml"))
    p.add_argument("--parts", default="A,B,C")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--lengths", help="comma list (smoke tests)")
    p.add_argument("--rows", help="comma list, overrides part A rows (smoke tests)")
    p.add_argument("--reps", type=int, help="overrides reps (smoke tests)")
    p.add_argument("--max-files", type=int, help="first N inputs (smoke tests)")
    args = p.parse_args()
    cfg = load(args.runs_yaml)
    L = cfg["layout"]
    if args.reps:
        L["reps"] = L["read_reps"] = L["chunk_reps"] = args.reps
        L["warmup"] = 1
    if args.rows:
        L["rows"] = [int(x) for x in args.rows.split(",")]
    parts = args.parts.split(",")

    from vllm.v1.worker.workspace import init_workspace_manager

    init_workspace_manager(torch.device(0))
    _install()
    FLUSH["buf"] = torch.empty(int(L["flush_bytes"]), dtype=torch.uint8,
                               device="cuda")

    res_dir = os.path.join(args.out_dir, "results")
    os.makedirs(res_dir, exist_ok=True)
    paths = {pt: os.path.join(res_dir, f"{pt}.g{args.shard}.jsonl") for pt in parts}
    done: set = set()
    for path in paths.values():
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    try:
                        done.add(json.loads(line)["key"])
                    except (ValueError, KeyError):
                        pass

    gpu = torch.cuda.get_device_name()

    def writer(part: str, rec: dict) -> None:
        rec["gpu"] = gpu
        with open(paths[part], "a") as f:
            f.write(f"\n{json.dumps(rec)}\n")
        brief = {k: v for k, v in rec.items()
                 if k.startswith(("contiguous_speedup", "match", "b_", "scatter_"))
                 or k.endswith("gbps")}
        print(f"[layout g{args.shard}] {rec['key']}: {json.dumps(brief)}", flush=True)

    files = rb._captured_inputs(args.keys_dir)
    if not files:
        raise SystemExit(f"no captured keys under {args.keys_dir}")
    if args.lengths:
        keep = {int(x) for x in args.lengths.split(",")}
        files = [f for f in files
                 if int(os.path.basename(os.path.dirname(os.path.dirname(f)))[1:])
                 in keep]
    if args.max_files:
        files = files[: args.max_files]
    for f in files[args.shard::args.num_shards]:
        inp = rb._load_captured(f)
        if "B" in parts:
            part_b(cfg, inp, writer, done)
        if "A" in parts:
            part_a(cfg, inp, writer, done)
        if "C" in parts:
            part_c(cfg, inp, writer, done)
        inp = None
        torch.cuda.empty_cache()
    print(f"[layout g{args.shard}] complete", flush=True)


if __name__ == "__main__":
    main()
