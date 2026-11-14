# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment 2, step 2: does the cluster-ordered key copy pay for itself?

Calls the v0.32 IVF and kpool functions unchanged, on one GPU, and times each
step with CUDA events by wrapping the module-level names they call:

  a  reorder: ``ivf_run_scan_kernel`` + ``ivf_run_scatter_kernel`` (the
     counting sort and the copy of keys into cluster order, in build_ivf_index)
  b  contiguous scoring of the probed candidates: ``_score_grouped_gemm``
  c  scattered fetch and scoring of the same candidates, no reorder:
     ``_collect_by_scan`` + ``_score_per_row`` (the decode path)
  d  kpool, every key: ``cp_gather_indexer_k_quant_cache`` + DeepGEMM
     ``fp8_fp4_mqa_logits`` inside ``_dense_prefill_topk``

Both IVF modes get the same rows with every key visible (query position = last
pool), so they probe the same clusters; ``match`` is the mean share of each
row's top-k the two modes agree on. Reorder pays off where a + b < c, and
IVF's scoring beats kpool's where a + b < d.

Inputs are captured keys and queries (capture_run.py) and synthetic clustered
keys. Records go to ``<out>/results/<kind>.g<gpu>.jsonl`` as each cell
finishes; a requeued run skips cells already written.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
import sys
import time

import torch

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP_DIR)
from exp_config import load  # noqa: E402

# --------------------------------------------------------------------- timing


class _Recorder:
    """CUDA event pairs per tag for the rep in progress."""

    def __init__(self) -> None:
        self.pairs: dict[str, list] = {}
        self.on = False

    def wrap_fn(self, fn, tag):
        def timed(*a, **kw):
            if not self.on:
                return fn(*a, **kw)
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True)
            s.record()
            out = fn(*a, **kw)
            e.record()
            self.pairs.setdefault(tag, []).append((s, e))
            return out

        return timed

    def wrap_kernel(self, kernel, tag):
        rec = self

        class _Timed:
            def __getitem__(self, grid):
                return rec.wrap_fn(kernel[grid], tag)

            def __getattr__(self, name):
                return getattr(kernel, name)

        return _Timed()

    def collect(self) -> dict[str, float]:
        torch.cuda.synchronize()
        out = {t: sum(s.elapsed_time(e) for s, e in v) for t, v in self.pairs.items()}
        self.pairs = {}
        return out


REC = _Recorder()


def _install_timers() -> None:
    from vllm.models.glm5next.nvidia import ivf_sparse_indexer as ivf
    from vllm.utils import deep_gemm

    ivf.ivf_run_scan_kernel = REC.wrap_kernel(ivf.ivf_run_scan_kernel, "a_scan")
    ivf.ivf_run_scatter_kernel = REC.wrap_kernel(
        ivf.ivf_run_scatter_kernel, "a_scatter")
    ivf._score_grouped_gemm = REC.wrap_fn(ivf._score_grouped_gemm, "b_score")
    ivf._collect_by_scan = REC.wrap_fn(ivf._collect_by_scan, "c_collect")
    ivf._score_per_row = REC.wrap_fn(ivf._score_per_row, "c_score")
    ivf.ops.cp_gather_indexer_k_quant_cache = REC.wrap_fn(
        ivf.ops.cp_gather_indexer_k_quant_cache, "d_gather")
    deep_gemm.fp8_fp4_mqa_logits = REC.wrap_fn(
        deep_gemm.fp8_fp4_mqa_logits, "d_logits")


def _measure(fn, warmup: int, reps: int, total_tag: str) -> dict[str, list[float]]:
    """Per-tag ms of each rep (warm-up reps dropped); ``total_tag`` times fn."""
    samples: dict[str, list[float]] = {}
    for i in range(warmup + reps):
        REC.on = True
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True)
        s.record()
        fn()
        e.record()
        REC.on = False
        got = REC.collect()
        got[total_tag] = s.elapsed_time(e)
        if i >= warmup:
            for t, v in got.items():
                samples.setdefault(t, []).append(v)
    return samples


def _stats(v: list[float]) -> dict:
    v = sorted(v)
    return {"median": round(statistics.median(v), 4),
            "p90": round(v[min(len(v) - 1, math.ceil(0.9 * len(v)) - 1)], 4),
            "n": len(v)}


# --------------------------------------------------------------------- inputs


def _synthetic(cfg: dict, length: int, rows: int) -> dict:
    from vllm.models.glm5next.common.ivf_reference import (
        build_paged_indexer_cache,
        quantize_rows_fp8,
    )

    r = cfg["reorder"]["replay"]
    s = r["synthetic"]
    n = length // r["kpool"]
    g = torch.Generator().manual_seed(s["seed"] + length)
    mu = torch.randn(s["mixture_centers"], 128, generator=g) * s["center_scale"]
    keys = mu[torch.randint(0, s["mixture_centers"], (n,), generator=g)] + torch.randn(
        n, 128, generator=g)
    page = r["page"]
    blocks = -(-n // page)
    # The blocks land in a random order, as in a served cache; the table maps them.
    buf, block_table, _, _ = build_paged_indexer_cache(
        [keys], num_blocks=blocks, block_size=page, generator=g)
    # Queries aimed at the key mixture, as real queries align with their keys.
    q = mu[torch.randint(0, s["mixture_centers"], (rows, r["heads"]), generator=g)]
    q = q + torch.randn(rows, r["heads"], 128, generator=g)
    q, scale = quantize_rows_fp8(q, True)
    w = torch.randn(rows, r["heads"], generator=g) * scale
    return {"kind": "synthetic", "label": f"synthetic_L{length}", "layer": None,
            "length": length, "n_keys": n, "page": page,
            "pages": buf.view(blocks, page, 132), "perm_table": block_table,
            "cid_block": r["cid_block"], "cen_block": r["cen_block"],
            "q_fp8": q.to(torch.float8_e4m3fn).view(torch.uint8), "weights": w}


def _captured_inputs(keys_root: str) -> list[str]:
    return sorted(glob.glob(os.path.join(keys_root, "L*", "*", "L*.pt")))


def _load_captured(path: str) -> dict:
    rec = torch.load(path, map_location="cpu", weights_only=False)
    length = int(os.path.basename(os.path.dirname(os.path.dirname(path)))[1:])
    rec.update(kind="real", length=length, perm_table=None)
    return rec


# ---------------------------------------------------------------------- cells


class _Bench:
    """One input on the GPU: key cache, IVF state and queries."""

    def __init__(self, cfg: dict, inp: dict, clusters: int, probes: int) -> None:
        from vllm.models.glm5next.common.ivf_config import IvfIndexerConfig
        from vllm.models.glm5next.nvidia.ivf_sparse_indexer import (
            IvfKeyView,
            IvfStateView,
        )
        from vllm.models.glm5next.nvidia.ops.ivf_index import REC_BYTES

        r = cfg["reorder"]["replay"]
        dev = torch.device("cuda")
        self.n = n = inp["n_keys"]
        self.kpool = r["kpool"]
        self.topk = r["topk"]
        pages = inp["pages"]
        if inp.get("perm_table") is not None:
            # Synthetic keys sit in permuted blocks; the table says where.
            bt = inp["perm_table"][0, : pages.shape[0]].to(torch.int32)
        else:
            bt = torch.arange(pages.shape[0], dtype=torch.int32)
        self.kv = pages.to(dev).contiguous()
        self.bt1 = bt.view(1, -1).to(dev)
        cid_bs, cpb = inp["cid_block"], inp["cen_block"]
        cid_blocks = -(-n // cid_bs)
        cen_blocks = -(-clusters // cpb)
        self.cid = torch.zeros((cid_blocks, cid_bs, 1), dtype=torch.int16, device=dev)
        self.cen = torch.zeros((cen_blocks, cpb, REC_BYTES), dtype=torch.uint8,
                               device=dev)
        self.cid_bt1 = torch.arange(cid_blocks, dtype=torch.int32, device=dev).view(1, -1)
        self.cen_bt1 = torch.arange(cen_blocks, dtype=torch.int32, device=dev).view(1, -1)
        self.key1 = IvfKeyView.of(self.kv, self.bt1)
        self.state1 = IvfStateView.of(self.cid, self.cen, self.cid_bt1, self.cen_bt1,
                                      clusters)
        self.IvfKeyView, self.IvfStateView = IvfKeyView, IvfStateView
        self.config = IvfIndexerConfig(
            num_clusters=clusters, num_probes=probes, max_clusters=clusters,
            kmeans_iters=r["kmeans_iters"], kmeans_seed=0, topk=self.topk,
            kmeans_stop_fraction=r["kmeans_stop_fraction"], kpool=self.kpool,
            kmeans_warm_iters=r["kmeans_warm_iters"])
        self.q_all = inp["q_fp8"].view(torch.float8_e4m3fn).to(dev)
        self.w_all = inp["weights"].float().to(dev)
        self.layout = None

    def build(self):
        from vllm.models.glm5next.nvidia import ivf_sparse_indexer as ivf

        self.layout = ivf.build_ivf_index(
            key=self.key1, state=self.state1, batch_rows_cpu=[0],
            n_keys_cpu=[self.n], num_batch_rows=1, config=self.config,
            n_prev_keys_cpu=[0], final_cpu=[True])

    def queries(self, rows: int):
        """``rows`` query rows; captured rows repeat when fewer were saved."""
        have = self.q_all.shape[0]
        idx = torch.arange(rows, device=self.q_all.device) % have
        return self.q_all[idx].contiguous(), self.w_all[idx].contiguous()

    def select_prefill(self, q, w, out):
        from vllm.models.glm5next.nvidia import ivf_sparse_indexer as ivf

        rows = q.shape[0]
        dev = q.device
        ivf.select_ivf_topk(
            key=self.key1, state=self.state1, q_fp8=q, weights=w,
            row_batch=torch.zeros(rows, dtype=torch.int32, device=dev),
            query_pos=torch.full((rows,), self.n - 1, dtype=torch.int32, device=dev),
            config=self.config, out=out, topk_backend="auto", layout=self.layout,
            max_width=self.n)

    def decode_views(self, rows: int):
        """Every row is its own batch row over the same pages and state."""
        key = self.IvfKeyView.of(self.kv, self.bt1.expand(rows, -1))
        state = self.IvfStateView.of(
            self.cid, self.cen, self.cid_bt1.expand(rows, -1),
            self.cen_bt1.expand(rows, -1), self.config.num_clusters)
        return key, state

    def select_scattered(self, q, w, out, key, state):
        from vllm.models.glm5next.nvidia import ivf_sparse_indexer as ivf

        rows = q.shape[0]
        dev = q.device
        ivf.select_ivf_topk(
            key=key, state=state, q_fp8=q, weights=w,
            row_batch=torch.arange(rows, dtype=torch.int32, device=dev),
            query_pos=torch.full((rows,), self.n - 1, dtype=torch.int32, device=dev),
            config=self.config, out=out, topk_backend="auto",
            seq_lens=torch.full((rows,), self.n, dtype=torch.int32, device=dev))

    def kpool_dense(self, q, w, out):
        from vllm.models.glm5next.nvidia import ivf_sparse_indexer as ivf

        rows = q.shape[0]
        ivf._dense_prefill_topk(
            key_cache=self.kv, block_table=self.bt1, num_pools=[self.n],
            query_lens=[rows], q_fp8=q, weights=w,
            token_lens=torch.full((rows,), self.n * self.kpool, dtype=torch.int32,
                                  device=q.device),
            out=out, topk=self.topk, kpool=self.kpool)


def _agreement(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean share of each row's selected pools (>= 0) that both selections hold."""
    shares = []
    for x, y in zip(a.cpu(), b.cpu()):
        sx = {int(v) for v in x if v >= 0}
        sy = {int(v) for v in y if v >= 0}
        if sx or sy:
            shares.append(len(sx & sy) / max(len(sx), len(sy)))
    return round(sum(shares) / len(shares), 6) if shares else 1.0


def run_input(cfg: dict, inp: dict, probe_shares: list[int], rows_grid: list[int],
              writer, done: set) -> None:
    r = cfg["reorder"]["replay"]
    grid = cfg["ivf_grid"][inp["length"]]
    clusters = grid["clusters"]
    for share in probe_shares:
        probes = grid["probes"][share]
        bench = None
        for rows in rows_grid:
            key = f"{inp['kind']}|{inp['label']}|{inp['layer']}|{probes}|{rows}"
            if key in done:
                continue
            rec = {"key": key, "kind": inp["kind"], "label": inp["label"],
                   "layer": inp["layer"], "length": inp["length"],
                   "n_keys": inp["n_keys"], "clusters": clusters, "probes": probes,
                   "probe_share": round(probes / clusters, 4), "rows": rows,
                   "captured_rows": int(inp["q_fp8"].shape[0]),
                   "gpu": torch.cuda.get_device_name()}
            t0 = time.time()
            if bench is None:
                bench = _Bench(cfg, inp, clusters, probes)
                bench.build()
            q, w = bench.queries(rows)
            warm, reps = r["warmup"], r["reps"]
            timings: dict[str, dict] = {}
            # a: one reorder per build; the build reruns whole, only a is kept.
            for t, v in _measure(bench.build, warm, reps, "build_total").items():
                timings[t] = _stats(v)
            # b: prefill selection from the cluster-ordered copy.
            out_p = torch.full((rows, bench.topk), -1, dtype=torch.int32,
                               device="cuda")
            for t, v in _measure(lambda: bench.select_prefill(q, w, out_p),
                                 warm, reps, "select_contiguous_total").items():
                timings[t] = _stats(v)
            # c: the same rows through the scattered (decode) path.
            if rows * bench.n * 8 <= r["max_scatter_bytes"]:
                kv, st = bench.decode_views(rows)
                out_d = torch.full((rows, bench.topk), -1, dtype=torch.int32,
                                   device="cuda")
                for t, v in _measure(
                        lambda: bench.select_scattered(q, w, out_d, kv, st),
                        warm, reps, "select_scattered_total").items():
                    timings[t] = _stats(v)
                rec["match"] = _agreement(out_p, out_d)
                out_d = None
            else:
                rec["scattered_skipped"] = "max_scatter_bytes"
            # d: kpool scores every key.
            out_k = torch.full((rows, (bench.topk + 1) * bench.kpool), -1,
                               dtype=torch.int32, device="cuda")
            for t, v in _measure(lambda: bench.kpool_dense(q, w, out_k),
                                 warm, reps, "kpool_total").items():
                timings[t] = _stats(v)
            out_k = out_p = None
            rec["ms"] = timings
            med = {t: s["median"] for t, s in timings.items()}
            a = med.get("a_scan", 0) + med.get("a_scatter", 0)
            b = med.get("b_score")
            c = (med["c_collect"] + med["c_score"]) if "c_score" in med else None
            d = med.get("d_gather", 0) + med.get("d_logits", 0)
            rec["a_reorder_ms"], rec["b_contiguous_ms"] = round(a, 4), b
            rec["c_scattered_ms"] = None if c is None else round(c, 4)
            rec["d_kpool_ms"] = round(d, 4)
            if c is not None and b is not None:
                rec["reorder_pays"] = a + b < c
                saved_per_row = (c - b) / rows
                rec["breakeven_rows"] = (round(a / saved_per_row, 1)
                                         if saved_per_row > 0 else None)
            rec["ivf_beats_kpool_scoring"] = b is not None and a + b < d
            rec["wall_s"] = round(time.time() - t0, 1)
            writer(rec)
            done.add(key)
            torch.cuda.empty_cache()
        bench = None
        torch.cuda.empty_cache()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--runs-yaml", default=os.path.join(EXP_DIR, "runs.yaml"))
    p.add_argument("--kind", choices=["real", "synthetic"], required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--lengths", help="comma list, overrides runs.yaml (smoke tests)")
    p.add_argument("--rows", help="comma list, overrides runs.yaml (smoke tests)")
    p.add_argument("--reps", type=int, help="overrides runs.yaml (smoke tests)")
    args = p.parse_args()
    cfg = load(args.runs_yaml)
    r = cfg["reorder"]["replay"]
    if args.reps:
        r["reps"], r["warmup"] = args.reps, 1
    rows_grid = [int(x) for x in args.rows.split(",")] if args.rows else r["rows"]

    from vllm.v1.worker.workspace import init_workspace_manager

    init_workspace_manager(torch.device(0))
    _install_timers()

    res_dir = os.path.join(args.out_dir, "results")
    os.makedirs(res_dir, exist_ok=True)
    path = os.path.join(res_dir, f"{args.kind}.g{args.shard}.jsonl")
    done: set = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["key"])
                except (ValueError, KeyError):
                    pass

    def writer(rec: dict) -> None:
        with open(path, "a") as f:
            f.write(f"\n{json.dumps(rec)}\n")
        print(f"[reorder g{args.shard}] {rec['key']}: a={rec['a_reorder_ms']} "
              f"b={rec['b_contiguous_ms']} c={rec['c_scattered_ms']} "
              f"d={rec['d_kpool_ms']} match={rec.get('match')}", flush=True)

    if args.kind == "real":
        files = _captured_inputs(os.path.join(args.out_dir, "keys"))
        if not files:
            raise SystemExit("no captured keys under <out>/keys; run capture first")
        mine = files[args.shard::args.num_shards]
        lengths = ({int(x) for x in args.lengths.split(",")} if args.lengths
                   else None)
        for f in mine:
            inp = _load_captured(f)
            if lengths is None or inp["length"] in lengths:
                run_input(cfg, inp, r["probe_shares"], rows_grid, writer, done)
    else:
        lengths = ([int(x) for x in args.lengths.split(",")] if args.lengths
                   else r["synthetic"]["lengths"])
        # (length, probe share) items, longest first, dealt round-robin so the
        # long lengths spread over the shards.
        items = [(L, s) for L in sorted(lengths, reverse=True)
                 for s in r["probe_shares"]][args.shard::args.num_shards]
        for length in sorted({L for L, _ in items}):
            inp = _synthetic(cfg, length, max(rows_grid))
            shares = [s for L, s in items if L == length]
            run_input(cfg, inp, shares, rows_grid, writer, done)
    print(f"[reorder g{args.shard}] {args.kind} complete -> {path}", flush=True)


if __name__ == "__main__":
    main()
