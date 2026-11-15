# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment 5: E-means vs the fused k-means for the IVF build.

Replays experiment 2's captured keys (real GLM-5.3 indexer keys of layers 3,
23 and 43, and the final prefill chunk's queries) on one GPU. For every input
and build variant (``emeans.variants`` in runs.yaml: the algorithm and its
passes), two scenarios:

- ``oneshot``: the whole prefix clustered at once (a cold build), timed over
  ``build_reps``;
- ``chunked``: the served schedule, ``chunk_tokens`` per prefill chunk (cold,
  then appends and warm rebuilds at power-of-two key counts and the last
  chunk), each chunk's build timed once, after an untimed pass.

After each, on the final IVF state: the mean cosine of each key to its
centroid, cluster imbalance (k * sum(size^2) / n^2) and empty clusters; then,
per probe share, the selection time and the recall of the exact top-k (the
kpool dense indexer over every pool), for ``rows`` final-chunk queries that
see every key.

Output: ``<out>/results/{oneshot,chunked}.g<shard>.jsonl``; reruns skip
recorded keys. ``--ref-dir`` (a directory holding ``faiss_emeans_ref``, the
FAISS reference's torch modules) adds a check that this build's E-means
centroids equal the reference's on each one-shot input.
"""

from __future__ import annotations

import argparse
import dataclasses
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


def _ms(fn) -> float:
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    fn()
    e.record()
    e.synchronize()
    return s.elapsed_time(e)


def _config(bench, v: dict):
    c = bench.config
    return dataclasses.replace(
        c,
        kmeans_algo=v["algo"],
        kmeans_iters=v.get("iters", c.kmeans_iters),
        kmeans_warm_iters=v.get("warm_iters", c.kmeans_warm_iters),
        emeans_epochs=float(v.get("epochs", c.emeans_epochs)),
        emeans_warm_epochs=float(v.get("warm_epochs", c.emeans_warm_epochs)),
    )


def _build(bench, n_keys: int, n_prev: int, final: bool) -> None:
    from vllm.models.glm5next.nvidia import ivf_sparse_indexer as ivf

    bench.layout = ivf.build_ivf_index(
        key=bench.key1, state=bench.state1, batch_rows_cpu=[0], n_keys_cpu=[n_keys],
        num_batch_rows=1, config=bench.config, n_prev_keys_cpu=[n_prev],
        final_cpu=[final])


def _state_metrics(bench) -> dict:
    from vllm.models.glm5next.nvidia.ops import ivf_emeans as em

    n, k = bench.n, min(bench.config.num_clusters, bench.n)
    x = em.gather_keys(bench.key1, 0, n).float()
    cen = torch.empty((k, em.HEAD_DIM), dtype=torch.float32, device=x.device)
    em._records(bench.state1, 0, k, cen, cen, mode=0)
    cid = bench.cid.view(-1)[:n].long()
    cos = (x * cen[cid]).sum(1) / torch.linalg.vector_norm(x, dim=1).clamp_min(1e-30)
    sizes = torch.bincount(cid, minlength=k).double()
    return {"mean_cos": round(cos.mean().item(), 6),
            "imbalance": round((k * (sizes**2).sum() / n**2).item(), 4),
            "empty": int((sizes == 0).sum()), "max_size": int(sizes.max())}


def _selection(bench, q, w, exact, grid: dict, shares: list[int], reps: int) -> dict:
    out = {}
    base = bench.config
    for share in shares:
        p = grid["probes"][share]
        bench.config = dataclasses.replace(base, num_probes=p)
        sel = torch.full_like(exact, -1)
        bench.select_prefill(q, w, sel)
        out[f"recall_p{share}"] = rb._agreement(sel, exact)
        t = [_ms(lambda: bench.select_prefill(q, w, sel)) for _ in range(reps + 1)][1:]
        out[f"select_ms_p{share}"] = round(statistics.median(t), 4)
    bench.config = base
    return out


def _ref_check(ref_mod, bench, v: dict) -> dict:
    """This build's E-means against FAISS's on the same rows and seed."""
    from vllm.models.glm5next.nvidia.ops import ivf_emeans as em

    n, k = bench.n, min(bench.config.num_clusters, bench.n)
    x = em.gather_keys(bench.key1, 0, n)
    seed = bench.config.kmeans_seed
    unit, _, _ = em.emeans_spherical(x, k, float(v["epochs"]), seed)
    ref = ref_mod.Emeans(128, k, nepochs=float(v["epochs"]), seed=seed, spherical=True,
                         assignment_backend="native")
    c_ref, _ = ref.train(x)
    cos = (unit * c_ref.float()).sum(1)
    return {"ref_min_cos": round(cos.min().item(), 6),
            "ref_mean_cos": round(cos.mean().item(), 6)}


def run_input(cfg: dict, inp: dict, writer, done: set, ref_mod) -> None:
    E = cfg["emeans"]
    grid = cfg["ivf_grid"][inp["length"]]
    clusters = grid["clusters"]
    probes0 = grid["probes"][E["probe_shares"][0]]
    kpool = cfg["reorder"]["replay"]["kpool"]
    chunk = E["chunk_tokens"] // kpool
    base = rb._Bench(cfg, inp, clusters, probes0)
    q, w = base.queries(E["rows"])
    exact = torch.full((q.shape[0], base.topk), -1, dtype=torch.int32, device="cuda")
    base.kpool_dense(q, w, exact)
    n = base.n
    base = None
    head = {"label": inp["label"], "layer": inp["layer"], "length": inp["length"],
            "n_keys": n, "clusters": clusters, "rows": int(q.shape[0])}
    for v in E["variants"]:
        key = f"oneshot|{inp['label']}|{inp['layer']}|{v['name']}"
        if key not in done:
            t0 = time.time()
            bench = rb._Bench(cfg, inp, clusters, probes0)
            bench.config = _config(bench, v)
            _build(bench, n, 0, True)  # compile
            t = [_ms(lambda: _build(bench, n, 0, True)) for _ in range(E["build_reps"])]
            rec = {"key": key, **head, "variant": v["name"], **v,
                   "build_ms": round(statistics.median(t), 4),
                   **_state_metrics(bench),
                   **_selection(bench, q, w, exact, grid, E["probe_shares"],
                                E["select_reps"])}
            if ref_mod is not None and v["algo"] == "emeans":
                rec.update(_ref_check(ref_mod, bench, v))
            rec["wall_s"] = round(time.time() - t0, 1)
            writer("oneshot", rec)
            done.add(key)
        key = f"chunked|{inp['label']}|{inp['layer']}|{v['name']}"
        if key not in done:
            t0 = time.time()
            bounds = list(range(chunk, n, chunk)) + [n]
            for timed in (False, True):
                bench = rb._Bench(cfg, inp, clusters, probes0)
                bench.config = _config(bench, v)
                total, full, prev = 0.0, 0, 0
                for i, n_i in enumerate(bounds):
                    last = i == len(bounds) - 1
                    # A full build: first chunk, power-of-two crossing, last chunk
                    # (as build_ivf_index decides; appends otherwise).
                    full += int(prev == 0 or prev.bit_length() != n_i.bit_length()
                                or last)
                    total += _ms(lambda: _build(bench, n_i, prev, last))
                    prev = n_i
            rec = {"key": key, **head, "variant": v["name"], **v,
                   "chunk_tokens": E["chunk_tokens"], "chunks": len(bounds),
                   "full_builds": full, "build_ms": round(total, 4),
                   **_state_metrics(bench),
                   **_selection(bench, q, w, exact, grid, E["probe_shares"],
                                E["select_reps"]),
                   "wall_s": round(time.time() - t0, 1)}
            writer("chunked", rec)
            done.add(key)
        bench = None
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--keys-dir", required=True, help="experiment 2's <out>/keys")
    ap.add_argument("--runs-yaml", default=os.path.join(EXP_DIR, "runs.yaml"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--lengths", help="comma list (smoke tests)")
    ap.add_argument("--max-files", type=int, help="first N inputs (smoke tests)")
    ap.add_argument("--ref-dir", help="directory holding the faiss_emeans_ref package")
    args = ap.parse_args()
    cfg = load(args.runs_yaml)
    from vllm.v1.worker.workspace import init_workspace_manager

    init_workspace_manager(torch.device(0))
    ref_mod = None
    if args.ref_dir:
        sys.path.insert(0, args.ref_dir)
        from faiss_emeans_ref import e_means_torch as ref_mod
    files = rb._captured_inputs(args.keys_dir)
    if args.lengths:
        want = {int(x) for x in args.lengths.split(",")}
        files = [f for f in files
                 if int(os.path.basename(os.path.dirname(os.path.dirname(f)))[1:]) in want]
    if args.max_files:
        files = files[: args.max_files]
    files = files[args.shard :: args.num_shards]
    res = os.path.join(args.out_dir, "results")
    os.makedirs(res, exist_ok=True)
    done: set = set()
    paths = {s: os.path.join(res, f"{s}.g{args.shard}.jsonl")
             for s in ("oneshot", "chunked")}
    for p in paths.values():
        if os.path.exists(p):
            with open(p) as f:
                done.update(json.loads(line)["key"] for line in f if line.strip())

    def writer(scenario: str, rec: dict) -> None:
        with open(paths[scenario], "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[emeans] {rec['key']} build {rec['build_ms']} ms "
              f"recall_p1 {rec.get('recall_p1')} cos {rec['mean_cos']}", flush=True)

    for path in files:
        run_input(cfg, rb._load_captured(path), writer, done, ref_mod)
    print("[emeans] shard done", flush=True)


if __name__ == "__main__":
    main()
