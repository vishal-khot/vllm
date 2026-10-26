# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""glm5next IVF indexer Triton kernels against the torch reference."""

import dataclasses
from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = [
    pytest.mark.skipif(
        not current_platform.is_cuda()
        or not current_platform.has_device_capability(89),
        reason="needs an fp8-capable CUDA GPU",
    ),
    pytest.mark.usefixtures("workspace_init"),
]

from vllm.models.glm5next.common.ivf_backend import (  # noqa: E402
    IvfIndexerMetadata,
    IvfIndexerMetadataBuilder,
    IvfStateMetadata,
)
from vllm.models.glm5next.common.ivf_config import IvfIndexerConfig  # noqa: E402
from vllm.models.glm5next.common.ivf_reference import (  # noqa: E402
    KmeansResult,
    build_paged_indexer_cache,
    cluster_count,
    decode_assign_reference,
    exact_topk_reference,
    ivf_select_one_reference,
    key_scores_reference,
    kmeans_append_reference,
    kmeans_reference,
    quantize_rows_fp8,
    tie_aware_recall,
)
from vllm.models.glm5next.nvidia import ivf_sparse_indexer  # noqa: E402
from vllm.models.glm5next.nvidia.ivf_sparse_indexer import (  # noqa: E402
    IvfKeyView,
    IvfStateView,
    _prefill_rows,
    append_ivf_decode_keys,
    build_ivf_index,
    select_ivf_topk,
    sparse_attn_indexer_ivf,
)
from vllm.models.glm5next.nvidia.ops.ivf_index import REC_BYTES  # noqa: E402
from vllm.models.glm5next.nvidia.ops.kpool_compress import (  # noqa: E402
    expand_pools_and_append_tail,
    kpool_compress_and_write_cache,
)

DEV = "cuda"
HEADS = 16
TOPK = 2048
PAGE = 64
C = 64
# Lengths cover a single key, under top-k, a page boundary and long rows.
LENGTHS = [1, 70, 2100, 9000]
MAX_LEN = 16384
# Small state blocks so the kernels cross block boundaries.
CID_BS = 512
CPB = 20


def _keys(n, seed):
    g = torch.Generator().manual_seed(seed)
    mu = torch.randn(16, 128, generator=g) * 3
    return mu[torch.randint(0, 16, (n,), generator=g)] + torch.randn(
        n, 128, generator=g
    )


def _config(**kw):
    kw.setdefault("num_clusters", C)
    kw.setdefault("num_probes", C // 4)
    kw.setdefault("kmeans_iters", 5)
    return IvfIndexerConfig(
        max_clusters=C,
        kmeans_seed=0,
        topk=TOPK,
        **kw,
    )


def _block_table(num_rows, blocks_per_row, num_blocks, seed):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_blocks, generator=g)[: num_rows * blocks_per_row]
    return perm.view(num_rows, blocks_per_row).to(torch.int32)


class Setup:
    def __init__(self, config: IvfIndexerConfig, extra_decode_key: bool = False):
        self.config = config
        # With extra_decode_key the cache holds one more key than is indexed.
        cached = [n + int(extra_decode_key) for n in LENGTHS]
        keys = [_keys(n, seed=i) for i, n in enumerate(cached)]
        buf, bt, fp8_vals, scales = build_paged_indexer_cache(
            keys,
            num_blocks=sum((n + PAGE - 1) // PAGE for n in cached) + 8,
            block_size=PAGE,
        )
        self.fp8_vals, self.scales = fp8_vals, scales
        self.key = IvfKeyView.of(buf.view(-1, PAGE, 132).to(DEV), bt.to(DEV))

        rows = len(LENGTHS)
        cid_blocks = MAX_LEN // CID_BS
        cen_blocks = (C + CPB - 1) // CPB
        g = torch.Generator().manual_seed(3)
        # Garbage-filled state: the kernels must never depend on prior contents.
        self.cid = torch.randint(
            0, C, (rows * cid_blocks + 3, CID_BS, 1), generator=g, dtype=torch.int16
        ).to(DEV)
        self.cen = torch.randint(
            0,
            256,
            (rows * cen_blocks + 3, CPB, REC_BYTES),
            generator=g,
            dtype=torch.uint8,
        ).to(DEV)
        self.cid_bt = _block_table(rows, cid_blocks, self.cid.shape[0], 1).to(DEV)
        self.cen_bt = _block_table(rows, cen_blocks, self.cen.shape[0], 2).to(DEV)
        self.state = IvfStateView.of(self.cid, self.cen, self.cid_bt, self.cen_bt, C)

    def build(self, n_keys=LENGTHS, n_prev=None):
        return build_ivf_index(
            key=self.key,
            state=self.state,
            batch_rows_cpu=list(range(len(LENGTHS))),
            n_keys_cpu=list(n_keys),
            num_batch_rows=len(LENGTHS),
            config=self.config,
            n_prev_keys_cpu=n_prev,
        )

    def kmeans_from_state(self, b, n_keys):
        cen, cen_bt = self.cen.cpu(), self.cen_bt.cpu()
        recs = torch.stack([cen[cen_bt[b, c // CPB], c % CPB] for c in range(C)])
        size = recs[:, 256:260].contiguous().view(torch.int32).flatten()
        lengths = recs[:, 260:264].contiguous().view(torch.float32).flatten()
        cid, cid_bt = self.cid.cpu(), self.cid_bt.cpu()
        pos = torch.arange(n_keys)
        cids = cid[cid_bt[b, pos // CID_BS].long(), pos % CID_BS, 0]
        return KmeansResult(
            centroids=recs[:, :256].contiguous().view(torch.bfloat16).float(),
            cluster_of_pos=cids.long(),
            sizes=size.clamp(min=0).long(),
            n_clusters=int((size >= 0).sum()),
            lengths=lengths,
        )


class Queries(NamedTuple):
    fp8: torch.Tensor  # [rows, H, 128] e4m3 on the GPU, as the kernels take it
    w: torch.Tensor  # [rows, H] fp32 head weights on the GPU
    q: torch.Tensor  # [rows, H, 128] fp8 values as fp32 on the CPU
    w_cpu: torch.Tensor


def _queries(num_rows, seed=0, heads=HEADS) -> Queries:
    """fp8 query heads and signed head weights with the query scales folded
    in, as the indexer forward builds them."""
    g = torch.Generator().manual_seed(seed)
    q, scale = quantize_rows_fp8(torch.randn(num_rows, heads, 128, generator=g), True)
    w = torch.randn(num_rows, heads, generator=g) * scale
    return Queries(q.to(torch.float8_e4m3fn).to(DEV), w.to(DEV), q, w)


def _cosines(k: torch.Tensor, km: KmeansResult) -> torch.Tensor:
    """[N, C] cosine of every key to every live centroid (-inf elsewhere)."""
    sim = k @ km.centroids.T
    sim = sim / k.norm(dim=1, keepdim=True).clamp(min=1e-12)
    sim[:, km.n_clusters :] = float("-inf")
    return sim


@pytest.mark.parametrize("iters", [0, 5])
@pytest.mark.parametrize("num_clusters", [C, 20])
def test_build_matches_reference_and_is_deterministic(iters, num_clusters):
    """Fewer keys than num_clusters (n = 1) clamps the count to n."""
    setup = Setup(_config(kmeans_iters=iters, num_clusters=num_clusters))
    setup.build()
    first = (setup.cid.clone(), setup.cen.clone())
    setup.build()
    assert torch.equal(first[0], setup.cid) and torch.equal(first[1], setup.cen)

    for b, n in enumerate(LENGTHS):
        k, k_scale = setup.fp8_vals[b][:n], setup.scales[b][:n]
        ref = kmeans_reference(
            k,
            k_scale,
            max_clusters=C,
            num_clusters=num_clusters,
            iters=iters,
            seed=0,
        )
        got = setup.kmeans_from_state(b, n)
        assert got.n_clusters == ref.n_clusters == cluster_count(n, num_clusters, C)
        assert int(got.sizes.sum()) == n
        assert torch.equal(got.sizes, torch.bincount(got.cluster_of_pos, minlength=C))
        # Every key sits in its most cosine-similar cluster under the kernel's
        # own centroids, up to near-ties (bf16 MMA with fp32 sums vs torch).
        sim = _cosines(k, got)
        chosen = sim.gather(1, got.cluster_of_pos[:, None]).squeeze(1)
        gap = (sim.max(dim=1).values - chosen).max().item()
        assert gap <= 1e-4, (b, gap)
        if iters == 0:
            # Same seeded initialization. fp32 norms sum in a different order,
            # so an element may round to the neighbouring bf16 value; the
            # centroid directions match. With the nearest-centroid check above
            # this validates init and assignment.
            live = slice(0, ref.n_clusters)
            cos = torch.nn.functional.cosine_similarity(
                got.centroids[live], ref.centroids[live], dim=1
            )
            assert cos.min().item() > 0.999, (b, cos.min().item())
        else:
            # Centroid sums round in a different order than the reference's,
            # and with more clusters than content modes the trajectories part
            # on near-ties; the clustering quality must still match.
            got_fit = chosen.mean().item()
            ref_fit = _cosines(k, ref).max(dim=1).values.mean().item()
            assert got_fit >= ref_fit - 0.01, (b, got_fit, ref_fit)


def test_prefill_chunk_resumes_from_previous_centroids():
    """A chunk after the first starts from the request's stored centroids when
    its cluster count is unchanged, else it seeds afresh. With no iterations
    the build is just the final assignment against those centroids."""
    setup = Setup(_config())
    first = [max(n // 2, 1) for n in LENGTHS]
    setup.build(n_keys=first)
    before = [setup.kmeans_from_state(b, n) for b, n in enumerate(first)]
    setup.config = dataclasses.replace(setup.config, kmeans_iters=0)
    setup.build(n_prev=first)
    for b, n in enumerate(LENGTHS):
        got = setup.kmeans_from_state(b, n)
        if cluster_count(first[b], C, C) == cluster_count(n, C, C):
            assert torch.equal(got.centroids, before[b].centroids), b
        else:
            k, k_scale = setup.fp8_vals[b][:n], setup.scales[b][:n]
            seeded = kmeans_reference(
                k, k_scale, max_clusters=C, num_clusters=C, iters=0, seed=0
            )
            live = slice(0, seeded.n_clusters)
            cos = torch.nn.functional.cosine_similarity(
                got.centroids[live], seeded.centroids[live], dim=1
            )
            assert cos.min().item() > 0.999, b
        sim = _cosines(setup.fp8_vals[b][:n], got)
        chosen = sim.gather(1, got.cluster_of_pos[:, None]).squeeze(1)
        assert (sim.max(dim=1).values - chosen).max().item() <= 1e-4, b


def test_prefill_chunk_appends_without_crossing_a_power_of_two():
    """A chunk whose key count stays below the next power of two appends: the
    earlier keys keep their clusters, the chunk's keys join their nearest
    stored centroid, and each centroid absorbs its new members' sum. Covers
    no new keys (n = 1), pages split between old and new keys, and a long
    append; the result is deterministic."""
    setup = Setup(_config())
    first = [1, 65, 2050, 8200]
    setup.build(n_keys=first)
    before = [setup.kmeans_from_state(b, n) for b, n in enumerate(first)]
    saved = (setup.cid.clone(), setup.cen.clone())
    layout = setup.build(n_prev=first)
    once = (setup.cid.clone(), setup.cen.clone())
    setup.cid.copy_(saved[0])
    setup.cen.copy_(saved[1])
    setup.build(n_prev=first)
    assert torch.equal(once[0], setup.cid) and torch.equal(once[1], setup.cen)

    run_size = layout.run_size.cpu()
    for b, n in enumerate(LENGTHS):
        k, k_scale = setup.fp8_vals[b][:n], setup.scales[b][:n]
        got = setup.kmeans_from_state(b, n)
        ref = kmeans_append_reference(k, k_scale, before[b])
        assert got.n_clusters == before[b].n_clusters
        assert torch.equal(got.cluster_of_pos[: first[b]], before[b].cluster_of_pos)
        assert torch.equal(got.sizes, torch.bincount(got.cluster_of_pos, minlength=C))
        assert torch.equal(run_size[b].long(), got.sizes), b
        # New keys sit in their nearest pre-append centroid, up to near-ties.
        sim = _cosines(k[first[b] :], before[b])
        chosen = sim.gather(1, got.cluster_of_pos[first[b] :, None]).squeeze(1)
        if chosen.numel():
            gap = (sim.max(dim=1).values - chosen).max().item()
            assert gap <= 1e-4, (b, gap)
        live = slice(0, got.n_clusters)
        cos = torch.nn.functional.cosine_similarity(
            got.centroids[live], ref.centroids[live], dim=1
        )
        assert cos.min().item() > 0.9999, (b, cos.min().item())
        torch.testing.assert_close(
            got.lengths[live], ref.lengths[live], rtol=1e-3, atol=1e-3
        )


def test_final_prefill_chunk_reclusters_every_key():
    """A request's final prefill chunk always runs k-means over all its keys
    (warm, from the stored centroids), even where it would otherwise append.
    With no warm iterations that keeps the centroids and moves every key, old
    ones included, to its nearest one."""
    setup = Setup(_config(kmeans_warm_iters=0))
    first = [1, 65, 2050, 8200]
    setup.build(n_keys=first)
    before = [setup.kmeans_from_state(b, n) for b, n in enumerate(first)]
    build_ivf_index(
        key=setup.key,
        state=setup.state,
        batch_rows_cpu=list(range(len(LENGTHS))),
        n_keys_cpu=LENGTHS,
        num_batch_rows=len(LENGTHS),
        config=setup.config,
        n_prev_keys_cpu=first,
        final_cpu=[True] * len(LENGTHS),
    )
    for b, n in enumerate(LENGTHS):
        got = setup.kmeans_from_state(b, n)
        assert torch.equal(got.centroids, before[b].centroids), b
        sim = _cosines(setup.fp8_vals[b][:n], got)
        chosen = sim.gather(1, got.cluster_of_pos[:, None]).squeeze(1)
        assert (sim.max(dim=1).values - chosen).max().item() <= 1e-4, b


def test_kmeans_stops_at_a_fixed_point():
    """With iterations to spare, k-means stops once an assignment moves no key:
    every centroid is then the normalized mean of its members."""
    setup = Setup(_config(kmeans_iters=100))
    setup.build()
    for b, n in enumerate(LENGTHS):
        got = setup.kmeans_from_state(b, n)
        keys = setup.fp8_vals[b][:n] * setup.scales[b][:n, None]
        sums = torch.zeros(C, 128).index_add_(0, got.cluster_of_pos, keys)
        full = got.sizes > 0
        cos = torch.nn.functional.cosine_similarity(
            sums[full], got.centroids[full], dim=1
        )
        assert cos.min().item() > 0.999, (b, cos.min().item())


@pytest.mark.parametrize("sms", [1, 4096])
def test_kmeans_result_is_independent_of_program_count(sms, monkeypatch):
    """The fused k-means claims each phase's work from counters, so a single
    program, or far more programs than fit on the GPU at once (late ones start
    only after earlier ones exit), builds the same index without hanging."""
    n = 40000
    buf, bt, _, _ = build_paged_indexer_cache(
        [_keys(n, seed=5)], num_blocks=n // PAGE + 8, block_size=PAGE
    )
    key = IvfKeyView.of(buf.view(-1, PAGE, 132).to(DEV), bt.to(DEV))

    def build():
        cid = torch.zeros((n + CID_BS - 1) // CID_BS, CID_BS, 1, dtype=torch.int16)
        cen = torch.zeros((C + CPB - 1) // CPB, CPB, REC_BYTES, dtype=torch.uint8)
        cid, cen = cid.to(DEV), cen.to(DEV)
        state = IvfStateView.of(
            cid,
            cen,
            torch.arange(cid.shape[0], dtype=torch.int32, device=DEV)[None],
            torch.arange(cen.shape[0], dtype=torch.int32, device=DEV)[None],
            C,
        )
        build_ivf_index(
            key=key,
            state=state,
            batch_rows_cpu=[0],
            n_keys_cpu=[n],
            num_batch_rows=1,
            config=_config(),
        )
        return cid, cen

    want = build()
    monkeypatch.setattr(ivf_sparse_indexer, "_num_sms", lambda _: sms)
    got = build()
    assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])


def test_layout_runs_are_ascending_partitions():
    """Runs partition each request's keys by cluster, ascending, and carry the
    keys and scales in run order (the grouped scorer's key operand)."""
    setup = Setup(_config())
    layout = setup.build()
    run_pos = layout.run_pos.cpu()
    run_start, run_size = layout.run_start.cpu(), layout.run_size.cpu()
    run_keys, run_scales = layout.run_keys.float().cpu(), layout.run_scales.cpu()
    for b, n in enumerate(LENGTHS):
        got = setup.kmeans_from_state(b, n)
        assert torch.equal(run_size[b].long(), got.sizes)
        seen = torch.zeros(n, dtype=torch.int32)
        for c in range(C):
            start, size = int(run_start[b, c]), int(run_size[b, c])
            run = run_pos[start : start + size].long()
            assert torch.all(run[1:] > run[:-1])
            assert torch.all(got.cluster_of_pos[run] == c)
            assert torch.equal(run_keys[start : start + size], setup.fp8_vals[b][run])
            assert torch.equal(run_scales[start : start + size], setup.scales[b][run])
            seen[run] += 1
        assert torch.all(seen == 1)


def _run_prefill_select(setup, layout, rows_per_req, seed=0):
    row_batch, query_pos = [], []
    for b, n in enumerate(LENGTHS):
        picks = sorted({0, n - 1, *torch.randint(0, n, (rows_per_req,)).tolist()})
        row_batch += [b] * len(picks)
        query_pos += picks
    qb = _queries(len(row_batch), seed)
    out = torch.full((len(row_batch), TOPK + 128), -7, dtype=torch.int32, device=DEV)
    select_ivf_topk(
        key=setup.key,
        state=setup.state,
        q_fp8=qb.fp8,
        weights=qb.w,
        row_batch=torch.tensor(row_batch, dtype=torch.int32, device=DEV),
        query_pos=torch.tensor(query_pos, dtype=torch.int32, device=DEV),
        config=setup.config,
        out=out,
        topk_backend="auto",
        layout=layout,
    )
    return out.cpu(), row_batch, query_pos, qb


def test_prefill_build_and_select_never_sync_with_a_host_width_bound():
    """Given a host-known width bound (the longest prefix), a prefill chunk's
    build and select run without one host sync, as the serving op calls them,
    and select the same keys as sizing by the exact read-back width."""
    setup = Setup(_config())
    first = [1, 65, 2050, 8200]
    setup.build(n_keys=first)
    saved = (setup.cid.clone(), setup.cen.clone())
    row_batch = [b for b, n in enumerate(LENGTHS) for _ in range(min(n, 40))]
    query_pos = [t for n in LENGTHS for t in range(n - min(n, 40), n)]
    qb = _queries(len(row_batch))
    rb = torch.tensor(row_batch, dtype=torch.int32, device=DEV)
    qp = torch.tensor(query_pos, dtype=torch.int32, device=DEV)

    def step(max_width):
        out = torch.full(
            (len(row_batch), TOPK + 128), -7, dtype=torch.int32, device=DEV
        )
        layout = setup.build(n_prev=first)
        select_ivf_topk(
            key=setup.key,
            state=setup.state,
            q_fp8=qb.fp8,
            weights=qb.w,
            row_batch=rb,
            query_pos=qp,
            config=setup.config,
            out=out,
            topk_backend="auto",
            layout=layout,
            max_width=max_width,
        )
        return out

    want = step(None).cpu()
    setup.cid.copy_(saved[0])
    setup.cen.copy_(saved[1])
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        got = step(max(LENGTHS))
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert torch.equal(got.cpu().sort(dim=1).values, want.sort(dim=1).values)


def _reference(setup, b, t, qb: Queries, r, n, decode):
    km = setup.kmeans_from_state(b, n)
    k_fp8, k_scale = setup.fp8_vals[b][:n], setup.scales[b][:n]
    q, w = qb.q[r], qb.w_cpu[r]
    scores = key_scores_reference(q, w, k_fp8, k_scale)
    ref = ivf_select_one_reference(
        q_fp8=q,
        weights=w,
        k_fp8=k_fp8,
        k_scale=k_scale,
        kmeans=km,
        t=t,
        topk=TOPK,
        num_probes=setup.config.num_probes,
        decode=decode,
        scores_all=scores,
    )
    return ref, scores


@pytest.mark.parametrize("num_probes", [5, 16])
@pytest.mark.parametrize("num_clusters", [C, 20])
def test_prefill_select_matches_reference(num_probes, num_clusters, monkeypatch):
    """Fixed P probes, every visible key of them a candidate (no cap), top-k by
    the DSA score: the candidate count and the selection match the reference."""
    counts = []
    real = ivf_sparse_indexer._topk_and_emit

    def spy(**kw):
        counts.append(kw["counts"].cpu())
        real(**kw)

    monkeypatch.setattr(ivf_sparse_indexer, "_topk_and_emit", spy)
    setup = Setup(_config(num_probes=num_probes, num_clusters=num_clusters))
    layout = setup.build()
    out, row_batch, query_pos, qb = _run_prefill_select(setup, layout, 6)
    counts = torch.cat(counts)
    for r, (b, t) in enumerate(zip(row_batch, query_pos)):
        ref, scores = _reference(setup, b, t, qb, r, LENGTHS[b], decode=False)
        assert int(counts[r]) == max(ref.total, 1), (b, t)
        got = out[r][out[r] >= 0].long()
        assert torch.all(out[r, : got.numel()] >= 0), (b, t)
        assert got.numel() == ref.selected.numel(), (b, t)
        assert int(got.max()) <= t
        assert tie_aware_recall(got, ref.selected, scores) > 0.999, (b, t)
        assert torch.all(out[r, TOPK:] == -1)


@pytest.mark.parametrize(
    "tuning",
    [
        {},
        {"gemm_row_tiles": 1},
        {"gemm_mma_cols": 64},
        {"gemm_block_n": 128, "gemm_num_stages": 3},
        {"gemm_key_chunks": 2, "gemm_chunked_keys": 0, "gemm_row_tiles": 1},
        {"gemm_chunked_keys": 0},
    ],
)
def test_grouped_gemm_scores_every_probed_key(tuning, monkeypatch):
    """The prefill grouped GEMM writes the DSA score sum_h w_h ReLU(q_h . k) of
    every visible key of every probed cluster into the row's candidate slots,
    with groups split across many units or few.
    A group's rows span several row tiles and its keys several chunks, so
    units whose rows see only part of their chunk, or none of it, are
    exercised."""
    captured = {}
    real = ivf_sparse_indexer._score_grouped_gemm

    def spy(**kw):
        real(**kw)
        captured.setdefault("passes", []).append(kw)

    monkeypatch.setattr(ivf_sparse_indexer, "_score_grouped_gemm", spy)
    monkeypatch.setattr(
        ivf_sparse_indexer,
        "TUNING",
        dataclasses.replace(ivf_sparse_indexer.TUNING, **tuning),
    )
    # Half the clusters: late rows of the 9000-key request exceed top-k.
    setup = Setup(_config(num_probes=C // 2))
    layout = setup.build()
    _run_prefill_select(setup, layout, rows_per_req=40)
    assert sum(_check_grouped_scores(kw, layout) for kw in captured["passes"]) > 0


def _check_grouped_scores(captured, layout):
    q, w = captured["q_fp8"].float(), captured["weights"]
    scores = captured["scores"]
    sel, vis, ends = captured["sel"], captured["vis_ws"], captured["ends"]
    topk_lens, row_batch = captured["topk_lens"], captured["row_batch"]
    keys, key_scales = layout.run_keys.float(), layout.run_scales
    checked = 0
    for r in torch.nonzero(topk_lens > 0).flatten().tolist():
        b = int(row_batch[r])
        for c in torch.nonzero(sel[r]).flatten().tolist():
            first = int(ends[r, c] - vis[r, c])
            n = int(vis[r, c])
            k0 = int(layout.run_start[b, c])
            per_head = torch.relu(keys[k0 : k0 + n] @ q[r].T) * w[r]
            want = per_head.sum(1) * key_scales[k0 : k0 + n]
            got = scores[r, first : first + n]
            # fp8 MMA: exact products; K = 128 sums in one promotion interval.
            # Signed head weights cancel, so bound by the absolute head sum.
            bound = per_head.abs().sum(1) * key_scales[k0 : k0 + n]
            tol = 2e-3 * bound.max().item() + 1e-6
            torch.testing.assert_close(got, want, rtol=2e-3, atol=tol)
            checked += n
    return checked


def test_probing_every_cluster_is_exact():
    setup = Setup(_config(num_probes=C))
    layout = setup.build()
    out, row_batch, query_pos, qb = _run_prefill_select(setup, layout, 4)
    for r, (b, t) in enumerate(zip(row_batch, query_pos)):
        n = LENGTHS[b]
        scores = key_scores_reference(
            qb.q[r], qb.w_cpu[r], setup.fp8_vals[b][:n], setup.scales[b][:n]
        )
        exact = exact_topk_reference(scores, t, TOPK)
        got = out[r][out[r] >= 0].long()
        assert got.numel() == min(t + 1, TOPK), (b, t)
        assert torch.all(out[r, : got.numel()] >= 0), (b, t)
        assert int(got.max()) <= t and got.unique().numel() == got.numel()
        assert tie_aware_recall(got, exact, scores) > 0.999, (b, t)


def test_prefill_every_row_spanning_probe_tiles():
    """With more clusters than one probe tile, every prefill row probing every
    cluster selects distinct visible positions: all of them up to top-k, else
    exactly top-k. Guards the probe's cross-tile workspace re-read."""
    num_clusters, n = 4 * ivf_sparse_indexer.TUNING.centroid_block, 4000
    buf, bt, _, _ = build_paged_indexer_cache(
        [_keys(n, seed=0)], num_blocks=n // PAGE + 8, block_size=PAGE
    )
    key = IvfKeyView.of(buf.view(-1, PAGE, 132).to(DEV), bt.to(DEV))
    cid = torch.zeros(
        (n + CID_BS - 1) // CID_BS, CID_BS, 1, dtype=torch.int16, device=DEV
    )
    cen = torch.zeros(
        (num_clusters + CPB - 1) // CPB, CPB, REC_BYTES, dtype=torch.uint8, device=DEV
    )
    state = IvfStateView.of(
        cid,
        cen,
        torch.arange(cid.shape[0], dtype=torch.int32, device=DEV)[None],
        torch.arange(cen.shape[0], dtype=torch.int32, device=DEV)[None],
        num_clusters,
    )
    config = IvfIndexerConfig(
        num_clusters=num_clusters,
        num_probes=num_clusters,
        max_clusters=num_clusters,
        kmeans_iters=5,
        kmeans_seed=0,
        topk=TOPK,
    )
    layout = build_ivf_index(
        key=key,
        state=state,
        batch_rows_cpu=[0],
        n_keys_cpu=[n],
        num_batch_rows=1,
        config=config,
    )
    out = torch.full((n, TOPK), -7, dtype=torch.int32, device=DEV)
    qb = _queries(n)
    select_ivf_topk(
        key=key,
        state=state,
        q_fp8=qb.fp8,
        weights=qb.w,
        row_batch=torch.zeros(n, dtype=torch.int32, device=DEV),
        query_pos=torch.arange(n, dtype=torch.int32, device=DEV),
        config=config,
        out=out,
        topk_backend="auto",
        layout=layout,
    )
    out, pos = out.cpu(), torch.arange(n)
    valid = out >= 0
    assert torch.all(out[~valid] == -1)
    assert torch.all(out[valid] <= pos[:, None].expand_as(out)[valid])
    assert torch.equal(valid.sum(1), (pos + 1).clamp(max=TOPK))
    ordered = out.sort(dim=1).values
    assert not torch.any((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0))


def test_prefill_rows_from_host_lengths():
    """Each prefill row's batch row, last complete pool and visible tokens
    follow from its request's context and query lengths."""
    kpool, first_row = 4, 2
    seq_lens, query_lens = [5, 9, 3], [2, 4, 3]
    row_batch, pool_pos, token_lens = _prefill_rows(
        seq_lens, query_lens, first_row, kpool, DEV
    )
    want_rows, want_tokens = [], []
    for i, (n, q) in enumerate(zip(seq_lens, query_lens)):
        want_rows += [first_row + i] * q
        want_tokens += list(range(n - q + 1, n + 1))
    assert row_batch.tolist() == want_rows
    assert token_lens.tolist() == want_tokens
    assert pool_pos.tolist() == [t // kpool - 1 for t in want_tokens]


def test_prefill_emit_expands_pools_like_the_kpool_indexer():
    """With token lengths, the prefill select writes the kpool indexer's
    expanded row (each pool's tokens, the incomplete tail pool, then -1) for
    the pools it selects, straight into a wider top-k buffer."""
    kpool, width = 4, TOPK * 4 + 128
    setup = Setup(_config())
    layout = setup.build()
    row_batch, query_pos = [], []
    for b, n in enumerate(LENGTHS):
        picks = sorted({0, n - 1, *torch.randint(0, n, (40,)).tolist()})
        row_batch += [b] * len(picks)
        query_pos += picks
    rows = len(row_batch)
    qb = _queries(rows, seed=3)
    pos = torch.tensor(query_pos, dtype=torch.int32, device=DEV)
    # A row's last complete pool is its key position; 0..3 tail tokens follow.
    token_lens = (pos + 1) * kpool + torch.randint(
        0, kpool, (rows,), dtype=torch.int32, device=DEV
    )
    outs = []
    for out, kw in (
        (torch.full((rows, TOPK), -7, dtype=torch.int32, device=DEV), {}),
        (
            torch.full((rows, width), -7, dtype=torch.int32, device=DEV),
            {"token_lens": token_lens},
        ),
    ):
        select_ivf_topk(
            key=setup.key,
            state=setup.state,
            q_fp8=qb.fp8,
            weights=qb.w,
            row_batch=torch.tensor(row_batch, dtype=torch.int32, device=DEV),
            query_pos=pos,
            config=dataclasses.replace(setup.config, kpool=kpool),
            out=out,
            topk_backend="auto",
            layout=layout,
            **kw,
        )
        outs.append(out)
    pools, fused = outs
    want = torch.full((rows, width), -1, dtype=torch.int32, device=DEV)
    expanded = expand_pools_and_append_tail(pools, token_lens, kpool)
    want[:, : expanded.shape[1]] = expanded
    # Top-k lists a row's pools in any order: compare each row's tokens.
    assert torch.equal(fused.sort(dim=1).values, want.sort(dim=1).values)


def test_prefill_scratch_bound_drops_lowest_scoring_probes():
    """A scratch bound that fits every row's candidates changes nothing; one
    below them drops a row's lowest-scoring probed clusters, so it still
    selects distinct visible keys from at least its best probe."""
    setup = Setup(_config())
    layout = setup.build()
    rows = 64
    qb = _queries(rows, seed=5)
    b, n = 3, LENGTHS[3]
    pos = torch.arange(n - rows, n, dtype=torch.int32, device=DEV)
    outs = []
    for max_scratch in (None, rows * 4 * (n + TOPK), 1):
        out = torch.full((rows, TOPK), -7, dtype=torch.int32, device=DEV)
        select_ivf_topk(
            key=setup.key,
            state=setup.state,
            q_fp8=qb.fp8,
            weights=qb.w,
            row_batch=torch.full((rows,), b, dtype=torch.int32, device=DEV),
            query_pos=pos,
            config=setup.config,
            out=out,
            topk_backend="auto",
            layout=layout,
            max_scratch_bytes=max_scratch,
            max_width=n,
        )
        outs.append(out.cpu())
    uncapped, generous, capped = outs
    assert torch.equal(uncapped.sort(dim=1).values, generous.sort(dim=1).values)
    valid = capped >= 0
    assert torch.all(capped[~valid] == -1) and torch.all(valid.any(dim=1))
    assert torch.all(capped[valid] <= pos.cpu()[:, None].expand_as(capped)[valid])
    ordered = capped.sort(dim=1).values
    assert not torch.any((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0))
    # The bound (top-k slots) is below these rows' candidates: some dropped.
    assert not torch.equal(ordered, uncapped.sort(dim=1).values)


def _decode_step(setup, seq_lens, qb: Queries):
    append_ivf_decode_keys(key=setup.key, state=setup.state, seq_lens=seq_lens)
    rows = seq_lens.shape[0]
    out = torch.full((rows, TOPK + 128), -7, dtype=torch.int32, device=DEV)
    select_ivf_topk(
        key=setup.key,
        state=setup.state,
        q_fp8=qb.fp8,
        weights=qb.w,
        row_batch=torch.arange(rows, dtype=torch.int32, device=DEV),
        query_pos=seq_lens - 1,
        config=setup.config,
        out=out,
        topk_backend="auto",
        seq_lens=seq_lens,
    )
    return out


def test_decode_matches_reference():
    setup = Setup(_config(), extra_decode_key=True)
    setup.build()
    before = [setup.kmeans_from_state(b, n) for b, n in enumerate(LENGTHS)]
    seq_lens = torch.tensor([n + 1 for n in LENGTHS], dtype=torch.int32, device=DEV)
    qb = _queries(len(LENGTHS), seed=7)
    out = _decode_step(setup, seq_lens, qb).cpu()
    for b, n in enumerate(LENGTHS):
        want = decode_assign_reference(
            setup.fp8_vals[b][n],
            before[b].centroids,
            before[b].n_clusters,
        )
        km = setup.kmeans_from_state(b, n + 1)
        assert int(km.cluster_of_pos[n]) == want
        assert int(km.sizes[want]) == int(before[b].sizes[want]) + 1
        ref, scores = _reference(setup, b, n, qb, b, n + 1, decode=True)
        got = out[b][out[b] >= 0].long()
        assert got.numel() == ref.selected.numel()
        assert tie_aware_recall(got, ref.selected, scores) > 0.999


def test_decode_first_key_seeds_one_cluster():
    """A one-token prompt decodes against garbage state: it must get one
    cluster holding its own key, and select exactly position 0."""
    setup = Setup(_config())
    seq_lens = torch.ones(len(LENGTHS), dtype=torch.int32, device=DEV)
    out = _decode_step(setup, seq_lens, _queries(len(LENGTHS), seed=3)).cpu()
    for b in range(len(LENGTHS)):
        km = setup.kmeans_from_state(b, 1)
        assert km.n_clusters == 1
        assert int(km.cluster_of_pos[0]) == 0 and int(km.sizes[0]) == 1
        assert out[b][out[b] >= 0].tolist() == [0]


def test_decode_cuda_graph_matches_eager():
    setup = Setup(_config(), extra_decode_key=True)
    setup.build()
    snapshot = (setup.cid.clone(), setup.cen.clone())
    seq_lens = torch.tensor([n + 1 for n in LENGTHS], dtype=torch.int32, device=DEV)
    qb = _queries(len(LENGTHS), seed=7)

    def restore():
        setup.cid.copy_(snapshot[0])
        setup.cen.copy_(snapshot[1])

    eager = _decode_step(setup, seq_lens, qb)
    for _ in range(2):  # compile outside capture
        restore()
        _decode_step(setup, seq_lens, qb)
    restore()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _decode_step(setup, seq_lens, qb)
    restore()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured.sort(dim=1).values, eager.sort(dim=1).values)


KPOOL = 4
# GLM-5.3's indexer heads, which the dense (DeepGEMM) prefill path supports.
POOL_HEADS = 32
POOL_PAGE = 32  # the pooled cache's kernel page at block_size 1152
POOL_TOPK = TOPK // KPOOL
# Requests with more pools than top-k and fewer, plus ones whose last row sees
# exactly POOL_TOPK and POOL_TOPK + 1 pools; all end mid-pool so the tail is
# live and the first decode steps complete a pool.
POOLED_LENGTHS = [
    4 * 600 + 2,
    4 * 257 + 2,
    KPOOL * POOL_TOPK + 2,
    KPOOL * (POOL_TOPK + 1) + 2,
]


class PooledSetup:
    """Caches and metadata for ``sparse_attn_indexer_ivf`` on pooled keys."""

    def __init__(self, num_probes: int, exact_pools: int = 0):
        self.config = dataclasses.replace(
            _config(num_probes=num_probes),
            topk=POOL_TOPK,
            kpool=KPOOL,
            exact_pools=exact_pools,
        )
        g = torch.Generator().manual_seed(11)
        rows = len(POOLED_LENGTHS)
        cap = max(POOLED_LENGTHS) + 8
        self.k = [_keys(cap, seed=20 + b).to(torch.bfloat16) for b in range(rows)]
        self.gate = [
            torch.randn(cap, 128, generator=g).to(torch.bfloat16) for _ in range(rows)
        ]
        self.ape = (torch.randn(KPOOL, 128, generator=g) * 0.1).to(DEV)
        pages = (cap // KPOOL + POOL_PAGE - 1) // POOL_PAGE
        self.key_cache = torch.zeros(
            rows * pages + 3, POOL_PAGE, 132, dtype=torch.uint8, device=DEV
        )
        self.bt = _block_table(rows, pages, self.key_cache.shape[0], 4).to(DEV)
        ring = KPOOL
        self.tail = torch.zeros(rows, 2, ring, 128, dtype=torch.bfloat16, device=DEV)
        cid_blocks = MAX_LEN // CID_BS
        cen_blocks = (C + CPB - 1) // CPB
        self.cid = torch.randint(
            0, C, (rows * cid_blocks + 3, CID_BS, 1), generator=g, dtype=torch.int16
        ).to(DEV)
        self.cen = torch.randint(
            0, 256, (rows * cen_blocks + 3, CPB, REC_BYTES), generator=g
        ).to(device=DEV, dtype=torch.uint8)
        self.cid_bt = _block_table(rows, cid_blocks, self.cid.shape[0], 5).to(DEV)
        self.cen_bt = _block_table(rows, cen_blocks, self.cen.shape[0], 6).to(DEV)

    def pool_slot(self, b: int, pos: int) -> int:
        """The kpool slot written by the token at ``pos`` (-1 mid-pool)."""
        if (pos + 1) % KPOOL:
            return -1
        p = pos // KPOOL
        return int(self.bt[b, p // POOL_PAGE]) * POOL_PAGE + p % POOL_PAGE

    def step(self, monkeypatch, starts, ends, qb):
        """One indexer step: request b runs tokens [starts[b], ends[b])."""
        rows = len(starts)
        qlens = [e - s for s, e in zip(starts, ends)]
        tokens = [(b, pos) for b in range(rows) for pos in range(starts[b], ends[b])]
        decode = all(q == 1 for q in qlens) and min(starts) > 0
        k = torch.stack([self.k[b][pos] for b, pos in tokens]).to(DEV)
        gate = torch.stack([self.gate[b][pos] for b, pos in tokens]).to(DEV)
        positions = torch.tensor([pos for _, pos in tokens], device=DEV)
        slot = torch.tensor([self.pool_slot(b, pos) for b, pos in tokens], device=DEV)
        tail_slot = torch.tensor(
            [b * KPOOL + pos % KPOOL for b, pos in tokens], device=DEV
        )
        qsl = torch.tensor([0, *torch.tensor(qlens).cumsum(0).tolist()])
        seq_lens = list(ends)
        meta = IvfIndexerMetadata(
            slot_mapping=slot,
            block_table=self.bt,
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=DEV),
            query_start_loc=qsl.to(device=DEV, dtype=torch.int32),
            num_reqs=rows,
            num_decodes=rows if decode else 0,
            num_decode_tokens=len(tokens) if decode else 0,
            num_prefills=0 if decode else rows,
            num_prefill_tokens=0 if decode else len(tokens),
            req_index=torch.arange(rows, dtype=torch.int32, device=DEV),
            prefill_seq_lens_cpu=[] if decode else seq_lens,
            prefill_query_lens_cpu=[] if decode else qlens,
            prefill_final_cpu=[] if decode else [True] * rows,
        )
        attn_metadata = {
            "k": meta,
            "tail": SimpleNamespace(slot_mapping=tail_slot),
            "cid": IvfStateMetadata(block_table=self.cid_bt),
            "cen": IvfStateMetadata(block_table=self.cen_bt),
        }
        monkeypatch.setattr(
            ivf_sparse_indexer,
            "get_forward_context",
            lambda: SimpleNamespace(attn_metadata=attn_metadata),
        )
        out = torch.full((len(tokens), TOPK + 128), -7, dtype=torch.int32, device=DEV)
        sparse_attn_indexer_ivf(
            k,
            "k",
            "tail",
            "cid",
            "cen",
            self.key_cache,
            self.tail,
            self.cid,
            self.cen,
            qb.fp8,
            qb.w,
            k,
            gate,
            self.ape,
            positions,
            self.config,
            out,
            "auto",
            1 << 30,
        )
        return out.cpu(), tokens

    def pooled_keys(self, b: int, num_pools: int):
        """Request b's first pools as stock compresses them (fp8 as fp32, scale)."""
        n = num_pools * KPOOL
        slot_k = self.k[b][:n].view(num_pools, KPOOL, 128).to(DEV)
        slot_score = self.gate[b][:n].view(num_pools, KPOOL, 128).to(DEV)
        loc = torch.zeros(num_pools, dtype=torch.int64, device=DEV)
        fp8, scale = kpool_compress_and_write_cache(
            self.key_cache,
            slot_k,
            slot_score,
            self.ape,
            loc,
            KPOOL,
            return_compressed=True,
            write_cache=False,
        )
        return fp8.float().cpu(), scale.cpu()

    def cached_pool(self, b: int, p: int):
        """Pool p of request b as read back through the IVF key view."""
        page = int(self.bt[b, p // POOL_PAGE])
        flat = self.key_cache[page].view(-1).cpu()
        off = p % POOL_PAGE
        fp8 = flat[off * 128 : (off + 1) * 128].view(torch.float8_e4m3fn).float()
        scale = flat[POOL_PAGE * 128 :].view(torch.float32)[off]
        return fp8, scale


def _check_row(got, t, scores, recall=True):
    """A row selects its top pools (expanded) plus its tail, nothing else."""
    num_pools = (t + 1) // KPOOL
    tail = set(range(KPOOL * num_pools, t + 1))
    assert len(got) == len(set(got)) and max(got) <= t
    assert tail <= set(got)
    pools = {pos // KPOOL for pos in got if pos < KPOOL * num_pools}
    # Selected pools come whole.
    assert sum(pos < KPOOL * num_pools for pos in got) == KPOOL * len(pools)
    if num_pools <= POOL_TOPK:
        assert pools == set(range(num_pools))
    else:
        # Probing continues until the candidates hold top-k pools.
        assert len(pools) == POOL_TOPK
        if recall:
            exact = exact_topk_reference(scores, num_pools - 1, POOL_TOPK)
            selected = torch.tensor(sorted(pools))
            assert tie_aware_recall(selected, exact, scores) > 0.999


def test_page_table_counts_cache_pages():
    """The group's block table counts kernel blocks (64 tokens), smaller than
    the pooled cache's pages (128 tokens = 32 pools); IVF reads and writes
    through the page table, as the kpool indexer does."""
    pages = torch.tensor([[7, 2, 5], [0, 9, 4]], dtype=torch.int32, device=DEV)
    kernel_blocks = torch.stack([2 * pages, 2 * pages + 1], dim=2).flatten(1)
    builder = SimpleNamespace(
        kv_cache_spec=SimpleNamespace(block_size=128),
        kernel_block_size=64,
        page_table_buffer=None,
        max_num_seqs=4,
    )
    cam = SimpleNamespace(block_table_tensor=kernel_blocks)
    got = IvfIndexerMetadataBuilder._page_table(builder, cam)
    assert torch.equal(got, pages)
    # A persistent buffer: decode CUDA graphs keep its address.
    again = IvfIndexerMetadataBuilder._page_table(builder, cam)
    assert again.data_ptr() == got.data_ptr()


@pytest.mark.parametrize(("num_probes", "exact_pools"), [(C, 0), (4, 0), (4, 512)])
def test_pooled_prefill_and_decode(monkeypatch, num_probes, exact_pools):
    """IVF on the kpool indexer's pools: stock-written pools are what the IVF
    key view reads; probing every cluster gives the exact top pools; rows
    seeing at most top-k pools keep them all; every row keeps its tail; decode
    appends a pool exactly when it completes. A prefill request with at most
    ``exact_pools`` pools scores them densely (here the middle two of a batch
    whose outer two probe), so it selects the exact top pools."""
    setup = PooledSetup(num_probes, exact_pools)
    rows = len(POOLED_LENGTHS)
    total = sum(POOLED_LENGTHS)
    qb = _queries(total, seed=5, heads=POOL_HEADS)
    out, tokens = setup.step(monkeypatch, [0] * rows, POOLED_LENGTHS, qb)

    for b, n in enumerate(POOLED_LENGTHS):
        fp8, scale = setup.pooled_keys(b, n // KPOOL)
        for p in (0, n // KPOOL - 1):
            got_fp8, got_scale = setup.cached_pool(b, p)
            assert torch.equal(got_fp8, fp8[p]) and float(got_scale) == float(scale[p])
    keys = [setup.pooled_keys(b, n // KPOOL) for b, n in enumerate(POOLED_LENGTHS)]
    for r, (b, t) in enumerate(tokens):
        if t % 97 and t != POOLED_LENGTHS[b] - 1:
            continue
        got = out[r][out[r] >= 0].tolist()
        scores = key_scores_reference(qb.q[r], qb.w_cpu[r], *keys[b])
        exact = POOLED_LENGTHS[b] // KPOOL <= exact_pools
        _check_row(got, t, scores, recall=num_probes == C or exact)
    assert int(out.min()) >= -1

    def live_size(b):
        recs = torch.stack(
            [setup.cen.cpu()[setup.cen_bt[b, c // CPB], c % CPB] for c in range(C)]
        )
        size = recs[:, 256:260].contiguous().view(torch.int32).flatten()
        return int(size.clamp(min=0).sum())

    # Two decode tokens: the first leaves the pool open, the second completes it.
    lens = list(POOLED_LENGTHS)
    for completes in (False, True):
        before = [live_size(b) for b in range(rows)]
        dq = _queries(rows, seed=6 + completes, heads=POOL_HEADS)
        out, _ = setup.step(monkeypatch, lens, [n + 1 for n in lens], dq)
        lens = [n + 1 for n in lens]
        for b, n in enumerate(lens):
            assert live_size(b) == before[b] + completes == n // KPOOL
            num_pools = n // KPOOL
            cached = [setup.cached_pool(b, p) for p in range(num_pools)]
            fp8, scale = setup.pooled_keys(b, num_pools)
            assert torch.equal(torch.stack([c[0] for c in cached]), fp8)
            assert torch.equal(torch.stack([c[1] for c in cached]), scale)
            scores = key_scores_reference(dq.q[b], dq.w_cpu[b], fp8, scale)
            got = out[b][out[b] >= 0].tolist()
            _check_row(got, n - 1, scores, recall=num_probes == C)
        assert int(out.min()) >= -1
