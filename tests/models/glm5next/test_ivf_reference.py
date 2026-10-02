# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the IVF indexer reference, the oracle for the kernel tests."""

import pytest
import torch

from vllm.models.glm5next.common.ivf_reference import (
    build_paged_indexer_cache,
    cluster_count,
    collect_reference,
    exact_topk_reference,
    ivf_select_one_reference,
    key_scores_reference,
    kmeans_reference,
    probe_scores_reference,
    quantize_rows_fp8,
    tie_aware_recall,
)

TOPK = 64
C_MAX = 64
HEADS = 16


def _keys(num_keys: int, seed: int = 0, centers: int = 8) -> torch.Tensor:
    """Keys drawn around a few directions, like content-clustered indexer keys."""
    g = torch.Generator().manual_seed(seed)
    mu = torch.randn(centers, 128, generator=g) * 3
    which = torch.randint(0, centers, (num_keys,), generator=g)
    return mu[which] + torch.randn(num_keys, 128, generator=g)


def _query(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """fp8 query heads [H, 128] (values) and signed head weights [H]."""
    g = torch.Generator().manual_seed(1000 + seed)
    q, scale = quantize_rows_fp8(torch.randn(HEADS, 128, generator=g), True)
    return q, torch.randn(HEADS, generator=g) * scale


def _kmeans(k_fp8, k_scale, num_clusters=32, iters=5, seed=0, stop_fraction=0.0):
    return kmeans_reference(
        k_fp8,
        k_scale,
        max_clusters=C_MAX,
        num_clusters=num_clusters,
        iters=iters,
        seed=seed,
        stop_fraction=stop_fraction,
    )


def _select(k_fp8, k_scale, km, t, *, num_probes, decode, seed=0):
    q_fp8, weights = _query(seed)
    return ivf_select_one_reference(
        q_fp8=q_fp8,
        weights=weights,
        k_fp8=k_fp8,
        k_scale=k_scale,
        kmeans=km,
        t=t,
        topk=TOPK,
        num_probes=num_probes,
        decode=decode,
    )


def test_cluster_count_is_fixed_within_keys_and_capacity():
    assert [cluster_count(n, 16, 1000) for n in (1, 5, 16, 5000)] == [1, 5, 16, 16]
    assert cluster_count(10**6, 256, 203) == 203


@pytest.mark.parametrize("num_keys", [1, 5, 32, 777, 3000])
@pytest.mark.parametrize("num_clusters", [32, 7])
def test_kmeans_partitions_every_position_once(num_keys, num_clusters):
    k_fp8, k_scale = quantize_rows_fp8(_keys(num_keys), True)
    result = _kmeans(k_fp8, k_scale, num_clusters=num_clusters)
    assert result.n_clusters == cluster_count(num_keys, num_clusters, C_MAX)
    assert result.cluster_of_pos.shape == (num_keys,)
    assert int(result.cluster_of_pos.max()) < result.n_clusters
    assert int(result.sizes.sum()) == num_keys
    assert torch.equal(
        result.sizes, torch.bincount(result.cluster_of_pos, minlength=C_MAX)
    )
    assert torch.isfinite(result.centroids).all()


def test_kmeans_is_deterministic():
    k_fp8, k_scale = quantize_rows_fp8(_keys(2000), True)
    a = _kmeans(k_fp8, k_scale)
    b = _kmeans(k_fp8, k_scale)
    assert torch.equal(a.cluster_of_pos, b.cluster_of_pos)
    assert torch.equal(a.centroids, b.centroids)
    c = _kmeans(k_fp8, k_scale, seed=1)
    assert not torch.equal(a.cluster_of_pos, c.cluster_of_pos)


@pytest.mark.parametrize("stop_fraction", [0.0, 0.01])
def test_kmeans_stops_once_assignments_settle(stop_fraction):
    k_fp8, k_scale = quantize_rows_fp8(_keys(3000), True)
    stopped = _kmeans(k_fp8, k_scale, iters=100, stop_fraction=stop_fraction)
    assert 0 < stopped.updates < 100
    # Running exactly that many updates gives the same clustering.
    exact = _kmeans(k_fp8, k_scale, iters=stopped.updates)
    assert torch.equal(stopped.cluster_of_pos, exact.cluster_of_pos)
    assert torch.equal(stopped.centroids, exact.centroids)


def test_kmeans_resumes_from_previous_chunk_centroids():
    k_fp8, k_scale = quantize_rows_fp8(_keys(3000), True)
    first = _kmeans(k_fp8[:2000], k_scale[:2000])
    resumed = kmeans_reference(
        k_fp8,
        k_scale,
        max_clusters=C_MAX,
        num_clusters=32,
        iters=0,
        seed=0,
        init_centroids=first.centroids,
    )
    assert torch.equal(resumed.centroids, first.centroids)


def test_empty_cluster_keeps_its_centroid():
    # Identical keys tie everywhere: argmax picks cluster 0, the rest stay empty.
    k_fp8, k_scale = quantize_rows_fp8(torch.ones(256, 128), True)
    init = _kmeans(k_fp8, k_scale, num_clusters=4, iters=0)
    after = _kmeans(k_fp8, k_scale, num_clusters=4, iters=3)
    assert int(after.sizes[0]) == 256
    assert torch.equal(after.centroids[1:4], init.centroids[1:4])


@pytest.mark.parametrize("decode", [False, True])
def test_probing_every_cluster_is_exact_topk(decode):
    """With P = C, the selection is the exact top-k by the DSA score over visible
    keys (``run_brute_force_topk`` of the reference file, causally)."""
    num_keys = 1500
    k_fp8, k_scale = quantize_rows_fp8(_keys(num_keys), True)
    km = _kmeans(k_fp8, k_scale)
    for seed, t in enumerate([10, 200, 999, num_keys - 1]):
        if decode:
            t = num_keys - 1
        scores = key_scores_reference(*_query(seed), k_fp8, k_scale)
        trace = _select(k_fp8, k_scale, km, t, num_probes=32, decode=decode, seed=seed)
        exact = exact_topk_reference(scores, t, TOPK)
        assert tie_aware_recall(trace.selected, exact, scores) == 1.0


def test_fixed_probes_take_every_visible_key_of_them():
    num_keys = 3000
    k_fp8, k_scale = quantize_rows_fp8(_keys(num_keys), True)
    km = _kmeans(k_fp8, k_scale, num_clusters=C_MAX)
    want_probes = 16
    for seed, t in enumerate([0, 40, 63, 500, 2999]):
        trace = _select(
            k_fp8, k_scale, km, t, num_probes=want_probes, decode=False, seed=seed
        )
        visible = torch.bincount(km.cluster_of_pos[: t + 1], minlength=C_MAX)
        assert int(trace.probed.sum()) == min(want_probes, int((visible > 0).sum()))
        cand = trace.candidates
        # Uncapped: every visible key of every probed cluster, nothing else.
        assert cand.numel() == trace.total == int(visible[trace.probed].sum())
        assert int(cand.max()) <= t
        assert cand.unique().numel() == cand.numel()
        assert torch.all(trace.probed[km.cluster_of_pos[cand]])


def test_probe_picks_the_best_centroids():
    k_fp8, k_scale = quantize_rows_fp8(_keys(2000), True)
    km = _kmeans(k_fp8, k_scale)
    q_fp8, weights = _query(3)
    trace = _select(k_fp8, k_scale, km, 1999, num_probes=5, decode=True, seed=3)
    score = probe_scores_reference(q_fp8, weights, km.centroids)
    # Only live clusters with a visible (here: any) key can be probed.
    score[km.n_clusters :] = float("-inf")
    score[km.sizes == 0] = float("-inf")
    assert torch.equal(
        torch.nonzero(trace.probed).flatten(), torch.topk(score, 5).indices.sort()[0]
    )


def test_degenerate_probe_keeps_own_position():
    # Cluster 0 holds only late positions, so query 0 probing it sees nothing.
    cid = torch.tensor([1, 1, 0, 0])
    probed = torch.tensor([True, False])
    cand = collect_reference(probed=probed, cluster_of_pos=cid, t=0, cluster_major=True)
    assert cand.tolist() == [0]


@pytest.mark.parametrize("block_size", [64, 128])
def test_paged_cache_round_trip(block_size):
    lens = (1, block_size, 2 * block_size + 2)
    keys = [_keys(n, seed=n) for n in lens]
    buf, block_table, fp8_vals, scales = build_paged_indexer_cache(
        keys, num_blocks=16, block_size=block_size
    )
    scale_off = block_size * 128
    for b, n in enumerate(lens):
        for pos in (0, n - 1):
            blk = int(block_table[b, pos // block_size])
            off = pos % block_size
            raw = buf[blk, off * 128 : (off + 1) * 128]
            assert torch.equal(raw.view(torch.float8_e4m3fn).float(), fp8_vals[b][pos])
            s = buf[blk, scale_off + 4 * off : scale_off + 4 * (off + 1)]
            assert s.view(torch.float32).item() == scales[b][pos].item()
