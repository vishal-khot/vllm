# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-torch reference of the glm5next IVF (k-means) DSA indexer.

The algorithm of ``ivf-dsa-kmeans/compare_decode_kmeans.py`` on the indexer's
data, mirroring the Triton kernels' arithmetic one request and one query at a
time. It is the oracle for the kernel tests and runs on CPU.

- Keys are the cached, FWHT-rotated indexer keys (one head: MQA): fp8 values
  and one ue8m0 scale per key. k-means clusters them by cosine.
- A query is the stock indexer's input: H rotated fp8 heads q_h (values as
  fp32) and head weights w_h with the query scales folded in.
- Probing scores cluster c by Σ_h w_h ReLU(cos(q_h, c)) and takes a fixed
  number (P) of clusters.
- Every visible key of the probed clusters is a candidate (no cap); the top-k
  candidates by the DSA score Σ_h w_h ReLU(q_h · k) · scale_k are selected.
"""

from typing import NamedTuple

import torch

FP8_MAX = 448.0
_SUM_SCALE = float(1 << 24)
HEAD_DIM = 128


def cluster_count(num_keys: int, num_clusters: int, max_clusters: int) -> int:
    """Clusters of a request with ``num_keys`` keys.

    Args:
        num_keys: Keys being clustered.
        num_clusters: The configured count.
        max_clusters: Capacity of the centroid state.

    """
    return max(1, min(num_clusters, num_keys, max_clusters))


def key_scores_reference(
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
) -> torch.Tensor:
    """DSA scores Σ_h w_h ReLU(q_h · k) · scale_k of fp8 keys [N, 128].

    Args:
        q_fp8: [H, 128] query heads (fp8 values).
        weights: [H] head weights.
        k_fp8: [N, 128] key fp8 values.
        k_scale: [N] key scales.

    """
    per_head = torch.relu(k_fp8.float() @ q_fp8.float().T)
    return (per_head * weights.float()).sum(1) * k_scale.float()


def unit_hash_reference(seed: int, idx: torch.Tensor) -> torch.Tensor:
    mask = 0xFFFFFFFF
    h = (idx.to(torch.int64) + seed * 2654435769) & mask
    h = (h * 2246822519) & mask
    h = h ^ (h >> 15)
    h = (h * 3266489917) & mask
    h = h ^ (h >> 13)
    return (h >> 8).to(torch.float32) / 16777216.0


def quantize_rows_fp8(
    x: torch.Tensor, round_pow2: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row fp8 e4m3.

    Args:
        x: [..., D] values.
        round_pow2: Round scales up to a power of two (ue8m0, as the indexer
            key cache does). Otherwise use amax / 448.

    Returns:
        The fp8 values as fp32 and the fp32 per-row scale.

    """
    amax = x.abs().amax(dim=-1)
    if round_pow2:
        scale = torch.exp2(torch.ceil(torch.log2(amax.clamp(min=1e-4) / FP8_MAX)))
    else:
        scale = amax.clamp(min=1e-12) / FP8_MAX
    q = (x / scale[..., None]).clamp(-FP8_MAX, FP8_MAX)
    q = q.to(torch.float8_e4m3fn).to(torch.float32)
    if not round_pow2:
        q = torch.where(amax[..., None] > 0, q, torch.zeros_like(q))
        scale = torch.where(amax > 0, scale, torch.zeros_like(scale))
    return q, scale


class KmeansResult(NamedTuple):
    # [C, 128] stored unit centroids: bf16 values as fp32 (zero if disabled).
    centroids: torch.Tensor
    # [T] cluster of every position and [C] cluster sizes, both int64.
    cluster_of_pos: torch.Tensor
    sizes: torch.Tensor
    n_clusters: int
    # Centroid updates run before the assignments settled (<= iters).
    updates: int = 0


def kmeans_reference(
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    *,
    max_clusters: int,
    num_clusters: int,
    iters: int,
    seed: int,
    init_centroids: torch.Tensor | None = None,
    stop_fraction: float = 0.0,
) -> KmeansResult:
    """Cosine k-means over one request's keys (fp8 values [T, 128], scales [T]).

    Args:
        k_fp8: Key fp8 values as fp32.
        k_scale: Per-key scales.
        max_clusters: Centroid-state capacity C (the result's array size).
        num_clusters: Configured count (see ``cluster_count``).
        iters: Most assign + update rounds before the final assignment.
        seed: Initialization seed.
        init_centroids: [C, 128] centroids to start from (the previous chunk's,
            same cluster count) instead of seeding.
        stop_fraction: The rounds stop once an assignment moves at most this
            fraction of the keys (0: once it repeats the previous one).

    """
    k_fp8 = k_fp8.float()
    num_keys = k_fp8.shape[0]
    C = max_clusters
    c_eff = cluster_count(num_keys, num_clusters, max_clusters)
    fixed = (k_fp8 * (k_scale.float() * _SUM_SCALE)[:, None]).to(torch.int64)

    c = torch.arange(C)
    jitter = (
        (unit_hash_reference(seed, c) * num_keys)
        .to(torch.int64)
        .clamp(max=num_keys - 1)
    )
    pos = (c * num_keys + jitter) // max(c_eff, 1)
    live = c < c_eff
    init = k_fp8[pos.clamp(max=num_keys - 1)]
    init = init / init.norm(dim=1, keepdim=True).clamp(min=1e-12)
    if init_centroids is not None:
        init = init_centroids.float()
    c_f32 = torch.where(live[:, None], init, torch.zeros_like(init))
    c_bf16 = c_f32.to(torch.bfloat16).float()

    def assign() -> torch.Tensor:
        # The key norm cannot change the argmax (cosine assignment).
        sim = k_fp8 @ c_bf16.T
        sim[:, c_eff:] = float("-inf")
        return sim.argmax(dim=1)

    updates, prev = 0, None
    for _ in range(iters):
        cid = assign()
        moved = None if prev is None else int((cid != prev).sum())
        if moved is not None and moved <= int(stop_fraction * num_keys):
            break
        prev = cid
        updates += 1
        counts = torch.bincount(cid, minlength=C)
        sums = torch.zeros((C, HEAD_DIM), dtype=torch.int64).index_add_(0, cid, fixed)
        mean = sums.to(torch.float32)
        norm = mean.norm(dim=1)
        unit = mean / norm.clamp(min=1e-30)[:, None]
        keep_old = (counts == 0) | (norm == 0)
        c_f32 = torch.where(keep_old[:, None], c_f32, unit)
        c_f32 = torch.where(live[:, None], c_f32, torch.zeros_like(c_f32))
        c_bf16 = c_f32.to(torch.bfloat16).float()
    cid = assign()
    return KmeansResult(
        centroids=c_bf16,
        cluster_of_pos=cid,
        sizes=torch.bincount(cid, minlength=C),
        n_clusters=c_eff,
        updates=updates,
    )


def decode_assign_reference(
    key: torch.Tensor, centroids: torch.Tensor, n_clusters: int
) -> int:
    sim = centroids @ key.float()
    sim[n_clusters:] = float("-inf")
    return int(sim.argmax())


def probe_scores_reference(
    q_fp8: torch.Tensor, weights: torch.Tensor, centroids: torch.Tensor
) -> torch.Tensor:
    """Cluster scores Σ_h w_h ReLU(cos(q_h, c)) of unit ``centroids`` [C, 128]."""
    q = q_fp8.float()
    q_unit = q / q.norm(dim=1, keepdim=True).clamp(min=1e-12)
    return (torch.relu(centroids.float() @ q_unit.T) * weights.float()).sum(1)


def probe_reference(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    kmeans: KmeansResult,
    visible: torch.Tensor,
    num_probes: int,
) -> torch.Tensor:
    """Bool [C] of probed clusters; ``visible`` [C] is each cluster's visible size.

    Scores the live centroids by ``probe_scores_reference`` and probes the best
    ``num_probes`` of those with a visible key, ties to the lower cluster id.
    """
    C = kmeans.sizes.shape[0]
    score = probe_scores_reference(q_fp8, weights, kmeans.centroids)
    eligible = (torch.arange(C) < kmeans.n_clusters) & (visible > 0)
    ids = torch.nonzero(eligible).flatten()
    order = torch.argsort(score[ids], descending=True, stable=True)
    probed = torch.zeros(C, dtype=torch.bool)
    probed[ids[order[:num_probes]]] = True
    return probed


def collect_reference(
    *,
    probed: torch.Tensor,
    cluster_of_pos: torch.Tensor,
    t: int,
    cluster_major: bool,
) -> torch.Tensor:
    """Candidate positions: every visible member of the probed clusters.

    Prefill lays runs out cluster by cluster; decode scans positions in order.
    A query with no candidates keeps its own position.
    """
    visible_cid = cluster_of_pos[: t + 1]
    pos = torch.nonzero(probed[visible_cid]).flatten()
    if cluster_major:
        order = torch.argsort(visible_cid[pos], stable=True)
        pos = pos[order]
    return pos if pos.numel() else torch.tensor([t])


def select_reference(
    *, cand: torch.Tensor, scores: torch.Tensor, topk: int
) -> torch.Tensor:
    if cand.numel() <= topk:
        return cand
    return cand[torch.topk(scores, topk).indices]


def exact_topk_reference(scores_all: torch.Tensor, t: int, topk: int) -> torch.Tensor:
    """Dense top-k over positions [0, t]: stock DSA at per-token resolution."""
    visible = scores_all[: t + 1]
    return torch.topk(visible, min(topk, visible.numel())).indices


def tie_aware_recall(
    selected: torch.Tensor, exact: torch.Tensor, scores_all: torch.Tensor
) -> float:
    """Share of the exact top-k matched; a key tied with the k-th counts as a hit."""
    if exact.numel() == 0:
        return 1.0
    kth = scores_all[exact].min()
    hits = int((scores_all[selected] >= kth).sum())
    return min(hits, exact.numel()) / exact.numel()


class SelectionTrace(NamedTuple):
    selected: torch.Tensor
    candidates: torch.Tensor
    total: int
    probed: torch.Tensor


def ivf_select_one_reference(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    kmeans: KmeansResult,
    t: int,
    topk: int,
    num_probes: int,
    decode: bool,
    scores_all: torch.Tensor | None = None,
) -> SelectionTrace:
    """Full selection for one query (heads ``q_fp8``, ``weights``) at position ``t``."""
    cid = kmeans.cluster_of_pos
    C = kmeans.sizes.shape[0]
    visible = kmeans.sizes if decode else torch.bincount(cid[: t + 1], minlength=C)
    probed = probe_reference(
        q_fp8=q_fp8,
        weights=weights,
        kmeans=kmeans,
        visible=visible,
        num_probes=num_probes,
    )
    total = int(visible[probed].sum())
    cand = collect_reference(
        probed=probed, cluster_of_pos=cid, t=t, cluster_major=not decode
    )
    if scores_all is None:
        scores = key_scores_reference(q_fp8, weights, k_fp8[cand], k_scale[cand])
    else:
        scores = scores_all[cand]
    return SelectionTrace(
        selected=select_reference(cand=cand, scores=scores, topk=topk),
        candidates=cand,
        total=total,
        probed=probed,
    )


def build_paged_indexer_cache(
    keys_per_request: list[torch.Tensor],
    *,
    num_blocks: int,
    block_size: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    """Quantize keys as the per-token indexer cache does; scatter into random blocks.

    Each block is ``[block_size x 128 fp8][block_size x fp32 scale]``, the
    layout written by ``indexer_k_quant_and_cache``.

    Returns:
        The cache ``[num_blocks, block_size * 132]`` uint8, the block table
        ``[B, max_blocks]`` int32, and per request the fp8 values [T, 128] as
        fp32 and the scales [T].

    """
    entry = HEAD_DIM + 4
    buf = torch.zeros((num_blocks, block_size * entry), dtype=torch.uint8)
    max_blocks = max(
        (k.shape[0] + block_size - 1) // block_size for k in keys_per_request
    )
    block_table = torch.zeros((len(keys_per_request), max_blocks), dtype=torch.int32)
    perm = torch.randperm(num_blocks, generator=generator)
    used = 0
    fp8_vals, scales = [], []
    scale_off = block_size * HEAD_DIM
    for b, keys in enumerate(keys_per_request):
        q, s = quantize_rows_fp8(keys.float(), round_pow2=True)
        fp8_vals.append(q)
        scales.append(s)
        n_blocks = (keys.shape[0] + block_size - 1) // block_size
        blocks = perm[used : used + n_blocks]
        used += n_blocks
        block_table[b, :n_blocks] = blocks.to(torch.int32)
        raw = q.to(torch.float8_e4m3fn).view(torch.uint8)
        for i in range(n_blocks):
            rows = slice(i * block_size, min((i + 1) * block_size, keys.shape[0]))
            n = rows.stop - rows.start
            blk = int(blocks[i])
            buf[blk, : n * HEAD_DIM] = raw[rows].reshape(-1)
            buf[blk, scale_off : scale_off + 4 * n] = (
                s[rows].contiguous().view(torch.uint8)
            )
    return buf, block_table, fp8_vals, scales
