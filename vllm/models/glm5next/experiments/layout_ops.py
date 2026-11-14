# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment 3 kernels: the v0.32 grouped scoring with keys read in place.

Copies of three v0.32 pieces (``nvidia/ops/ivf_index.py`` and
``_score_grouped_gemm`` in ``nvidia/ivf_sparse_indexer.py``), changed only
where the layout differs, so a timing difference is the key layout alone:

* ``grouped_score_inplace_kernel``: ``ivf_grouped_score_kernel`` with the
  key-chunk load swapped. v0.32 TMA-loads a cluster's keys from the
  cluster-ordered copy ``run_keys``; this one looks each key's position up in
  ``run_pos`` and loads it (and its scale) from the paged KV cache. Grouping,
  work units, row tiles and the fp8 product are unchanged.
* ``run_scatter_positions_kernel``: ``ivf_run_scatter_kernel`` without the key
  and scale copy: the counting sort's positions only, which is all the
  in-place scoring needs.
* ``read_keys_kernel``: a pure read of every key, contiguous from ``run_keys``,
  gathered in cluster order from the cache, or in token order from the cache.

Nothing here is imported by the served model.
"""

from __future__ import annotations

import torch

from vllm.models.glm5next.nvidia.ops.ivf_index import (
    _key_page,
    _load_key_scales,
    _load_keys,
    _score_key_chunk,
)
from vllm.triton_utils import tl, triton


@triton.jit
def _load_key_chunk_inplace(
    k_u8_ptr,
    k_f32_ptr,
    k_page_stride,
    bt_row_ptr,
    run_pos_ptr,
    key0,
    m0,
    vis_max,
    BLOCK_N: tl.constexpr,
    PAGE: tl.constexpr,
):
    """A cluster's keys m0.. and scales, from the paged cache (0 past it)."""
    offs_m = m0 + tl.arange(0, BLOCK_N)
    mask = offs_m < vis_max
    pos = tl.load(run_pos_ptr + key0 + offs_m, mask=mask, other=0)
    page = tl.load(bt_row_ptr + pos // PAGE, mask=mask, other=0).to(tl.int64)
    k = _load_keys(k_u8_ptr, k_page_stride, page, pos, PAGE, mask)
    s = _load_key_scales(k_f32_ptr, k_page_stride, page, pos, PAGE, mask)
    return k, s


@triton.jit(do_not_specialize=["cand_cap", "num_groups", "log_g"])
def grouped_score_inplace_kernel(
    q_fp8_ptr,
    w_ptr,
    k_u8_ptr,
    k_f32_ptr,
    k_page_stride,
    k_bt_ptr,
    k_bt_stride,
    run_pos_ptr,
    run_start_ptr,
    group_m_ptr,
    group_vis_ptr,
    group_off_ptr,
    unit_off_ptr,
    row_list_ptr,
    row_vis_ptr,
    row_first_ptr,
    score_ptr,
    cand_cap,
    num_groups,
    log_g,
    C: tl.constexpr,
    PAGE: tl.constexpr,
    H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    KEY_CHUNKS: tl.constexpr,
    ROWS: tl.constexpr,
    ROW_TILES: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """``ivf_grouped_score_kernel`` with in-place key loads (see module doc).

    Group g is (batch row g // C, cluster g % C); the batch row picks the
    block-table row the keys are looked up in.
    """
    UNIT_ROWS: tl.constexpr = ROWS * ROW_TILES
    UNIT_KEYS: tl.constexpr = BLOCK_N * KEY_CHUNKS
    offs_d = tl.arange(0, 128)
    total = tl.load(unit_off_ptr + num_groups)
    u = tl.atomic_add(unit_off_ptr + num_groups + 1, 1, sem="relaxed")
    while u < total:
        lo = tl.full([], 0, tl.int32)
        hi = tl.full([], 0, tl.int32) + num_groups
        for _ in range(log_g):
            active = lo < hi
            mid = (lo + hi) // 2
            right = tl.load(unit_off_ptr + mid + 1, mask=active, other=0) <= u
            lo = tl.where(active & right, mid + 1, lo)
            hi = tl.where(active & ~right, mid, hi)
        g = lo
        rel = u - tl.load(unit_off_ptr + g)
        vis_max = tl.load(group_vis_ptr + g)
        n_chunks = tl.cdiv(vis_max, UNIT_KEYS)
        m0 = (rel % n_chunks) * UNIT_KEYS
        r_begin = (rel // n_chunks) * UNIT_ROWS
        r_end = tl.minimum(tl.load(group_m_ptr + g), r_begin + UNIT_ROWS)
        row_base = tl.load(group_off_ptr + g)
        if m0 < tl.load(row_vis_ptr + row_base + r_end - 1):
            key0 = tl.load(run_start_ptr + g)
            bt_row = k_bt_ptr + (g // C).to(tl.int64) * k_bt_stride
            k0, s0 = _load_key_chunk_inplace(
                k_u8_ptr, k_f32_ptr, k_page_stride, bt_row, run_pos_ptr, key0,
                m0, vis_max, BLOCK_N, PAGE)
            if KEY_CHUNKS > 1:
                k1, s1 = _load_key_chunk_inplace(
                    k_u8_ptr, k_f32_ptr, k_page_stride, bt_row, run_pos_ptr,
                    key0, m0 + BLOCK_N, vis_max, BLOCK_N, PAGE)
            if KEY_CHUNKS > 2:
                k2, s2 = _load_key_chunk_inplace(
                    k_u8_ptr, k_f32_ptr, k_page_stride, bt_row, run_pos_ptr,
                    key0, m0 + 2 * BLOCK_N, vis_max, BLOCK_N, PAGE)
                k3, s3 = _load_key_chunk_inplace(
                    k_u8_ptr, k_f32_ptr, k_page_stride, bt_row, run_pos_ptr,
                    key0, m0 + 3 * BLOCK_N, vis_max, BLOCK_N, PAGE)
            for r0 in tl.range(r_begin, r_end, ROWS, num_stages=NUM_STAGES):
                offs_r = r0 + tl.arange(0, ROWS)
                mask_r = offs_r < r_end
                slots = row_base + offs_r
                rows = tl.load(row_list_ptr + slots, mask=mask_r, other=0)
                vis = tl.load(row_vis_ptr + slots, mask=mask_r, other=0)
                first = tl.load(row_first_ptr + slots, mask=mask_r, other=0)
                heads = tl.reshape(
                    rows.to(tl.int64)[None, :] * H + tl.arange(0, H)[:, None],
                    (ROWS * H,),
                )
                q = tl.load(q_fp8_ptr + heads[:, None] * 128 + offs_d[None, :])
                w = tl.load(w_ptr + heads)
                _score_key_chunk(k0, s0, q, w, m0, rows, mask_r, vis, first,
                                 score_ptr, cand_cap, BLOCK_N, H, ROWS)
                if KEY_CHUNKS > 1:
                    seen = tl.max(vis, axis=0)
                    if m0 + BLOCK_N < seen:
                        _score_key_chunk(k1, s1, q, w, m0 + BLOCK_N, rows, mask_r,
                                         vis, first, score_ptr, cand_cap, BLOCK_N,
                                         H, ROWS)
                    if KEY_CHUNKS > 2:
                        if m0 + 2 * BLOCK_N < seen:
                            _score_key_chunk(k2, s2, q, w, m0 + 2 * BLOCK_N, rows,
                                             mask_r, vis, first, score_ptr,
                                             cand_cap, BLOCK_N, H, ROWS)
                        if m0 + 3 * BLOCK_N < seen:
                            _score_key_chunk(k3, s3, q, w, m0 + 3 * BLOCK_N, rows,
                                             mask_r, vis, first, score_ptr,
                                             cand_cap, BLOCK_N, H, ROWS)
        u = tl.atomic_add(unit_off_ptr + num_groups + 1, 1, sem="relaxed")


def make_inplace_scorer(get_key):
    """A drop-in for ``ivf._score_grouped_gemm`` that scores in place.

    ``get_key()`` returns the ``IvfKeyView`` the layout was built from (the
    v0.32 function does not take it, since it reads ``layout.run_keys``).
    The planning launches are v0.32's, copied unchanged.
    """
    from vllm.models.glm5next.nvidia import ivf_sparse_indexer as ivf

    def score(*, q_fp8, weights, row_batch, layout, sel, vis_ws, ends, topk_lens,
              scores, num_clusters, groups):
        key = get_key()
        tune = ivf.TUNING
        num_rows = q_fp8.shape[0]
        C = num_clusters
        device = q_fp8.device
        num_groups = groups.shape[1]
        group_m, group_vis = groups
        block_r = (tune.plan_short_block_r if C >= tune.plan_short_rows_clusters
                   else tune.plan_block_r)
        row_tiles = triton.cdiv(num_rows, block_r)
        (group_off, unit_off, row_list, row_vis, row_first, tile_cnt, tile_base,
         cluster_base) = ivf._alloc(
            torch.int32, device, (num_groups,), (num_groups + 2,),
            (num_rows * C,), (num_rows * C,), (num_rows * C,), (row_tiles, C),
            (row_tiles, C), (C,))
        plan_grid = (row_tiles, triton.cdiv(C, tune.plan_block_c))
        ivf.ivf_group_count_kernel[plan_grid](
            sel, vis_ws, topk_lens, row_batch, group_m, group_vis, tile_cnt,
            num_rows, C=C, BLOCK_R=block_r, BLOCK_C=tune.plan_block_c,
            PDL=tune.pdl, launch_pdl=tune.pdl)
        rows = max(1, tune.gemm_mma_cols // q_fp8.shape[1])
        chunked = layout.run_pos.shape[0] >= tune.gemm_chunked_keys * C
        key_chunks = tune.gemm_key_chunks if chunked else 1
        ivf.ivf_group_plan_kernel[(triton.cdiv(C, tune.plan_scan_clusters) + 1,)](
            tile_cnt, tile_base, group_m, group_vis, cluster_base, unit_off,
            row_tiles, num_groups, C=C, C_PAD=triton.next_power_of_2(C),
            UNIT_KEYS=tune.gemm_block_n * key_chunks,
            UNIT_ROWS=rows * tune.gemm_row_tiles, BLOCK_T=tune.plan_scan_tiles,
            BLOCK=1024, BLOCK_C=tune.plan_scan_clusters, PDL=tune.pdl,
            launch_pdl=tune.pdl)
        ivf.ivf_group_fill_kernel[plan_grid](
            sel, vis_ws, ends, topk_lens, row_batch, tile_base, cluster_base,
            row_list, row_vis, row_first, group_off, num_rows, C=C,
            BLOCK_R=block_r, BLOCK_C=tune.plan_block_c, PDL=tune.pdl,
            launch_pdl=tune.pdl)
        per_sm = (tune.gemm_chunked_programs_per_sm if chunked
                  else tune.gemm_programs_per_sm)
        num_programs = per_sm * ivf._num_sms(device.index or 0)
        grouped_score_inplace_kernel[(num_programs,)](
            q_fp8, weights, key.u8, key.f32, key.page_stride, key.block_table,
            key.block_table.stride(0), layout.run_pos, layout.run_start, group_m,
            group_vis, group_off, unit_off, row_list, row_vis, row_first, scores,
            scores.shape[1], num_groups, num_groups.bit_length(), C=C,
            PAGE=key.page, H=q_fp8.shape[1], BLOCK_N=tune.gemm_block_n,
            KEY_CHUNKS=key_chunks, ROWS=rows, ROW_TILES=tune.gemm_row_tiles,
            NUM_STAGES=tune.gemm_num_stages, num_warps=tune.gemm_num_warps)

    return score


@triton.jit
def run_scatter_positions_kernel(
    k_u8_ptr,
    k_f32_ptr,
    k_page_stride,
    k_bt_ptr,
    k_bt_stride,
    tile_item_ptr,
    tile_start_ptr,
    req_ptr,
    n_keys_ptr,
    cu_keys_ptr,
    packed_cid_ptr,
    tile_off_ptr,
    run_start_ptr,
    run_pos_ptr,
    run_keys_ptr,
    run_scales_ptr,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
    PAGE: tl.constexpr,
    PDL: tl.constexpr = False,
):
    """``ivf_run_scatter_kernel`` minus the key and scale copy (same args)."""
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    tile = tl.program_id(0)
    i = tl.load(tile_item_ptr + tile)
    start = tl.load(tile_start_ptr + tile)
    b = tl.load(req_ptr + i)
    n_keys = tl.load(n_keys_ptr + i)
    cu = tl.load(cu_keys_ptr + i)
    pos = start + tl.arange(0, PAGE)
    mask_t = pos < n_keys
    cl = tl.load(packed_cid_ptr + cu + pos, mask=mask_t, other=-1)
    offs_k = tl.arange(0, PAGE)
    earlier = (cl[:, None] == cl[None, :]) & (offs_k[None, :] < offs_k[:, None])
    rank = tl.sum(earlier.to(tl.int32), axis=1)
    c_safe = tl.where(mask_t, cl, 0)
    base = tl.load(run_start_ptr + b * C + c_safe, mask=mask_t, other=0)
    base += tl.load(tile_off_ptr + tile * C + c_safe, mask=mask_t, other=0)
    tl.store(run_pos_ptr + base + rank, pos, mask=mask_t)


@triton.jit
def read_keys_kernel(
    src_ptr,
    run_pos_ptr,
    bt_ptr,
    page_stride,
    out_ptr,
    n,
    MODE: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Read keys [BLOCK, 128] bytes: MODE 0 slot i of ``run_keys``; 1 the
    cache key at ``run_pos[i]`` (cluster order); 2 cache key i (token order).
    A per-program byte sum is stored so the reads are kept."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    offs_d = tl.arange(0, 128)
    if MODE == 0:
        base = offs.to(tl.int64) * 128
    else:
        if MODE == 1:
            pos = tl.load(run_pos_ptr + offs, mask=mask, other=0)
        else:
            pos = offs
        page = _key_page(bt_ptr, 0, 0, pos, PAGE, mask)
        base = page * page_stride + (pos % PAGE) * 128
    x = tl.load(src_ptr + base[:, None] + offs_d[None, :], mask=mask[:, None],
                other=0)
    tl.store(out_ptr + pid, tl.sum(tl.sum(x.to(tl.int32), axis=1), axis=0))


def read_keys(mode: int, src: torch.Tensor, run_pos: torch.Tensor,
              block_table_row: torch.Tensor, page_stride: int, page: int, n: int,
              out: torch.Tensor, block: int = 64) -> None:
    read_keys_kernel[(triton.cdiv(n, block),)](
        src, run_pos, block_table_row, page_stride, out, n, MODE=mode, PAGE=page,
        BLOCK=block)
