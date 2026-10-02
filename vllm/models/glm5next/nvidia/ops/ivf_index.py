# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for the glm5next IVF (k-means) DSA indexer.

Keys are read from the per-token indexer cache. A kernel page holds ``PAGE``
tokens as ``[PAGE, 128]`` fp8 bytes followed by ``PAGE`` fp32 ue8m0 scales,
the layout written by ``ivf_write_keys_kernel`` (and ``indexer_k_quant_and_cache``).

Per-request IVF state lives in two KV cache groups addressed by block table:
  cluster ids: int16 per position, ``cid_bs`` positions per block.
  centroids:   ``cpb`` records per block; a 272-byte record is the unit
               centroid in bf16 (256 bytes), the int32 cluster size (-1 marks
               a disabled cluster) and padding to 16-byte alignment.

Pipeline per DSA layer:
  key write:  FWHT + fp8 quant + scatter for every token of the step
  build:      one persistent kernel: seed -> iters x (assign + accumulate,
              update) -> final assign
  decode:     append (assign the new key to its nearest fixed centroid)
  select:     probe (score, select) -> [decode: collect by scan] -> score
              -> radix top-k -> emit (prefill maps slots through the runs)
"""

from vllm.triton_utils import tl, triton

HEAD_DIM = 128
REC_BYTES = 272  # 128 bf16 + int32 size + padding
FP8_MAX = 448.0


@triton.jit
def _unit_hash(seed, idx):
    """Deterministic uniform in [0, 1); mirrored by ``unit_hash_reference``."""
    h = (idx.to(tl.int64) + seed.to(tl.int64) * 2654435769) & 0xFFFFFFFF
    h = (h * 2246822519) & 0xFFFFFFFF
    h = h ^ (h >> 15)
    h = (h * 3266489917) & 0xFFFFFFFF
    h = h ^ (h >> 13)
    return (h >> 8).to(tl.float32) * (1.0 / 16777216.0)


@triton.jit
def _float_order_key(x):
    """Map fp32 to int64 so that integer order equals float order."""
    bits = x.to(tl.int32, bitcast=True)
    return tl.where(bits >= 0, bits, bits ^ 0x7FFFFFFF).to(tl.int64)


@triton.jit
def _kth_largest(sorted_desc, k, SIZE: tl.constexpr):
    """Element k-1 of a descending-sorted vector (k >= 1)."""
    return tl.sum(tl.where(tl.arange(0, SIZE) == k - 1, sorted_desc, 0))


@triton.jit
def _cluster_count(n_keys, num_clusters, C: tl.constexpr):
    """Clusters of a request: num_clusters within [1, min(n_keys, C)]. Mirrors
    ``cluster_count``."""
    return tl.maximum(tl.minimum(tl.minimum(num_clusters, n_keys), C), 1)


@triton.jit
def _rec(cen_bt_ptr, cen_bt_stride, b, c, cpb, cen_stride, mask):
    """Byte offset of centroid record ``c`` of batch row ``b``."""
    blk = tl.load(cen_bt_ptr + b * cen_bt_stride + c // cpb, mask=mask, other=0)
    return blk.to(tl.int64) * cen_stride + (c % cpb).to(tl.int64) * 272


@triton.jit
def _key_page(k_bt_ptr, k_bt_stride, b, pos, PAGE: tl.constexpr, mask):
    page = tl.load(k_bt_ptr + b * k_bt_stride + pos // PAGE, mask=mask, other=0)
    return page.to(tl.int64)


@triton.jit
def _load_keys(k_u8_ptr, k_page_stride, page, pos, PAGE: tl.constexpr, mask):
    """fp8 keys ``[N, 128]`` at ``pos`` in ``page``."""
    offs_d = tl.arange(0, 128)
    base = page * k_page_stride + (pos % PAGE) * 128
    k = tl.load(
        k_u8_ptr + base[:, None] + offs_d[None, :],
        mask=mask[:, None],
        other=0,
    )
    return k.to(tl.float8e4nv, bitcast=True)


@triton.jit
def _load_key_scales(k_f32_ptr, k_page_stride, page, pos, PAGE: tl.constexpr, mask):
    return tl.load(
        k_f32_ptr + page * (k_page_stride // 4) + PAGE * 32 + pos % PAGE,
        mask=mask,
        other=0.0,
    )


@triton.jit
def _load_centroids(cen_ptr, rec, mask, CACHE: tl.constexpr = ""):
    """bf16 centroids ``[N, 128]`` of the records at byte offsets ``rec``."""
    offs_d = tl.arange(0, 128)
    return tl.load(
        cen_ptr + (rec // 2)[:, None] + offs_d[None, :],
        mask=mask[:, None],
        other=0.0,
        cache_modifier=CACHE,
    )


@triton.jit
def _store_centroid(x, c_f32_ptr, f32_row, cen_ptr, rec, mask_c):
    """Store unit rows ``x [BLOCK, 128]`` as the fp32 master and bf16 records."""
    offs_d = tl.arange(0, 128)
    master = f32_row.to(tl.int64)[:, None] * 128 + offs_d[None, :]
    tl.store(c_f32_ptr + master, x, mask=mask_c[:, None])
    tl.store(
        cen_ptr + (rec // 2)[:, None] + offs_d[None, :],
        x.to(tl.bfloat16),
        mask=mask_c[:, None],
    )


@triton.jit
def _fwht_stage(x, N: tl.constexpr, GROUPS: tl.constexpr, STRIDE: tl.constexpr):
    x3 = tl.reshape(x, (GROUPS, 2, STRIDE))
    x3 = tl.trans(x3, 0, 2, 1)
    a, b = tl.split(x3)
    x3 = tl.join(a + b, a - b)
    x3 = tl.trans(x3, 0, 2, 1)
    return tl.reshape(x3, (N,))


@triton.jit
def ivf_write_keys_kernel(
    k_ptr,
    slot_mapping_ptr,
    k_u8_ptr,
    k_f32_ptr,
    k_page_stride,
    n_rows,
    PAGE: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """Hadamard-128 rotate, ue8m0 fp8 quantize and scatter keys by slot.

    The rotation matches ``fwht128_quant_fp8`` on the query so that q . k holds
    in the rotated basis. Slots < 0 are skipped.
    """
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    rmask = rows < n_rows
    offs = tl.arange(0, 128)
    x = tl.load(
        k_ptr + rows[:, None] * 128 + offs[None, :], mask=rmask[:, None], other=0.0
    ).to(tl.float32)
    N: tl.constexpr = BLOCK_R * 128
    x = tl.reshape(x, (N,))
    x = _fwht_stage(x, N, BLOCK_R * 64, 1)
    x = _fwht_stage(x, N, BLOCK_R * 32, 2)
    x = _fwht_stage(x, N, BLOCK_R * 16, 4)
    x = _fwht_stage(x, N, BLOCK_R * 8, 8)
    x = _fwht_stage(x, N, BLOCK_R * 4, 16)
    x = _fwht_stage(x, N, BLOCK_R * 2, 32)
    x = _fwht_stage(x, N, BLOCK_R, 64)
    x = x * 0.08838834764831845
    x = x.to(tl.bfloat16).to(tl.float32)
    x = tl.reshape(x, (BLOCK_R, 128))

    absmax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-4)
    scale = tl.exp2(tl.ceil(tl.log2(absmax * (1.0 / 448.0))))
    y = tl.minimum(tl.maximum(x / scale[:, None], -448.0), 448.0)

    slot = tl.load(slot_mapping_ptr + rows, mask=rmask, other=-1).to(tl.int64)
    ok = rmask & (slot >= 0)
    page = slot // PAGE
    off = slot % PAGE
    tl.store(
        k_u8_ptr + (page * k_page_stride + off * 128)[:, None] + offs[None, :],
        y.to(tl.float8e4nv).to(tl.uint8, bitcast=True),
        mask=ok[:, None],
    )
    tl.store(k_f32_ptr + page * (k_page_stride // 4) + PAGE * 32 + off, scale, mask=ok)


@triton.jit
def _phase_done(sync_ptr):
    """Count one finished work unit of a phase, after the unit's writes."""
    tl.debug_barrier()
    tl.atomic_add(sync_ptr, 1, sem="release")


@triton.jit
def _phase_wait(sync_ptr, target):
    """Wait until ``target`` work units of a phase are done."""
    done = tl.load(sync_ptr, volatile=True)
    while done < target:
        done = tl.load(sync_ptr, volatile=True)
    tl.atomic_add(sync_ptr, 0, sem="acquire")


@triton.jit
def _nearest_centroid(kb, cf_lo, cf_hi, live_lo, live_hi, HALF: tl.constexpr):
    """Id of each key's most similar centroid, ties to the lower id.

    Scores the centroid halves one at a time to halve the live similarity
    tile. The key norm is a positive per-key factor, so it cannot change the
    argmax and is never computed.
    """
    s = tl.dot(kb, tl.trans(cf_lo))
    s = tl.where(live_lo[None, :], s, float("-inf"))
    v_lo, i_lo = tl.max(
        s, axis=1, return_indices=True, return_indices_tie_break_left=True
    )
    s = tl.dot(kb, tl.trans(cf_hi))
    s = tl.where(live_hi[None, :], s, float("-inf"))
    v_hi, i_hi = tl.max(
        s, axis=1, return_indices=True, return_indices_tie_break_left=True
    )
    return tl.where(v_hi > v_lo, i_hi + HALF, i_lo).to(tl.int32)


@triton.jit
def _segment_tiles(n_tiles, SEGS: tl.constexpr, MIN_SEG_TILES: tl.constexpr):
    """Key pages per k-means segment of a request (``kmeans_segments``)."""
    return tl.maximum(tl.cdiv(n_tiles, SEGS), MIN_SEG_TILES)


@triton.jit(do_not_specialize=["seed", "iters", "max_moved"])
def ivf_kmeans_kernel(
    k_u8_ptr,
    k_f32_ptr,
    k_page_stride,
    k_bt_ptr,
    k_bt_stride,
    cen_ptr,
    cen_i32_ptr,
    cen_stride,
    cen_bt_ptr,
    cen_bt_stride,
    cpb,
    cid_ptr,
    cid_stride,
    cid_bt_ptr,
    cid_bt_stride,
    cid_bs,
    req_ptr,
    n_keys_ptr,
    n_prev_ptr,
    tile_first_ptr,
    seg_first_ptr,
    cu_keys_ptr,
    c_f32_ptr,
    part_ptr,
    packed_cid_ptr,
    tile_counts_ptr,
    sync_ptr,
    num_items,
    num_clusters,
    seed,
    iters,
    max_moved,
    C: tl.constexpr,
    C_PAD: tl.constexpr,
    BLOCK_U: tl.constexpr,
    BLOCK_S: tl.constexpr,
    SEGS: tl.constexpr,
    MIN_SEG_TILES: tl.constexpr,
    PAGE: tl.constexpr,
):
    """The whole cosine k-means of every listed request in one launch.

    Phases: seed, up to iters x (assign + partial sums, update), final assign.
    The iterations stop early once an assign moves at most ``max_moved`` keys
    of the batch from their previous clusters (0: once the next update would
    reproduce the same centroids).
    Programs claim each phase's work units from a counter and wait for all of
    them to be done before the next phase, so progress never depends on how
    many programs are resident.

    An assign unit is one segment of a request's key pages
    (``_segment_tiles``). It loads the request's bf16 centroids once, keeps
    them on chip, and sums its keys per cluster in registers (one-hot MMA,
    fp32); the segment's sums are stored once, with no atomics. An update unit
    adds a cluster block's segment sums in segment order. Segments depend only
    on the key count, so the result is independent of the program count.

    Seeding reuses the request's centroids from its previous prefill chunk
    (``n_prev`` keys clustered, same cluster count C_r); otherwise it draws
    centroid c from the c-th of C_r position strata. Clusters past C_r are
    disabled (size -1). An update sets each centroid to the
    normalized sum of its members; an empty cluster keeps its centroid. The
    final assign writes every position's cluster id, the cluster sizes and
    each page's per-cluster key counts (``tile_counts [tiles, C]``) for the
    run layout.

    ``sync`` holds per phase a done counter, a unit claim counter, a
    moved-key count and one segment claim counter per request
    (``num_items + 3`` int32, zeroed by the caller). ``part`` is
    ``[segments, C, 128]`` fp32.
    """
    NB: tl.constexpr = (C + BLOCK_U - 1) // BLOCK_U
    HALF: tl.constexpr = C_PAD // 2
    offs_d = tl.arange(0, 128)
    offs_t = tl.arange(0, PAGE)
    offs_cp = tl.arange(0, C_PAD)
    offs_h = tl.arange(0, HALF)
    num_segs = tl.load(seg_first_ptr + num_items)
    num_units = num_items * NB
    stride = num_items + 3
    last = 2 * iters + 1
    p = last * 0
    while p <= last:
        sync = sync_ptr + p * stride
        if p % 2 == 1:
            final = p == last
            for i in range(0, num_items):
                b = tl.load(req_ptr + i)
                n_keys = tl.load(n_keys_ptr + i)
                c_eff = _cluster_count(n_keys, num_clusters, C)
                t0 = tl.load(tile_first_ptr + i)
                t_end = tl.load(tile_first_ptr + i + 1)
                seg_tiles = _segment_tiles(t_end - t0, SEGS, MIN_SEG_TILES)
                s0 = tl.load(seg_first_ptr + i)
                n_seg = tl.load(seg_first_ptr + i + 1) - s0
                cu = tl.load(cu_keys_ptr + i)
                s = tl.atomic_add(sync + 3 + i, 1, sem="relaxed")
                if s < n_seg:
                    live_lo = offs_h < c_eff
                    live_hi = HALF + offs_h < c_eff
                    rec_lo = _rec(
                        cen_bt_ptr, cen_bt_stride, b, offs_h, cpb, cen_stride, live_lo
                    )
                    rec_hi = _rec(
                        cen_bt_ptr,
                        cen_bt_stride,
                        b,
                        HALF + offs_h,
                        cpb,
                        cen_stride,
                        live_hi,
                    )
                    cf_lo = _load_centroids(cen_ptr, rec_lo, live_lo, ".cg")
                    cf_hi = _load_centroids(cen_ptr, rec_hi, live_hi, ".cg")
                    live_c = offs_cp < c_eff
                    rec_all = _rec(
                        cen_bt_ptr, cen_bt_stride, b, offs_cp, cpb, cen_stride, live_c
                    )
                    while s < n_seg:
                        acc = tl.zeros([C_PAD, 128], tl.float32)
                        ta = t0 + s * seg_tiles
                        tb = tl.minimum(ta + seg_tiles, t_end)
                        for t in range(ta, tb):
                            start = (t - t0) * PAGE
                            pos = start + offs_t
                            mask_t = pos < n_keys
                            page = tl.load(k_bt_ptr + b * k_bt_stride + start // PAGE)
                            page = page.to(tl.int64)
                            kb = _load_keys(
                                k_u8_ptr, k_page_stride, page, pos, PAGE, mask_t
                            ).to(tl.bfloat16)
                            best_c = _nearest_centroid(
                                kb, cf_lo, cf_hi, live_lo, live_hi, HALF
                            )
                            hit = (best_c[:, None] == offs_cp[None, :]) & mask_t[
                                :, None
                            ]
                            if final:
                                blk_count = tl.sum(hit.to(tl.int32), axis=0)
                                tl.atomic_add(
                                    cen_i32_ptr + (rec_all + 256) // 4,
                                    blk_count,
                                    mask=live_c & (blk_count > 0),
                                    sem="relaxed",
                                )
                                tl.store(
                                    tile_counts_ptr + t * C + offs_cp,
                                    blk_count,
                                    mask=offs_cp < C,
                                )
                                cblk = tl.load(
                                    cid_bt_ptr + b * cid_bt_stride + pos // cid_bs,
                                    mask=mask_t,
                                    other=0,
                                ).to(tl.int64)
                                tl.store(
                                    cid_ptr + cblk * cid_stride + pos % cid_bs,
                                    best_c.to(tl.int16),
                                    mask=mask_t,
                                )
                                tl.store(packed_cid_ptr + cu + pos, best_c, mask=mask_t)
                            else:
                                # packed_cid holds the previous assign's ids (none
                                # before the first assign, which never stops).
                                prev = tl.load(
                                    packed_cid_ptr + cu + pos,
                                    mask=mask_t,
                                    other=0,
                                    cache_modifier=".cg",
                                )
                                tl.store(packed_cid_ptr + cu + pos, best_c, mask=mask_t)
                                moved = tl.sum(((best_c != prev) & mask_t).to(tl.int32))
                                if moved > 0:
                                    tl.atomic_add(sync + 2, moved, sem="relaxed")
                                # ue8m0 scales are powers of two: the bf16
                                # one-hot weights and every product are exact.
                                k_scale = _load_key_scales(
                                    k_f32_ptr, k_page_stride, page, pos, PAGE, mask_t
                                )
                                onehot = tl.where(hit, k_scale[:, None], 0.0)
                                acc = tl.dot(tl.trans(onehot.to(tl.bfloat16)), kb, acc)
                        if not final:
                            row = ((s0 + s) * C + offs_cp).to(tl.int64)
                            tl.store(
                                part_ptr + row[:, None] * 128 + offs_d[None, :],
                                acc,
                                mask=(offs_cp < C)[:, None],
                            )
                        _phase_done(sync)
                        s = tl.atomic_add(sync + 3 + i, 1, sem="relaxed")
            target = num_segs
        else:
            u = tl.atomic_add(sync + 1, 1, sem="relaxed")
            # Branch-local names differ from the assign branch's: Triton merges
            # names bound in both branches of an if.
            while u < num_units:
                u_item = u // NB
                u_c = (u % NB) * BLOCK_U + tl.arange(0, BLOCK_U)
                u_b = tl.load(req_ptr + u_item)
                u_n = tl.load(n_keys_ptr + u_item)
                u_eff = _cluster_count(u_n, num_clusters, C)
                u_mask = u_c < C
                u_live = u_c < u_eff
                u_row = (u_item * C + u_c).to(tl.int64)
                u_off = u_row[:, None] * 128 + offs_d[None, :]
                u_rec = _rec(
                    cen_bt_ptr, cen_bt_stride, u_b, u_c, cpb, cen_stride, u_mask
                )
                if p == 0:
                    u_prev = tl.load(n_prev_ptr + u_item)
                    warm = (u_prev > 0) & (
                        _cluster_count(u_prev, num_clusters, C) == u_eff
                    )
                    if warm:
                        new = _load_centroids(cen_ptr, u_rec, u_live).to(tl.float32)
                    else:
                        jitter = tl.minimum(
                            (_unit_hash(seed, u_c) * u_n).to(tl.int64), u_n - 1
                        )
                        seed_pos = (u_c.to(tl.int64) * u_n + jitter) // u_eff
                        seed_pos = tl.where(u_live, seed_pos, 0)
                        seed_page = _key_page(
                            k_bt_ptr, k_bt_stride, u_b, seed_pos, PAGE, u_live
                        )
                        new = _load_keys(
                            k_u8_ptr, k_page_stride, seed_page, seed_pos, PAGE, u_live
                        ).to(tl.float32)
                        norm = tl.sqrt(tl.sum(new * new, axis=1))
                        new = new / tl.maximum(norm, 1e-12)[:, None]
                    tl.store(
                        cen_i32_ptr + (u_rec + 256) // 4,
                        tl.where(u_live, 0, -1),
                        mask=u_mask,
                    )
                else:
                    # Centroid = normalized sum of its members; an empty
                    # cluster (zero sum) keeps its centroid.
                    seg_lo = tl.load(seg_first_ptr + u_item)
                    seg_hi = tl.load(seg_first_ptr + u_item + 1)
                    total = tl.zeros([BLOCK_U, 128], tl.float32)
                    # BLOCK_S segments per load: independent loads in flight
                    # rather than one L2 round trip per segment.
                    for g0 in range(seg_lo, seg_hi, BLOCK_S):
                        g = g0 + tl.arange(0, BLOCK_S)
                        g_row = (g[:, None] * C + u_c[None, :]).to(tl.int64)
                        g_mask = (g < seg_hi)[:, None] & u_mask[None, :]
                        total += tl.sum(
                            tl.load(
                                part_ptr
                                + g_row[:, :, None] * 128
                                + offs_d[None, None, :],
                                mask=g_mask[:, :, None],
                                other=0.0,
                                cache_modifier=".cg",
                            ),
                            axis=0,
                        )
                    norm = tl.sqrt(tl.sum(total * total, axis=1))
                    unit = total / tl.maximum(norm, 1e-30)[:, None]
                    old = tl.load(
                        c_f32_ptr + u_off,
                        mask=u_mask[:, None],
                        other=0.0,
                        cache_modifier=".cg",
                    )
                    new = tl.where((norm == 0)[:, None], old, unit)
                new = tl.where(u_live[:, None], new, 0.0)
                _store_centroid(new, c_f32_ptr, u_row, cen_ptr, u_rec, u_mask)
                _phase_done(sync)
                u = tl.atomic_add(sync + 1, 1, sem="relaxed")
            target = num_units
        nxt = p + 1
        if p < last:
            _phase_wait(sync, target)
            settled = tl.load(sync + 2, volatile=True) <= max_moved
            if (p % 2 == 1) & (p > 1) & settled:
                nxt = last
        p = nxt


@triton.jit
def ivf_run_scan_kernel(
    tile_counts_ptr,
    tile_first_ptr,
    req_ptr,
    cu_keys_ptr,
    tile_off_ptr,
    run_start_ptr,
    run_size_ptr,
    C: tl.constexpr,
    C_PAD: tl.constexpr,
    TILES: tl.constexpr,
):
    """Per request: where each tile's keys of a cluster start inside the
    cluster's run, the run sizes, and the run starts in packed key order."""
    i = tl.program_id(0)
    t0 = tl.load(tile_first_ptr + i)
    t1 = tl.load(tile_first_ptr + i + 1)
    offs_c = tl.arange(0, C_PAD)
    mask_c = offs_c < C
    carry = tl.zeros([C_PAD], tl.int32)
    for t in range(t0, t1, TILES):
        offs_t = t + tl.arange(0, TILES)
        m = (offs_t < t1)[:, None] & mask_c[None, :]
        off = offs_t[:, None] * C + offs_c[None, :]
        cnt = tl.load(tile_counts_ptr + off, mask=m, other=0)
        tl.store(tile_off_ptr + off, carry[None, :] + tl.cumsum(cnt, axis=0) - cnt, m)
        carry += tl.sum(cnt, axis=0)
    b = tl.load(req_ptr + i)
    start = tl.load(cu_keys_ptr + i) + tl.cumsum(carry, axis=0) - carry
    tl.store(run_size_ptr + b * C + offs_c, carry, mask=mask_c)
    tl.store(run_start_ptr + b * C + offs_c, start, mask=mask_c)


@triton.jit
def ivf_run_scatter_kernel(
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
):
    """Stable counting-sort scatter of one page of keys into cluster runs.

    Writes each key's position, fp8 bytes and scale at its slot of the
    (request, cluster) run, positions ascending within a run.
    """
    tile = tl.program_id(0)
    i = tl.load(tile_item_ptr + tile)
    start = tl.load(tile_start_ptr + tile)
    b = tl.load(req_ptr + i)
    n_keys = tl.load(n_keys_ptr + i)
    cu = tl.load(cu_keys_ptr + i)
    pos = start + tl.arange(0, PAGE)
    mask_t = pos < n_keys
    cl = tl.load(packed_cid_ptr + cu + pos, mask=mask_t, other=-1)
    dest = tl.zeros([PAGE], tl.int32)
    for c0 in range(0, C, BLOCK_C):
        offs_c = c0 + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C
        hit = cl[:, None] == offs_c[None, :]
        rank = tl.cumsum(hit.to(tl.int32), axis=0) - 1
        base = tl.load(run_start_ptr + b * C + offs_c, mask=mask_c, other=0)
        base += tl.load(tile_off_ptr + tile * C + offs_c, mask=mask_c, other=0)
        dest += tl.sum(tl.where(hit, rank + base[None, :], 0), axis=1)
    tl.store(run_pos_ptr + dest, pos, mask=mask_t)

    offs_d = tl.arange(0, 128)
    page = _key_page(k_bt_ptr, k_bt_stride, b, start, PAGE, True)
    src = page * k_page_stride + (pos % PAGE) * 128
    k = tl.load(k_u8_ptr + src[:, None] + offs_d[None, :], mask=mask_t[:, None])
    dst = dest.to(tl.int64)[:, None] * 128 + offs_d[None, :]
    tl.store(run_keys_ptr + dst, k, mask=mask_t[:, None])
    scale = _load_key_scales(k_f32_ptr, k_page_stride, page, pos, PAGE, mask_t)
    tl.store(run_scales_ptr + dest, scale, mask=mask_t)


@triton.jit
def ivf_decode_append_kernel(
    k_u8_ptr,
    k_page_stride,
    k_bt_ptr,
    k_bt_stride,
    cen_ptr,
    cen_i32_ptr,
    cen_stride,
    cen_bt_ptr,
    cen_bt_stride,
    cpb,
    cid_ptr,
    cid_stride,
    cid_bt_ptr,
    cid_bt_stride,
    cid_bs,
    seq_lens_ptr,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
    PAGE: tl.constexpr,
):
    """Assign the key just cached at position seq_len - 1 to its nearest centroid.

    A request whose first key this is (a one-token prompt) gets one cluster
    seeded by that key, which is what k-means over one key produces.
    """
    b = tl.program_id(0)
    pos = tl.load(seq_lens_ptr + b) - 1
    if pos >= 0:
        offs_d = tl.arange(0, 128)
        page = tl.load(k_bt_ptr + b * k_bt_stride + pos // PAGE).to(tl.int64)
        key = (
            tl.load(k_u8_ptr + page * k_page_stride + (pos % PAGE) * 128 + offs_d)
            .to(tl.float8e4nv, bitcast=True)
            .to(tl.float32)
        )

        best = tl.full([], float("-inf"), tl.float32)
        best_c = tl.full([], 0, tl.int32)
        for c0 in range(0, C, BLOCK_C):
            offs_c = c0 + tl.arange(0, BLOCK_C)
            mask_c = offs_c < C
            rec = _rec(cen_bt_ptr, cen_bt_stride, b, offs_c, cpb, cen_stride, mask_c)
            size_ptr = cen_i32_ptr + (rec + 256) // 4
            if pos == 0:
                tl.store(size_ptr, tl.full([BLOCK_C], -1, tl.int32), mask=mask_c)
            else:
                live = mask_c & (tl.load(size_ptr, mask=mask_c, other=-1) >= 0)
                cf = _load_centroids(cen_ptr, rec, live).to(tl.float32)
                sim = tl.sum(cf * key[None, :], axis=1)
                sim = tl.where(live, sim, float("-inf"))
                blk_best = tl.max(sim, axis=0)
                blk_arg = tl.argmax(sim, axis=0).to(tl.int32) + c0
                better = blk_best > best
                best = tl.where(better, blk_best, best)
                best_c = tl.where(better, blk_arg, best_c)

        # The first-key reset above must land before cluster 0 is written.
        tl.debug_barrier()
        blk0 = tl.load(cen_bt_ptr + b * cen_bt_stride + best_c // cpb).to(tl.int64)
        rec0 = blk0 * cen_stride + (best_c % cpb).to(tl.int64) * 272
        cblk = tl.load(cid_bt_ptr + b * cid_bt_stride + pos // cid_bs).to(tl.int64)
        tl.store(cid_ptr + cblk * cid_stride + pos % cid_bs, best_c.to(tl.int16))
        if pos == 0:
            unit = key / tl.maximum(tl.sqrt(tl.sum(key * key, axis=0)), 1e-12)
            tl.store(cen_ptr + rec0 // 2 + offs_d, unit.to(tl.bfloat16))
            tl.store(cen_i32_ptr + (rec0 + 256) // 4, 1)
        else:
            tl.atomic_add(cen_i32_ptr + (rec0 + 256) // 4, 1, sem="relaxed")


@triton.jit
def ivf_probe_score_kernel(
    q_fp8_ptr,
    w_ptr,
    row_batch_ptr,
    pos_ptr,
    cen_ptr,
    cen_i32_ptr,
    cen_stride,
    cen_bt_ptr,
    cen_bt_stride,
    cpb,
    run_start_ptr,
    run_pos_ptr,
    run_size_ptr,
    score_ws_ptr,
    vis_ws_ptr,
    num_rows,
    C: tl.constexpr,
    H: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    IS_DECODE: tl.constexpr,
):
    """Centroid scores and visible key counts of a rows x clusters tile.

    A row's score of cluster c is sum_h w_h ReLU(cos(q_h, c)) over its H fp8
    query heads ``q`` [rows, H, 128] and head weights ``w`` [rows, H]. Rows of
    one request share the centroid tile; fp8 is exact in bf16, so all
    ``BLOCK_R * H`` heads take one bf16 MMA with fp32 sums. Visible counts: the
    cluster size at decode, the run prefix at or before the row's position at
    prefill. -1 marks a disabled cluster.
    """
    offs_r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_d = tl.arange(0, 128)
    mask_r = offs_r < num_rows
    in_c = offs_c < C
    rows_b = tl.load(row_batch_ptr + offs_r, mask=mask_r, other=0)
    t = tl.load(pos_ptr + offs_r, mask=mask_r, other=-1).to(tl.int32)
    RH: tl.constexpr = BLOCK_R * H
    heads = tl.reshape(offs_r[:, None] * H + tl.arange(0, H)[None, :], (RH,))
    mask_rh = heads < num_rows * H
    q = tl.load(
        q_fp8_ptr + heads.to(tl.int64)[:, None] * 128 + offs_d[None, :],
        mask=mask_rh[:, None],
        other=0.0,
    ).to(tl.float32)
    inv_norm = 1.0 / tl.maximum(tl.sqrt(tl.sum(q * q, axis=1)), 1e-12)
    q = q.to(tl.bfloat16)
    w = tl.load(w_ptr + heads, mask=mask_rh, other=0.0)
    out = offs_r[:, None] * C + offs_c[None, :]

    pending = mask_r
    while tl.max(pending.to(tl.int32), axis=0) > 0:
        b = tl.min(tl.where(pending, rows_b, 2147483647), axis=0)
        mine = pending & (rows_b == b)
        rec = _rec(cen_bt_ptr, cen_bt_stride, b, offs_c, cpb, cen_stride, in_c)
        size = tl.load(cen_i32_ptr + (rec + 256) // 4, mask=in_c, other=-1)
        live = in_c & (size >= 0)
        cf = _load_centroids(cen_ptr, rec, live)
        cos = tl.dot(q, tl.trans(cf)) * inv_norm[:, None]
        per_head = tl.maximum(cos, 0.0) * w[:, None]
        score = tl.sum(tl.reshape(per_head, (BLOCK_R, H, BLOCK_C)), axis=1)
        if IS_DECODE:
            # Padded graph rows (seq_len 0, t < 0) see nothing and emit -1.
            vis = tl.where((t >= 0)[:, None], size[None, :], 0)
        else:
            # Runs hold positions ascending, so the visible part is a prefix;
            # fully visible and fully hidden runs skip the search.
            size = tl.load(run_size_ptr + b * C + offs_c, mask=live, other=0)
            run_start = tl.load(run_start_ptr + b * C + offs_c, mask=live, other=0)
            has = size > 0
            first = tl.load(run_pos_ptr + run_start, mask=has, other=0)
            last = tl.load(run_pos_ptr + run_start + size - 1, mask=has, other=0)
            tt = t[:, None]
            partial = has[None, :] & (first[None, :] <= tt) & (last[None, :] > tt)
            full = has[None, :] & (last[None, :] <= tt)
            lo = tl.where(full, size[None, :], tl.where(partial, 1, 0))
            hi = tl.where(partial, size[None, :] - 1, lo)
            while tl.max(tl.max((lo < hi).to(tl.int32), axis=1), axis=0) > 0:
                active = lo < hi
                mid = (lo + hi) // 2
                p = tl.load(
                    run_pos_ptr + run_start[None, :] + mid, mask=active, other=0
                )
                lo = tl.where(active & (p <= tt), mid + 1, lo)
                hi = tl.where(active & (p > tt), mid, hi)
            vis = lo
        m = mine[:, None] & in_c[None, :]
        tl.store(score_ws_ptr + out, score, mask=m)
        tl.store(vis_ws_ptr + out, tl.where(live[None, :], vis, -1), mask=m)
        pending = pending & ~mine


@triton.jit
def ivf_probe_select_kernel(
    pos_ptr,
    score_ws_ptr,
    vis_ws_ptr,
    sel_out_ptr,
    ends_out_ptr,
    count_out_ptr,
    total_out_ptr,
    topk_len_out_ptr,
    cand_ptr,
    cand_width,
    num_probes,
    TOPK: tl.constexpr,
    C: tl.constexpr,
    C_PAD: tl.constexpr,
    IS_DECODE: tl.constexpr,
):
    """Probe a fixed number of clusters per row and lay out its candidates.

    A row probes the ``num_probes`` best clusters among those with a key
    visible to it, by their score (``ivf_probe_score_kernel``), ties to the
    lower cluster id. Writes which clusters are probed, the
    inclusive candidate end of every cluster (cluster-id order), the candidate
    count (every visible key of the probed clusters) and the top-k length (0
    when the row keeps every candidate).
    """
    row = tl.program_id(0)
    offs_all = tl.arange(0, C_PAD)
    in_range = offs_all < C
    score = tl.load(score_ws_ptr + row * C + offs_all, mask=in_range, other=0.0)
    vis = tl.load(vis_ws_ptr + row * C + offs_all, mask=in_range, other=-1)
    eligible = vis > 0
    n_eligible = tl.sum(eligible.to(tl.int32), axis=0)
    none = -1099511627776  # below every float order key
    # Unique (score, lower id first) keys: the k-th largest selects exactly k.
    keys = tl.where(eligible, _float_order_key(score), none) * 65536 + (
        65535 - offs_all
    )
    k = tl.minimum(num_probes, n_eligible)
    tau = _kth_largest(tl.sort(keys, descending=True), tl.maximum(k, 1), C_PAD)
    selected = eligible & (keys >= tau) & (k > 0)
    take = tl.where(selected, vis, 0)
    ends = tl.cumsum(take, axis=0)
    total = tl.sum(take, axis=0)
    tl.store(sel_out_ptr + row * C + offs_all, selected.to(tl.int8), mask=in_range)
    tl.store(ends_out_ptr + row * C + offs_all, ends, mask=in_range)
    tl.store(total_out_ptr + row, total)
    if IS_DECODE:
        # Degenerate probe: fall back to the query's own position. (Prefill
        # emits it from the run layout.)
        t = tl.load(pos_ptr + row).to(tl.int32)
        tl.store(cand_ptr + row * cand_width, t, mask=total == 0)
    count = tl.maximum(total, 1)
    tl.store(count_out_ptr + row, count)
    tl.store(topk_len_out_ptr + row, tl.where(count > TOPK, count, 0))


@triton.jit
def ivf_scan_count_kernel(
    row_batch_ptr,
    seq_lens_ptr,
    sel_ptr,
    cid_ptr,
    cid_stride,
    cid_bt_ptr,
    cid_bt_stride,
    cid_bs,
    block_count_ptr,
    num_blocks,
    C: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Decode pass 1: count each block's positions whose cluster is probed."""
    row = tl.program_id(0)
    blk = tl.program_id(1)
    b = tl.load(row_batch_ptr + row)
    seq_len = tl.load(seq_lens_ptr + row)
    count = tl.full([], 0, tl.int32)
    if blk * BLOCK < seq_len:
        pos = blk * BLOCK + tl.arange(0, BLOCK)
        m = pos < seq_len
        cblk = tl.load(
            cid_bt_ptr + b * cid_bt_stride + pos // cid_bs, mask=m, other=0
        ).to(tl.int64)
        cl = tl.load(cid_ptr + cblk * cid_stride + pos % cid_bs, mask=m, other=0)
        flags = tl.load(sel_ptr + row * C + cl.to(tl.int32), mask=m, other=0)
        count = tl.sum(flags.to(tl.int32), axis=0)
    tl.store(block_count_ptr + row * num_blocks + blk, count)


@triton.jit
def ivf_scan_write_kernel(
    row_batch_ptr,
    seq_lens_ptr,
    sel_ptr,
    cid_ptr,
    cid_stride,
    cid_bt_ptr,
    cid_bt_stride,
    cid_bs,
    block_count_ptr,
    num_blocks,
    cand_ptr,
    cand_cap,
    C: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCKS_PAD: tl.constexpr,
):
    """Decode pass 2: write probed positions in ascending order, capped."""
    row = tl.program_id(0)
    blk = tl.program_id(1)
    b = tl.load(row_batch_ptr + row)
    seq_len = tl.load(seq_lens_ptr + row)
    if blk * BLOCK < seq_len:
        pos = blk * BLOCK + tl.arange(0, BLOCK)
        m = pos < seq_len
        cblk = tl.load(
            cid_bt_ptr + b * cid_bt_stride + pos // cid_bs, mask=m, other=0
        ).to(tl.int64)
        cl = tl.load(cid_ptr + cblk * cid_stride + pos % cid_bs, mask=m, other=0)
        sel = tl.load(sel_ptr + row * C + cl.to(tl.int32), mask=m, other=0) != 0
        prior = tl.arange(0, NUM_BLOCKS_PAD)
        base = tl.sum(
            tl.load(
                block_count_ptr + row * num_blocks + prior, mask=prior < blk, other=0
            )
        )
        rank = base + tl.cumsum(sel.to(tl.int32), axis=0) - 1
        tl.store(cand_ptr + row * cand_cap + rank, pos, mask=sel & (rank < cand_cap))


@triton.jit
def ivf_score_kernel(
    q_fp8_ptr,
    w_ptr,
    row_batch_ptr,
    cand_ptr,
    count_ptr,
    k_u8_ptr,
    k_f32_ptr,
    k_page_stride,
    k_bt_ptr,
    k_bt_stride,
    score_ptr,
    cand_cap,
    TOPK: tl.constexpr,
    H: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """Scores sum_h w_h ReLU(q_h . k) * k_scale for every NUM_SPLITS-th tile.

    The DSA indexer score: fp8 query heads ``q`` [rows, H, 128] against fp8
    keys, with the query scales folded into the head weights ``w`` [rows, H].
    """
    row = tl.program_id(0)
    count = tl.load(count_ptr + row)
    # A row at or under top-k keeps every candidate; its scores are never read.
    if count > TOPK:
        b = tl.load(row_batch_ptr + row)
        offs_h = tl.arange(0, H)
        q = tl.load(
            q_fp8_ptr + (row * H + offs_h)[:, None] * 128 + tl.arange(0, 128)[None, :]
        )
        w = tl.load(w_ptr + row * H + offs_h)
        for j0 in tl.range(
            tl.program_id(1) * BLOCK_N,
            count,
            NUM_SPLITS * BLOCK_N,
            num_stages=NUM_STAGES,
        ):
            offs_n = j0 + tl.arange(0, BLOCK_N)
            mask_n = offs_n < count
            pos = tl.load(cand_ptr + row * cand_cap + offs_n, mask=mask_n, other=0)
            page = _key_page(k_bt_ptr, k_bt_stride, b, pos, PAGE, mask_n)
            k = _load_keys(k_u8_ptr, k_page_stride, page, pos, PAGE, mask_n)
            k_scale = _load_key_scales(
                k_f32_ptr, k_page_stride, page, pos, PAGE, mask_n
            )
            per_head = tl.maximum(tl.dot(k, tl.trans(q)), 0.0) * w[None, :]
            score = tl.sum(per_head, axis=1) * k_scale
            tl.store(score_ptr + row * cand_cap + offs_n, score, mask=mask_n)


@triton.jit
def ivf_group_count_kernel(
    sel_ptr,
    vis_ws_ptr,
    topk_len_ptr,
    row_batch_ptr,
    group_m_ptr,
    group_vis_ptr,
    num_rows,
    C: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Per (batch row, cluster) group: how many over-budget query rows probed
    the cluster, and the most keys of it any of them sees."""
    offs_r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_r = offs_r < num_rows
    need = tl.load(topk_len_ptr + offs_r, mask=mask_r, other=0) > 0
    b = tl.load(row_batch_ptr + offs_r, mask=mask_r, other=0)
    m = (need & mask_r)[:, None] & (offs_c < C)[None, :]
    off = offs_r[:, None] * C + offs_c[None, :]
    hit = m & (tl.load(sel_ptr + off, mask=m, other=0) != 0)
    vis = tl.load(vis_ws_ptr + off, mask=hit, other=0)
    # Rows of the tile's first request reduce first: one atomic per cluster.
    b0 = tl.load(row_batch_ptr + tl.program_id(0) * BLOCK_R)
    same = hit & (b == b0)[:, None]
    n0 = tl.sum(same.to(tl.int32), axis=0)
    v0 = tl.max(tl.where(same, vis, 0), axis=0)
    g0 = b0 * C + offs_c
    tl.atomic_add(group_m_ptr + g0, n0, mask=n0 > 0, sem="relaxed")
    tl.atomic_max(group_vis_ptr + g0, v0, mask=n0 > 0, sem="relaxed")
    other = hit & ~same
    g = b[:, None] * C + offs_c[None, :]
    one = tl.full([BLOCK_R, BLOCK_C], 1, tl.int32)
    tl.atomic_add(group_m_ptr + g, one, other, sem="relaxed")
    tl.atomic_max(group_vis_ptr + g, vis, other, sem="relaxed")


@triton.jit
def ivf_group_offsets_kernel(
    group_m_ptr,
    group_vis_ptr,
    group_off_ptr,
    cursor_ptr,
    unit_off_ptr,
    num_groups,
    BLOCK_M: tl.constexpr,
    UNIT_ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program: each group's first row-list slot (also its fill cursor)
    and first work unit; unit_off[G] is the unit total.

    A unit is one BLOCK_M-key tile of a group against UNIT_ROWS of its rows.
    """
    rows_carry = tl.full([], 0, tl.int32)
    unit_carry = tl.full([], 0, tl.int32)
    tl.store(unit_off_ptr, 0)
    for g0 in range(0, num_groups, BLOCK):
        g = g0 + tl.arange(0, BLOCK)
        mask = g < num_groups
        m = tl.load(group_m_ptr + g, mask=mask, other=0)
        vis = tl.load(group_vis_ptr + g, mask=mask, other=0)
        units = tl.cdiv(vis, BLOCK_M) * tl.cdiv(m, UNIT_ROWS)
        start = rows_carry + tl.cumsum(m, axis=0) - m
        tl.store(group_off_ptr + g, start, mask=mask)
        tl.store(cursor_ptr + g, start, mask=mask)
        tl.store(unit_off_ptr + 1 + g, unit_carry + tl.cumsum(units, axis=0), mask)
        rows_carry += tl.sum(m, axis=0)
        unit_carry += tl.sum(units, axis=0)


@triton.jit
def ivf_group_fill_kernel(
    sel_ptr,
    topk_len_ptr,
    row_batch_ptr,
    cursor_ptr,
    row_list_ptr,
    num_rows,
    C: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Append every over-budget row to the row list of each group it probed.

    Order inside a group is arbitrary; each score depends only on its own
    query row and key, so results do not.
    """
    offs_r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_r = offs_r < num_rows
    need = tl.load(topk_len_ptr + offs_r, mask=mask_r, other=0) > 0
    b = tl.load(row_batch_ptr + offs_r, mask=mask_r, other=0)
    m = (need & mask_r)[:, None] & (offs_c < C)[None, :]
    hit = m & (tl.load(sel_ptr + offs_r[:, None] * C + offs_c[None, :], m, 0) != 0)
    # As in group_count: the first request's rows take one cursor bump per
    # cluster and number themselves by rank.
    b0 = tl.load(row_batch_ptr + tl.program_id(0) * BLOCK_R)
    same = hit & (b == b0)[:, None]
    n0 = tl.sum(same.to(tl.int32), axis=0)
    base = tl.atomic_add(cursor_ptr + b0 * C + offs_c, n0, mask=n0 > 0, sem="relaxed")
    rank = tl.cumsum(same.to(tl.int32), axis=0) - 1
    other = hit & ~same
    g = b[:, None] * C + offs_c[None, :]
    one = tl.full([BLOCK_R, BLOCK_C], 1, tl.int32)
    slot = tl.atomic_add(cursor_ptr + g, one, other, sem="relaxed")
    slot = tl.where(same, base[None, :] + rank, slot)
    rows = tl.broadcast_to(offs_r[:, None], [BLOCK_R, BLOCK_C])
    tl.store(row_list_ptr + slot, rows, mask=hit)


@triton.jit
def ivf_grouped_score_kernel(
    q_fp8_ptr,
    w_ptr,
    run_keys_ptr,
    run_scales_ptr,
    run_start_ptr,
    group_m_ptr,
    group_vis_ptr,
    group_off_ptr,
    unit_off_ptr,
    row_list_ptr,
    vis_ws_ptr,
    ends_ptr,
    score_ptr,
    cand_cap,
    num_groups,
    C: tl.constexpr,
    H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    ROWS: tl.constexpr,
    TILES: tl.constexpr,
    LOG_G: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """Persistent grouped GEMM: per (batch row, cluster) group, the cluster's
    keys times the query heads of every row that probed it.

    A work unit holds one BLOCK_M-key tile and streams TILES tiles of ROWS
    rows, each with H fp8 query heads (gathered from ``q_fp8`` [rows, H, 128]).
    Each step is one ``[BLOCK_M, 128] x [128, ROWS * H]`` fp8 product (wgmma);
    the DSA score sum_h w_h ReLU(q_h . k) times the key scale goes to the
    row's candidate slot.
    """
    RH: tl.constexpr = ROWS * H
    offs_d = tl.arange(0, 128)
    total = tl.load(unit_off_ptr + num_groups)
    for u in range(tl.program_id(0), total, tl.num_programs(0)):
        # The group holding unit u: the first g with unit_off[g + 1] > u.
        lo = tl.full([], 0, tl.int32)
        hi = tl.full([], 0, tl.int32) + num_groups
        for _ in range(LOG_G):
            active = lo < hi
            mid = (lo + hi) // 2
            right = tl.load(unit_off_ptr + mid + 1, mask=active, other=0) <= u
            new_lo = tl.where(active & right, mid + 1, lo)
            hi = tl.where(active & ~right, mid, hi)
            lo = new_lo
        g = lo
        rel = u - tl.load(unit_off_ptr + g)
        vis_max = tl.load(group_vis_ptr + g)
        n_key_tiles = tl.cdiv(vis_max, BLOCK_M)
        m0 = (rel % n_key_tiles) * BLOCK_M
        r0 = (rel // n_key_tiles) * (ROWS * TILES)
        m_g = tl.load(group_m_ptr + g)
        c = g % C
        row_base = tl.load(group_off_ptr + g)

        offs_m = m0 + tl.arange(0, BLOCK_M)
        mask_m = offs_m < vis_max
        key0 = tl.load(run_start_ptr + g).to(tl.int64)
        k = tl.load(
            run_keys_ptr + (key0 + offs_m)[:, None] * 128 + offs_d[None, :],
            mask=mask_m[:, None],
            other=0.0,
        )
        k_scale = tl.load(run_scales_ptr + key0 + offs_m, mask=mask_m, other=0.0)
        # Row indices run one tile ahead, so a tile's gather never waits on an
        # index load of the same iteration.
        offs_n = r0 + tl.arange(0, ROWS)
        rows_next = tl.load(
            row_list_ptr + row_base + offs_n, mask=offs_n < m_g, other=0
        )
        for t in tl.range(0, TILES, num_stages=NUM_STAGES):
            offs_r = r0 + t * ROWS + tl.arange(0, ROWS)
            mask_r = offs_r < m_g
            rows = rows_next
            offs_n = offs_r + ROWS
            rows_next = tl.load(
                row_list_ptr + row_base + offs_n, mask=offs_n < m_g, other=0
            )
            heads = tl.reshape(
                rows.to(tl.int64)[:, None] * H + tl.arange(0, H)[None, :], (RH,)
            )
            q = tl.load(q_fp8_ptr + heads[:, None] * 128 + offs_d[None, :])
            w = tl.load(w_ptr + heads)
            per_head = tl.maximum(tl.dot(k, tl.trans(q)), 0.0) * w[None, :]
            score = tl.sum(tl.reshape(per_head, (BLOCK_M, ROWS, H)), axis=2)
            score = score * k_scale[:, None]
            vis = tl.load(vis_ws_ptr + rows * C + c, mask=mask_r, other=0)
            first = tl.load(ends_ptr + rows * C + c, mask=mask_r, other=0) - vis
            slot = first[None, :] + offs_m[:, None]
            keep = (
                mask_r[None, :] & (offs_m[:, None] < vis[None, :]) & (slot < cand_cap)
            )
            dst = rows.to(tl.int64)[None, :] * cand_cap + slot
            tl.store(score_ptr + dst, score, mask=keep, eviction_policy="evict_first")


@triton.jit
def ivf_emit_kernel(
    count_ptr,
    selected_ptr,
    cand_ptr,
    out_ptr,
    out_stride,
    cand_cap,
    out_width,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Turn selected candidate slots into request-local positions, -1 padded."""
    row = tl.program_id(0)
    j = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    count = tl.load(count_ptr + row)
    under = count <= TOPK
    picked = tl.load(
        selected_ptr + row * TOPK + j, mask=(j < TOPK) & (~under), other=-1
    )
    slot = tl.where(under, tl.where(j < count, j, -1), picked)
    live = (slot >= 0) & (j < TOPK)
    pos = tl.load(cand_ptr + row * cand_cap + slot, mask=live, other=0)
    tl.store(
        out_ptr + row * out_stride + j, tl.where(live, pos, -1), mask=j < out_width
    )


@triton.jit
def ivf_emit_runs_kernel(
    count_ptr,
    total_ptr,
    selected_ptr,
    row_batch_ptr,
    pos_ptr,
    ends_ptr,
    run_start_ptr,
    run_pos_ptr,
    out_ptr,
    out_stride,
    out_width,
    TOPK: tl.constexpr,
    C: tl.constexpr,
    LOG_C: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Prefill emit: map selected candidate slots straight to positions.

    Slots run cluster after cluster (``ends``), each the visible prefix of the
    cluster's run, so slot j is entry j - start of the first cluster whose end
    > j. Only the selected slots are mapped; a row without candidates emits
    its own position.
    """
    row = tl.program_id(0)
    j = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    count = tl.load(count_ptr + row)
    total = tl.load(total_ptr + row)
    under = count <= TOPK
    picked = tl.load(
        selected_ptr + row * TOPK + j, mask=(j < TOPK) & (~under), other=-1
    )
    slot = tl.where(under, tl.where(j < count, j, -1), picked)
    live = (slot >= 0) & (j < TOPK)
    found = live & (total > 0)
    lo = tl.zeros([BLOCK], tl.int32)
    hi = tl.where(found, C, 0)
    for _ in range(LOG_C):
        active = lo < hi
        mid = (lo + hi) // 2
        end = tl.load(ends_ptr + row * C + mid, mask=active, other=0)
        lo = tl.where(active & (end <= slot), mid + 1, lo)
        hi = tl.where(active & (end > slot), mid, hi)
    start = tl.load(ends_ptr + row * C + lo - 1, mask=found & (lo > 0), other=0)
    b = tl.load(row_batch_ptr + row)
    run_start = tl.load(run_start_ptr + b * C + lo, mask=found, other=0)
    p = tl.load(run_pos_ptr + run_start + (slot - start), mask=found, other=0)
    p = tl.where(total > 0, p, tl.load(pos_ptr + row).to(tl.int32))
    tl.store(out_ptr + row * out_stride + j, tl.where(live, p, -1), mask=j < out_width)
