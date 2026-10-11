# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""IVF (k-means) top-k selection for the glm5next DSA indexer.

The algorithm of ``ivf-dsa-kmeans/compare_decode_kmeans.py`` over the kpool
indexer's pooled keys: the (single-head, MQA) keys are clustered by cosine
k-means per request and per layer into a fixed number of clusters. A query
keeps its H fp8 indexer heads q_h and head weights w_h (query scales folded in,
as for the stock indexer). It probes the fixed number of clusters with the best
DSA score of their unit centroid, sum_h w_h ReLU(q_h . c), scores every visible
key of them by the DSA score sum_h w_h ReLU(q_h . k), and selects the top-k,
which expand to tokens plus the tail pool as for the kpool indexer.

Multi-token steps (prefill and its chunks) rebuild the request's index over
prefix + chunk, or append the chunk to it (``build_ivf_index``); a one-token
step that completes a pool assigns it to the nearest fixed centroid. Decode
append and select take no host data, so the decode path is safe under CUDA
graph capture.
"""

import functools
import math
from dataclasses import dataclass

import numpy as np
import torch

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import get_current_vllm_config_or_none
from vllm.forward_context import get_forward_context
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.indexer_topk import get_indexer_topk
from vllm.models.glm5next.common.ivf_backend import (
    IvfIndexerMetadata,
    IvfStateMetadata,
)
from vllm.models.glm5next.common.ivf_config import IvfIndexerConfig
from vllm.models.glm5next.common.sparse_indexer import RADIX_TOPK_WORKSPACE_SIZE
from vllm.models.glm5next.nvidia.ops import kpool_compress as kpool_ops
from vllm.models.glm5next.nvidia.ops.ivf_index import (
    REC_BYTES,
    ivf_decode_append_kernel,
    ivf_emit_kernel,
    ivf_emit_runs_kernel,
    ivf_group_count_kernel,
    ivf_group_fill_kernel,
    ivf_group_offsets_kernel,
    ivf_group_scan_kernel,
    ivf_grouped_score_kernel,
    ivf_kmeans_append_kernel,
    ivf_kmeans_append_update_kernel,
    ivf_kmeans_kernel,
    ivf_probe_score_kernel,
    ivf_probe_select_kernel,
    ivf_run_scan_kernel,
    ivf_run_scatter_kernel,
    ivf_scan_count_kernel,
    ivf_scan_write_kernel,
    ivf_score_kernel,
)
from vllm.models.glm5next.nvidia.sparse_indexer import _kpool_compress_insert
from vllm.triton_utils import triton
from vllm.utils.deep_gemm import PAGED_MQA_PAGE_SIZES
from vllm.v1.worker.workspace import current_workspace_manager


@dataclass
class IvfKernelTuning:
    """Launch parameters, tuned on H200; sweep again on other GPUs."""

    centroid_block: int = 64
    score_block_n: int = 128
    score_num_warps: int = 4
    score_num_stages: int = 3
    # Score programs per SM to aim for; a row splits its tiles until reached.
    score_programs_per_sm: int = 4
    scan_block: int = 2048
    # Key tiles per step, and clusters per program, of the run-layout scan.
    scan_tiles: int = 128
    scan_block_c: int = 16
    emit_block: int = 256
    probe_num_warps: int = 4
    # Query rows per probe-score tile; its MMA has probe_rows x heads rows.
    probe_rows: int = 8
    # Fused k-means: programs per SM (its register-held [C, 128] member sums
    # fit one), warps, and clusters per seed/update unit. A request's keys
    # split into about kmeans_segments assign units of at least
    # kmeans_min_segment_tiles pages; both fix the summation order, so they
    # must not depend on the GPU.
    kmeans_programs_per_sm: int = 1
    kmeans_num_warps: int = 8
    kmeans_update_block: int = 2
    # Segment sums an update unit loads at once.
    kmeans_update_segments: int = 64
    kmeans_segments: int = 128
    kmeans_min_segment_tiles: int = 4
    # Keys per assign tile (whole pages, at least one) up to
    # kmeans_wide_tile_clusters clusters: 64 make both of its MMAs wgmma,
    # which pays off only while the [C, 128] member sums leave registers.
    kmeans_tile_keys: int = 64
    kmeans_wide_tile_clusters: int = 128
    # Prefill grouped GEMM: a work unit is 1 or gemm_key_chunks TMA-loaded
    # chunks of gemm_block_n keys of a cluster against up to gemm_row_tiles
    # tiles of gemm_mma_cols // heads of its query rows (gemm_mma_cols MMA
    # columns); a tile's queries are gathered once for all its chunks. The
    # gathers are bound by L2 bandwidth and latency: deep pipelines and
    # several programs per SM keep enough in flight (H200). Clusters of at
    # least gemm_chunked_keys keys on average take the chunked units, which
    # hold more shared memory (fewer programs per SM).
    gemm_block_n: int = 64
    gemm_key_chunks: int = 4
    gemm_chunked_keys: int = 96
    gemm_mma_cols: int = 128
    gemm_row_tiles: int = 32
    gemm_num_warps: int = 4
    gemm_num_stages: int = 5
    gemm_programs_per_sm: int = 3
    gemm_chunked_programs_per_sm: int = 2
    plan_block_r: int = 64
    plan_block_c: int = 64
    # Row tiles per step of the per-cluster row-tile scan.
    plan_scan_tiles: int = 16


TUNING = IvfKernelTuning()


def _alloc(
    dtype: torch.dtype,
    device: torch.device,
    *shapes: tuple[int, ...],
    zero: bool = False,
) -> list[torch.Tensor]:
    """Tensors of ``shapes`` from one allocation (one fill when ``zero``),
    each 256 B aligned."""
    align = 256 // dtype.itemsize
    numels = [math.prod(shape) for shape in shapes]
    sizes = [-(-n // align) * align for n in numels]
    new = torch.zeros if zero else torch.empty
    buf = new((sum(sizes),), dtype=dtype, device=device)
    return [
        part[:n].view(shape) for part, n, shape in zip(buf.split(sizes), numels, shapes)
    ]


@functools.cache
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@dataclass(frozen=True)
class IvfKeyView:
    """The pooled key cache of one layer, as ``[pages, PAGE, 132]`` bytes."""

    u8: torch.Tensor
    f32: torch.Tensor
    page_stride: int
    page: int
    block_table: torch.Tensor

    @classmethod
    def of(cls, kv_cache: torch.Tensor, block_table: torch.Tensor) -> "IvfKeyView":
        assert kv_cache.dtype == torch.uint8 and kv_cache.shape[-1] == 132
        assert block_table.stride(1) == 1
        return cls(
            u8=kv_cache,
            f32=kv_cache.view(torch.float32),
            page_stride=kv_cache.stride(0),
            page=kv_cache.shape[1],
            block_table=block_table,
        )


@dataclass(frozen=True)
class IvfStateView:
    """Cluster ids ``[blocks, cid_bs]`` int16 and centroid records of a layer."""

    cid: torch.Tensor
    cid_stride: int
    cid_bs: int
    cid_bt: torch.Tensor
    cen: torch.Tensor
    cen_i32: torch.Tensor
    cen_stride: int
    cpb: int
    cen_bt: torch.Tensor
    num_clusters: int

    @classmethod
    def of(
        cls,
        cid_cache: torch.Tensor,
        cen_cache: torch.Tensor,
        cid_bt: torch.Tensor,
        cen_bt: torch.Tensor,
        num_clusters: int,
    ) -> "IvfStateView":
        assert cid_cache.dtype == torch.int16
        assert cen_cache.dtype == torch.uint8 and cen_cache.shape[-1] == REC_BYTES
        assert cid_bt.stride(1) == 1 and cen_bt.stride(1) == 1
        return cls(
            cid=cid_cache,
            cid_stride=cid_cache.stride(0),
            cid_bs=cid_cache.shape[1],
            cid_bt=cid_bt,
            cen=cen_cache.view(torch.bfloat16),
            cen_i32=cen_cache.view(torch.int32),
            cen_stride=cen_cache.stride(0),
            cpb=cen_cache.shape[1],
            cen_bt=cen_bt,
            num_clusters=num_clusters,
        )

    def cid_args(self):
        cid_bt_stride = self.cid_bt.stride(0)
        return (self.cid, self.cid_stride, self.cid_bt, cid_bt_stride, self.cid_bs)

    def cen_args(self):
        return (
            self.cen,
            self.cen_i32,
            self.cen_stride,
            self.cen_bt,
            self.cen_bt.stride(0),
            self.cpb,
        )


@dataclass(frozen=True)
class IvfRunLayout:
    """Cluster-contiguous key layout of the requests built this step."""

    # [sum keys] int32 positions, ascending inside each (request, cluster) run.
    run_pos: torch.Tensor
    # [B, C] int32 first slot of each run in run_pos, and the run sizes; row b
    # is batch row b (rows of requests not built this step are unused).
    run_start: torch.Tensor
    run_size: torch.Tensor
    # [sum keys, 128] fp8 keys and [sum keys] fp32 scales in run_pos order: each
    # (request, cluster) run is a contiguous key matrix.
    run_keys: torch.Tensor
    run_scales: torch.Tensor
    # [B * C, 128] bf16 centroids and [B, C] int32 sizes (-1: disabled) of the
    # requests built this step, densely for the prefill probe's TMA loads.
    cen_dense: torch.Tensor
    cen_size: torch.Tensor


def build_ivf_index(
    *,
    key: IvfKeyView,
    state: IvfStateView,
    batch_rows_cpu: list[int],
    n_keys_cpu: list[int],
    num_batch_rows: int,
    config: IvfIndexerConfig,
    n_prev_keys_cpu: list[int] | None = None,
    final_cpu: list[bool] | None = None,
) -> IvfRunLayout:
    """Cluster every listed request's cached keys, replacing its IVF state.

    ``n_prev_keys_cpu`` are the keys each request had before this step (0 for
    a request's first step). A request with a previous chunk and the same
    cluster count either appends (its first ``n_prev`` keys keep their
    clusters; the chunk's keys are assigned and folded into the centroids) or
    rebuilds from its previous centroids with ``kmeans_warm_iters``: on its
    final prefill chunk (``final_cpu``, default none), when its key count
    crosses a power of two, or when ``kmeans_incremental`` is off.
    """
    if n_prev_keys_cpu is None:
        n_prev_keys_cpu = [0] * len(n_keys_cpu)
    if final_cpu is None:
        final_cpu = [False] * len(n_keys_cpu)
    device = key.u8.device
    C = state.num_clusters
    page = key.page
    tune = TUNING

    def clusters(n: int) -> int:
        return max(min(config.num_clusters, n, C), 1)

    def mode(n: int, n_prev: int, final: bool) -> str:
        if n_prev == 0 or clusters(n_prev) != clusters(n):
            return "cold"
        if (
            config.kmeans_incremental
            and not final
            and n_prev.bit_length() == n.bit_length()
        ):
            return "append"
        return "warm"

    modes = [mode(n, p, f) for n, p, f in zip(n_keys_cpu, n_prev_keys_cpu, final_cpu)]
    # Full builds first: the fused k-means takes the leading items.
    order = sorted(range(len(modes)), key=lambda i: modes[i] == "append")
    batch_rows_cpu = [batch_rows_cpu[i] for i in order]
    n_keys_cpu = [n_keys_cpu[i] for i in order]
    n_prev_keys_cpu = [n_prev_keys_cpu[i] for i in order]
    modes = [modes[i] for i in order]
    num_items = len(n_keys_cpu)
    num_full = sum(m != "append" for m in modes)

    # Per-item tables in one host copy: tiles (pages of keys), k-means
    # segments and the appended items' new tiles, each with item offsets.
    n = np.asarray(n_keys_cpu)
    n_prev = np.asarray(n_prev_keys_cpu)
    tiles = -(-n // page)
    seg_tiles = np.maximum(
        -(-tiles // tune.kmeans_segments), tune.kmeans_min_segment_tiles
    )
    grew = (np.asarray(modes) == "append") & (n_prev < n)

    def firsts(counts: np.ndarray) -> np.ndarray:
        return np.concatenate(([0], np.cumsum(counts)))

    tile_first = firsts(tiles)
    seg_first = firsts(-(-tiles // seg_tiles))
    new_tile_first = firsts(np.where(grew, tiles - n_prev // page, 0))
    cu_keys = firsts(n)
    num_tiles, total_keys = int(tile_first[-1]), int(cu_keys[-1])
    tile_item = np.repeat(np.arange(num_items), tiles)
    parts = [
        tile_item,
        (np.arange(num_tiles) - tile_first[tile_item]) * page,
        np.asarray(batch_rows_cpu),
        n,
        n_prev,
        cu_keys[:-1],
        tile_first,
        seg_first,
        new_tile_first,
    ]
    host = torch.empty((sum(map(len, parts)),), dtype=torch.int32, pin_memory=True)
    np.concatenate(parts, out=host.numpy(), casting="unsafe")
    host = host.to(device, non_blocking=True)
    (
        tile_item_t,
        tile_start_t,
        req_t,
        n_keys_t,
        n_prev_t,
        cu_keys_t,
        tile_first_t,
        seg_first_t,
        new_tile_first_t,
    ) = host.split(
        [
            num_tiles,
            num_tiles,
            num_items,
            num_items,
            num_items,
            num_items,
            num_items + 1,
            num_items + 1,
            num_items + 1,
        ]
    )

    # One launch runs one iteration count: a batch with a cold build runs
    # every full build with kmeans_iters.
    iters = (
        config.kmeans_iters
        if "cold" in modes
        else min(config.kmeans_warm_iters, config.kmeans_iters)
    )
    num_phases = 2 * iters + 2
    num_segs = int(seg_first[num_full])
    # One zero fill: the k-means phase counters, and the counting sort's run
    # sizes, run starts and per-request counter.
    sync, run_size, run_start, scan_done = _alloc(
        torch.int32,
        device,
        (num_phases * (num_full + 3) if num_full else 0,),
        (num_batch_rows, C),
        (num_batch_rows, C),
        (num_items,),
        zero=True,
    )
    packed_cid, run_pos, tile_counts, tile_off, cen_size = _alloc(
        torch.int32,
        device,
        (total_keys,),
        (total_keys,),
        (num_tiles, C),
        (num_tiles, C),
        (num_batch_rows, C),
    )
    cen_dense = torch.empty(
        (num_batch_rows * C, 128), dtype=torch.bfloat16, device=device
    )
    run_scales, c_f32, part = _alloc(
        torch.float32,
        device,
        (total_keys,),
        (num_full, C, 128),
        (num_segs if num_full else 0, C, 128),
    )
    C_PAD = max(triton.next_power_of_2(C), 32)
    if num_full > 0:
        full_keys = int(cu_keys[num_full])
        units = num_full * triton.cdiv(C, tune.kmeans_update_block)
        programs = tune.kmeans_programs_per_sm * _num_sms(device.index or 0)
        wide = tune.kmeans_wide_tile_clusters >= C_PAD
        tile_pages = max(tune.kmeans_tile_keys // page, 1) if wide else 1
        ivf_kmeans_kernel[(min(programs, max(num_segs, units)),)](
            key.u8,
            key.f32,
            key.page_stride,
            key.block_table,
            key.block_table.stride(0),
            *state.cen_args(),
            *state.cid_args(),
            req_t,
            n_keys_t,
            n_prev_t,
            tile_first_t,
            seg_first_t,
            cu_keys_t,
            c_f32,
            part,
            packed_cid,
            tile_counts,
            sync,
            num_full,
            config.num_clusters,
            config.kmeans_seed,
            iters,
            int(config.kmeans_stop_fraction * full_keys),
            C=C,
            # Two halves of >= 16 centroids each (tl.dot's minimum width).
            C_PAD=C_PAD,
            BLOCK_U=tune.kmeans_update_block,
            BLOCK_S=tune.kmeans_update_segments,
            SEGS=tune.kmeans_segments,
            MIN_SEG_TILES=tune.kmeans_min_segment_tiles,
            PAGE=page,
            TILE_PAGES=tile_pages,
            num_warps=tune.kmeans_num_warps,
        )
    if num_full < num_items:
        tile0 = int(tile_first[num_full])
        sums = torch.empty(
            (int(new_tile_first[-1]), C, 128), dtype=torch.float32, device=device
        )
        ivf_kmeans_append_kernel[(num_tiles - tile0,)](
            key.u8,
            key.f32,
            key.page_stride,
            key.block_table,
            key.block_table.stride(0),
            *state.cen_args(),
            *state.cid_args(),
            tile_item_t,
            tile_start_t,
            req_t,
            n_keys_t,
            n_prev_t,
            cu_keys_t,
            new_tile_first_t,
            packed_cid,
            tile_counts,
            sums,
            tile0,
            config.num_clusters,
            C=C,
            C_PAD=C_PAD,
            PAGE=page,
            num_warps=tune.kmeans_num_warps,
        )
        nb = triton.cdiv(C, tune.kmeans_update_block)
        ivf_kmeans_append_update_kernel[((num_items - num_full) * nb,)](
            *state.cen_args(),
            req_t,
            n_keys_t,
            new_tile_first_t,
            sums,
            num_full,
            config.num_clusters,
            C=C,
            BLOCK_U=tune.kmeans_update_block,
            BLOCK_S=tune.kmeans_update_segments,
        )

    # Stable counting sort of every request's keys into (request, cluster) runs.
    ivf_run_scan_kernel[(num_items, triton.cdiv(C, tune.scan_block_c))](
        tile_counts,
        tile_first_t,
        req_t,
        cu_keys_t,
        tile_off,
        run_start,
        run_size,
        scan_done,
        *state.cen_args(),
        cen_dense,
        cen_size,
        C=C,
        C_PAD=triton.next_power_of_2(C),
        BLOCK_C=tune.scan_block_c,
        TILES=tune.scan_tiles,
    )
    run_keys = torch.empty((total_keys, 128), dtype=torch.uint8, device=device)
    ivf_run_scatter_kernel[(num_tiles,)](
        key.u8,
        key.f32,
        key.page_stride,
        key.block_table,
        key.block_table.stride(0),
        tile_item_t,
        tile_start_t,
        req_t,
        n_keys_t,
        cu_keys_t,
        packed_cid,
        tile_off,
        run_start,
        run_pos,
        run_keys,
        run_scales,
        C=C,
        BLOCK_C=tune.centroid_block,
        PAGE=page,
    )
    return IvfRunLayout(
        run_pos=run_pos,
        run_start=run_start,
        run_size=run_size,
        run_keys=run_keys.view(torch.float8_e4m3fn),
        run_scales=run_scales,
        cen_dense=cen_dense,
        cen_size=cen_size,
    )


def append_ivf_decode_keys(
    *, key: IvfKeyView, state: IvfStateView, seq_lens: torch.Tensor
) -> None:
    """Assign each decode request's newest cached key to a fixed centroid.

    Decode rows are batch rows 0..len(seq_lens)-1.
    """
    num_rows = seq_lens.shape[0]
    if num_rows == 0:
        return
    ivf_decode_append_kernel[(num_rows,)](
        key.u8,
        key.page_stride,
        key.block_table,
        key.block_table.stride(0),
        *state.cen_args(),
        *state.cid_args(),
        seq_lens,
        C=state.num_clusters,
        BLOCK_C=TUNING.centroid_block,
        PAGE=key.page,
    )


def select_ivf_topk(
    *,
    key: IvfKeyView,
    state: IvfStateView,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    row_batch: torch.Tensor,
    query_pos: torch.Tensor,
    config: IvfIndexerConfig,
    out: torch.Tensor,
    topk_backend: str,
    layout: IvfRunLayout | None = None,
    seq_lens: torch.Tensor | None = None,
    max_scratch_bytes: int | None = None,
    max_width: int | None = None,
    token_lens: torch.Tensor | None = None,
) -> None:
    """Top-k over IVF candidates, written in place into ``out`` [rows, W].

    q_fp8 [rows, H, 128] fp8 e4m3 are the rotated indexer query heads and
    weights [rows, H] fp32 their head weights with the query scales folded in,
    as the stock indexer takes them. Row r is a query at position query_pos[r]
    of batch row row_batch[r]. Prefill passes ``layout``; decode passes
    ``seq_lens`` (rows are its batch rows).

    Candidates are every visible key of the probed clusters (at prefill, only
    the probed runs are searched for their visible prefix). Prefill scores
    every row in one pass into a buffer [rows, width]: ``width`` is
    ``max_width``, a host-known bound on any row's candidate count (such as
    its longest visible prefix), capped so the buffer fits
    ``max_scratch_bytes``; without ``max_width`` it reads back the largest
    count (a host sync). A row with more candidates than the width drops its
    lowest-scoring probed clusters. Decode uses the requests' position
    capacity as the width, so it can be captured.

    With ``token_lens`` (prefill: each row's visible token count) a row of
    ``out`` is the kpool indexer's expanded top-k instead: each selected
    pool's ``config.kpool`` token positions, then the tokens of the row's
    incomplete tail pool, then -1.
    """
    assert (layout is None) == (seq_lens is not None)
    assert token_lens is None or layout is not None
    num_rows = q_fp8.shape[0]
    if num_rows == 0:
        return
    num_heads = q_fp8.shape[1]
    assert q_fp8.shape[2] == 128 and q_fp8.dtype == torch.float8_e4m3fn
    assert q_fp8.is_contiguous() and weights.is_contiguous()
    assert weights.shape == (num_rows, num_heads) and weights.dtype == torch.float32
    device = q_fp8.device
    C = state.num_clusters
    topk = config.topk
    is_decode = layout is None
    tune = TUNING

    score_ws = torch.empty((num_rows, C), dtype=torch.float32, device=device)
    sel = torch.empty((num_rows, C), dtype=torch.int8, device=device)
    # A row seeing at most top-k keys probes every cluster with one of them.
    P_PAD = triton.next_power_of_2(max(min(max(config.num_probes, topk), C), 1))
    probe_rows = 0 if layout is None else num_rows
    vis_ws, ends, counts, totals, topk_lens, probe_end, probe_base = _alloc(
        torch.int32,
        device,
        (num_rows, C),
        (num_rows, C),
        (num_rows,),
        (num_rows,),
        (num_rows,),
        (probe_rows, P_PAD),
        (probe_rows, P_PAD),
    )
    if is_decode:
        width = max_count = key.block_table.shape[1] * key.page
        cand = torch.empty((num_rows, width), dtype=torch.int32, device=device)
    else:
        width, cand = 0, counts  # prefill emits from the run layout instead
        max_count = 2**31 - 1
        if max_scratch_bytes is not None:
            per_row = max_scratch_bytes // (num_rows * 4) - topk
            max_count = max(per_row, topk)
        if max_width is not None:
            max_count = min(max_count, max_width)

    run_args = (
        (counts, counts, counts)
        if is_decode
        else (layout.run_start, layout.run_pos, layout.run_size)
    )
    # One wgmma covers at most 256 query-head columns.
    assert tune.probe_rows * num_heads <= 256, "probe_rows * heads > 256"
    # Prefill: one program per row tile over every cluster (its queries load
    # once). Decode: few rows, so the clusters split across programs too.
    c_per_program = tune.centroid_block if is_decode else C
    probe_grid = (
        triton.cdiv(num_rows, tune.probe_rows),
        triton.cdiv(C, c_per_program),
    )
    cen_desc = cen_size = None
    if not is_decode:
        from triton.tools.tensor_descriptor import TensorDescriptor

        cen_desc = TensorDescriptor.from_tensor(
            layout.cen_dense, [tune.centroid_block, 128]
        )
        cen_size = layout.cen_size
    ivf_probe_score_kernel[probe_grid](
        q_fp8,
        weights,
        row_batch,
        query_pos,
        *state.cen_args(),
        *run_args,
        cen_desc,
        cen_size,
        score_ws,
        vis_ws,
        num_rows,
        c_per_program,
        C=C,
        H=num_heads,
        BLOCK_R=tune.probe_rows,
        BLOCK_C=tune.centroid_block,
        IS_DECODE=is_decode,
        num_warps=tune.probe_num_warps,
    )
    ivf_probe_select_kernel[(num_rows,)](
        query_pos,
        row_batch,
        *run_args[:2],
        score_ws,
        vis_ws,
        sel,
        ends,
        counts,
        totals,
        topk_lens,
        cand,
        width,
        probe_end,
        probe_base,
        config.num_probes,
        max_count,
        TOPK=topk,
        C=C,
        C_PAD=triton.next_power_of_2(C),
        P_PAD=P_PAD,
        IS_DECODE=is_decode,
        num_warps=tune.probe_num_warps,
    )
    if is_decode:
        _collect_by_scan(
            state=state, row_batch=row_batch, seq_lens=seq_lens, sel=sel, cand=cand
        )
        scores = torch.empty((num_rows, width), dtype=torch.float32, device=device)
        _score_per_row(
            key=key,
            q_fp8=q_fp8,
            weights=weights,
            row_batch=row_batch,
            cand=cand,
            counts=counts,
            scores=scores,
            topk=topk,
        )
        _topk_and_emit(
            counts=counts,
            topk_lens=topk_lens,
            scores=scores,
            cand=cand,
            out=out,
            topk=topk,
            topk_backend=topk_backend,
        )
        return

    width = int(counts.max()) if max_width is None else max_count
    scores = torch.empty((num_rows, width), dtype=torch.float32, device=device)
    _score_grouped_gemm(
        q_fp8=q_fp8,
        weights=weights,
        row_batch=row_batch,
        layout=layout,
        sel=sel,
        vis_ws=vis_ws,
        ends=ends,
        topk_lens=topk_lens,
        scores=scores,
        num_clusters=C,
    )
    _topk_and_emit(
        counts=counts,
        topk_lens=topk_lens,
        scores=scores,
        out=out,
        topk=topk,
        topk_backend=topk_backend,
        runs=_EmitRuns(
            totals=totals,
            query_pos=query_pos,
            probe_end=probe_end,
            probe_base=probe_base,
            run_pos=layout.run_pos,
            token_lens=token_lens,
            kpool=config.kpool if token_lens is not None else 1,
        ),
    )


@dataclass(frozen=True)
class _EmitRuns:
    """What the prefill emit needs to map candidate slots to positions."""

    totals: torch.Tensor
    query_pos: torch.Tensor
    probe_end: torch.Tensor
    probe_base: torch.Tensor
    run_pos: torch.Tensor
    # Visible tokens per row: expand pools to kpool tokens plus the tail.
    token_lens: torch.Tensor | None
    kpool: int


def _topk_and_emit(
    *,
    counts: torch.Tensor,
    topk_lens: torch.Tensor,
    scores: torch.Tensor,
    out: torch.Tensor,
    topk: int,
    topk_backend: str,
    cand: torch.Tensor | None = None,
    runs: _EmitRuns | None = None,
) -> None:
    """Top-k slots of over-budget rows, then candidate positions into ``out``.

    Rows with at most top-k candidates have length 0 and keep all of them.
    Decode reads positions from ``cand``; prefill maps the selected slots
    through the run layout (``runs``), so it never materializes candidates.
    """
    assert (cand is None) != (runs is None)
    num_rows, width = scores.shape
    selected = torch.empty((num_rows, topk), dtype=torch.int32, device=out.device)
    get_indexer_topk(topk_backend)(scores, topk_lens, 1, selected, topk, width)
    block = TUNING.emit_block
    if runs is not None:
        has_tail = runs.token_lens is not None
        tail = max(out.shape[1] - topk * runs.kpool, 1)
        ivf_emit_runs_kernel[(num_rows, triton.cdiv(topk, block))](
            counts,
            runs.totals,
            selected,
            runs.query_pos,
            runs.token_lens if has_tail else counts,
            runs.probe_end,
            runs.probe_base,
            runs.run_pos,
            out,
            out.stride(0),
            out.shape[1],
            TOPK=topk,
            KPOOL=runs.kpool,
            P_PAD=runs.probe_end.shape[1],
            LOG_P=runs.probe_end.shape[1].bit_length(),
            BLOCK=block,
            TAIL_BLOCK=triton.next_power_of_2(tail),
            HAS_TAIL=has_tail,
        )
        return
    grid = (num_rows, triton.cdiv(out.shape[1], block))
    ivf_emit_kernel[grid](
        counts,
        selected,
        cand,
        out,
        out.stride(0),
        width,
        out.shape[1],
        TOPK=topk,
        BLOCK=block,
    )


def _score_per_row(
    *,
    key: IvfKeyView,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    row_batch: torch.Tensor,
    cand: torch.Tensor,
    counts: torch.Tensor,
    scores: torch.Tensor,
    topk: int,
) -> None:
    tune = TUNING
    num_rows = q_fp8.shape[0]
    cap = cand.shape[1]
    tiles = triton.cdiv(cap, tune.score_block_n)
    target = tune.score_programs_per_sm * _num_sms(q_fp8.device.index or 0)
    splits = max(1, min(tiles, triton.cdiv(target, num_rows)))
    ivf_score_kernel[(num_rows, splits)](
        q_fp8,
        weights,
        row_batch,
        cand,
        counts,
        key.u8,
        key.f32,
        key.page_stride,
        key.block_table,
        key.block_table.stride(0),
        scores,
        cap,
        TOPK=topk,
        H=q_fp8.shape[1],
        PAGE=key.page,
        BLOCK_N=tune.score_block_n,
        NUM_SPLITS=splits,
        NUM_STAGES=tune.score_num_stages,
        num_warps=tune.score_num_warps,
    )


def _score_grouped_gemm(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    row_batch: torch.Tensor,
    layout: IvfRunLayout,
    sel: torch.Tensor,
    vis_ws: torch.Tensor,
    ends: torch.Tensor,
    topk_lens: torch.Tensor,
    scores: torch.Tensor,
    num_clusters: int,
) -> None:
    """Prefill scoring as one grouped GEMM over (batch row, cluster) groups.

    Groups, their row lists and the work list are built on the device; the
    persistent kernel reads the work total from memory, so nothing syncs.
    """
    from triton.tools.tensor_descriptor import TensorDescriptor

    tune = TUNING
    num_rows = q_fp8.shape[0]
    C = num_clusters
    device = q_fp8.device
    num_groups = layout.run_start.shape[0] * C

    row_tiles = triton.cdiv(num_rows, tune.plan_block_r)
    group_m, group_vis = _alloc(
        torch.int32, device, (num_groups,), (num_groups,), zero=True
    )
    (
        group_off,
        unit_off,
        row_list,
        tile_cnt,
        tile_base,
        cluster_total,
        cluster_base,
    ) = _alloc(
        torch.int32,
        device,
        (num_groups,),
        (num_groups + 2,),
        (num_rows * C,),
        (row_tiles, C),
        (row_tiles, C),
        (C,),
        (C,),
    )
    plan_grid = (row_tiles, triton.cdiv(C, tune.plan_block_c))
    ivf_group_count_kernel[plan_grid](
        sel,
        vis_ws,
        topk_lens,
        row_batch,
        group_m,
        group_vis,
        tile_cnt,
        num_rows,
        C=C,
        BLOCK_R=tune.plan_block_r,
        BLOCK_C=tune.plan_block_c,
    )
    ivf_group_scan_kernel[(triton.cdiv(C, tune.plan_block_c),)](
        tile_cnt,
        tile_base,
        cluster_total,
        row_tiles,
        C=C,
        BLOCK_T=tune.plan_scan_tiles,
        BLOCK_C=tune.plan_block_c,
    )
    rows = max(1, tune.gemm_mma_cols // q_fp8.shape[1])
    chunked = layout.run_keys.shape[0] >= tune.gemm_chunked_keys * C
    key_chunks = tune.gemm_key_chunks if chunked else 1
    ivf_group_offsets_kernel[(1,)](
        group_m,
        group_vis,
        cluster_total,
        cluster_base,
        unit_off,
        num_groups,
        C=C,
        UNIT_KEYS=tune.gemm_block_n * key_chunks,
        UNIT_ROWS=rows * tune.gemm_row_tiles,
        BLOCK=1024,
    )
    ivf_group_fill_kernel[plan_grid](
        sel,
        topk_lens,
        row_batch,
        tile_base,
        cluster_base,
        row_list,
        group_off,
        num_rows,
        C=C,
        BLOCK_R=tune.plan_block_r,
        BLOCK_C=tune.plan_block_c,
    )
    per_sm = tune.gemm_chunked_programs_per_sm if chunked else tune.gemm_programs_per_sm
    num_programs = per_sm * _num_sms(device.index or 0)
    key_desc = TensorDescriptor.from_tensor(layout.run_keys, [tune.gemm_block_n, 128])
    ivf_grouped_score_kernel[(num_programs,)](
        q_fp8,
        weights,
        key_desc,
        layout.run_scales,
        layout.run_start,
        group_m,
        group_vis,
        group_off,
        unit_off,
        row_list,
        vis_ws,
        ends,
        scores,
        scores.shape[1],
        num_groups,
        num_groups.bit_length(),
        C=C,
        H=q_fp8.shape[1],
        BLOCK_N=tune.gemm_block_n,
        KEY_CHUNKS=key_chunks,
        ROWS=rows,
        ROW_TILES=tune.gemm_row_tiles,
        NUM_STAGES=tune.gemm_num_stages,
        num_warps=tune.gemm_num_warps,
    )


def _collect_by_scan(
    *,
    state: IvfStateView,
    row_batch: torch.Tensor,
    seq_lens: torch.Tensor,
    sel: torch.Tensor,
    cand: torch.Tensor,
) -> None:
    num_rows = seq_lens.shape[0]
    block = TUNING.scan_block
    num_blocks = triton.cdiv(state.cid_bt.shape[1] * state.cid_bs, block)
    block_counts = torch.empty(
        (num_rows, num_blocks), dtype=torch.int32, device=cand.device
    )
    ivf_scan_count_kernel[(num_rows, num_blocks)](
        row_batch,
        seq_lens,
        sel,
        *state.cid_args(),
        block_counts,
        num_blocks,
        C=state.num_clusters,
        BLOCK=block,
    )
    ivf_scan_write_kernel[(num_rows, num_blocks)](
        row_batch,
        seq_lens,
        sel,
        *state.cid_args(),
        block_counts,
        num_blocks,
        cand,
        cand.shape[1],
        C=state.num_clusters,
        BLOCK=block,
        NUM_BLOCKS_PAD=triton.next_power_of_2(num_blocks),
    )


def _select_scratch_bytes(config: IvfIndexerConfig) -> int:
    """Prefill candidate scratch the profiler reserves: every scheduled row with
    a quarter of the longest context's pools as candidates (probing an eighth
    of the clusters finds at most ~n/4 keys on real prompts). A row with more
    candidates than fit drops its lowest-scoring probes."""
    cfg = get_current_vllm_config_or_none()
    if cfg is None:
        return 1 << 30
    max_tokens = cfg.scheduler_config.max_num_batched_tokens
    quarter = triton.cdiv(cfg.model_config.max_model_len // config.kpool, 4)
    return max_tokens * (quarter * 4 + config.topk * 4)


def _reserve_profile_memory(config: IvfIndexerConfig, device: torch.device) -> None:
    """Make the profiler see the IVF workspaces' worst case."""
    current_workspace_manager().get_simultaneous(
        ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
    )
    cfg = get_current_vllm_config_or_none()
    max_reqs = cfg.scheduler_config.max_num_seqs if cfg is not None else 1
    max_tokens = cfg.scheduler_config.max_num_batched_tokens if cfg is not None else 1
    max_len = cfg.model_config.max_model_len // config.kpool if cfg is not None else 0
    C = config.max_clusters
    # Probe tables (prefill lists up to every cluster per row), the selected
    # pools and their expansion to tokens.
    probe_bytes = max_tokens * (C * 25 + 16 + config.topk * 4)
    probe_bytes += max_tokens * (config.topk + 1) * config.kpool * 4
    prefill_bytes = _select_scratch_bytes(config)
    # Decode rows use the full pool capacity as their candidate width.
    decode_bytes = max_reqs * (max_len * 8 + config.topk * 4)
    # Per centroid: the fp32 master.
    build_bytes = max_reqs * C * 128 * 4 + max_tokens * 16
    # Run layout of one maximal context: packed keys, scales, positions and
    # cluster ids (144 B/key), the per-tile count/offset tables and the
    # k-means segment sums. A tile is a kernel page, at least the smallest one.
    min_page = min(PAGED_MQA_PAGE_SIZES)
    tiles = triton.cdiv(max_len, min_page)
    segs = triton.cdiv(tiles, TUNING.kmeans_min_segment_tiles)
    build_bytes += max_len * 144 + tiles * C * 8 + segs * C * 128 * 4
    # An append's new-key sums: one [C, 128] fp32 row per page of new keys.
    new_pages = triton.cdiv(max_tokens // config.kpool, min_page)
    build_bytes += (new_pages + max_reqs) * C * 128 * 4
    total = probe_bytes + max(prefill_bytes, decode_bytes) + build_bytes
    _ = torch.empty(total, dtype=torch.uint8, device=device)


def _prefill_rows(
    seq_lens: list[int],
    query_lens: list[int],
    first_row: int,
    kpool: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per prefill query row, from host lengths in one copy: its batch row,
    its last complete pool (-1: none yet) and its visible token count."""
    lens = np.asarray(query_lens)
    num_rows = int(lens.sum())
    host = torch.empty((3, num_rows), dtype=torch.int32, pin_memory=True)
    rows = host.numpy()
    rows[0] = np.repeat(np.arange(first_row, first_row + len(lens)), lens)
    # A row's visible tokens: its request's context before the chunk, plus
    # its offset in the chunk, plus itself.
    starts = np.cumsum(lens) - lens
    rows[2] = np.arange(1, num_rows + 1) + np.repeat(
        np.asarray(seq_lens) - lens - starts, lens
    )
    rows[1] = rows[2] // kpool - 1
    row_batch, pool_pos, token_lens = host.to(device, non_blocking=True)
    return row_batch, pool_pos, token_lens


@eager_break_during_capture
def sparse_attn_indexer_ivf(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    tail_prefix: str,
    cid_prefix: str,
    cen_prefix: str,
    key_cache: torch.Tensor,
    tail_cache: torch.Tensor,
    cid_cache: torch.Tensor,
    cen_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    k: torch.Tensor,
    gate_score: torch.Tensor,
    compress_ape: torch.Tensor,
    positions: torch.Tensor,
    config: IvfIndexerConfig,
    topk_indices_buffer: torch.Tensor,
    topk_backend: str,
    max_scratch_bytes: int,
) -> torch.Tensor:
    attn_metadata = get_forward_context().attn_metadata
    if not isinstance(attn_metadata, dict):
        _reserve_profile_memory(config, hidden_states.device)
        return topk_indices_buffer
    meta = attn_metadata[k_cache_prefix]
    assert isinstance(meta, IvfIndexerMetadata)
    tail_meta = attn_metadata[tail_prefix]
    cid_meta = attn_metadata[cid_prefix]
    cen_meta = attn_metadata[cen_prefix]
    assert isinstance(cid_meta, IvfStateMetadata)
    assert isinstance(cen_meta, IvfStateMetadata)

    kpool = config.kpool
    key = IvfKeyView.of(key_cache, meta.block_table)
    state = IvfStateView.of(
        cid_cache,
        cen_cache,
        cid_meta.block_table,
        cen_meta.block_table,
        config.max_clusters,
    )
    num_tokens = meta.slot_mapping.shape[0]
    k = k[:num_tokens]
    gate_score = gate_score[:num_tokens]
    head_dim = k.shape[-1]
    nd = meta.num_decodes
    # The kpool indexer's writes: a prefill caches its complete pools and its
    # incomplete tail pool; a decode token (one per request) joins the tail
    # and completes a pool every kpool tokens.
    if num_tokens > nd:
        prefill = slice(nd, num_tokens)
        _kpool_compress_insert(
            k[prefill],
            gate_score[prefill],
            compress_ape,
            key_cache,
            meta.slot_mapping[prefill],
            kpool,
            head_dim,
            round_scale=True,
        )
        kpool_ops.kpool_seed_tail_cache(
            tail_cache,
            k[prefill],
            gate_score[prefill],
            tail_meta.slot_mapping[prefill],
            kpool,
            head_dim,
        )
    num_pools_cpu = [n // kpool for n in meta.prefill_seq_lens_cpu]
    # Requests with a complete pool; the others select only their tail.
    built = [i for i, n in enumerate(num_pools_cpu) if n > 0]
    start = meta.num_decode_tokens
    end = start + meta.num_prefill_tokens
    if built:
        # The prefill emit writes its rows whole.
        topk_indices_buffer[:start] = -1
        topk_indices_buffer[end : hidden_states.shape[0]] = -1
    else:
        topk_indices_buffer[: hidden_states.shape[0]] = -1

    if nd > 0:
        assert meta.num_decode_tokens == nd, "IVF decode takes one token per request"
        kpool_ops.kpool_decode_update_and_maybe_write_cache_batched(
            key_cache,
            tail_cache,
            tail_meta.slot_mapping[:nd].view(nd, 1),
            k[:nd].view(nd, 1, head_dim),
            gate_score[:nd].view(nd, 1, head_dim),
            compress_ape,
            meta.slot_mapping[:nd].view(nd, 1),
            positions[:nd].to(torch.int32).view(nd, 1),
            kpool,
            head_dim,
            round_scale=True,
        )
        seq_lens = meta.seq_lens[:nd]
        num_pools = seq_lens // kpool
        # Only a step that completes a pool appends it.
        completed = torch.where(seq_lens % kpool == 0, num_pools, 0)
        append_ivf_decode_keys(key=key, state=state, seq_lens=completed)
        pools = torch.empty((nd, config.topk), dtype=torch.int32, device=q_fp8.device)
        select_ivf_topk(
            key=key,
            state=state,
            q_fp8=q_fp8[:nd],
            weights=weights[:nd],
            row_batch=meta.req_index[:nd],
            query_pos=num_pools - 1,
            config=config,
            out=pools,
            topk_backend=topk_backend,
            seq_lens=num_pools,
        )
        tokens = kpool_ops.expand_pools_and_append_tail(pools, seq_lens, kpool)
        topk_indices_buffer[:nd, : tokens.shape[1]] = tokens

    if meta.num_prefills > 0:
        device = q_fp8.device
        if built:
            layout = build_ivf_index(
                key=key,
                state=state,
                batch_rows_cpu=[nd + i for i in built],
                n_keys_cpu=[num_pools_cpu[i] for i in built],
                num_batch_rows=meta.num_reqs,
                config=config,
                n_prev_keys_cpu=[
                    (meta.prefill_seq_lens_cpu[i] - meta.prefill_query_lens_cpu[i])
                    // kpool
                    for i in built
                ],
                final_cpu=[meta.prefill_final_cpu[i] for i in built],
            )
        num_rows = end - start
        row_batch, pool_pos, token_lens = _prefill_rows(
            meta.prefill_seq_lens_cpu,
            meta.prefill_query_lens_cpu,
            nd,
            kpool,
            device,
        )
        if built:
            select_ivf_topk(
                key=key,
                state=state,
                q_fp8=q_fp8[start:end],
                weights=weights[start:end],
                row_batch=row_batch,
                query_pos=pool_pos,
                config=config,
                out=topk_indices_buffer[start:end],
                topk_backend=topk_backend,
                layout=layout,
                max_scratch_bytes=max_scratch_bytes,
                # A query sees at most its request's pools: no width read-back.
                max_width=max(num_pools_cpu),
                token_lens=token_lens,
            )
        else:
            pools = torch.full(
                (num_rows, config.topk), -1, dtype=torch.int32, device=device
            )
            tokens = kpool_ops.expand_pools_and_append_tail(pools, token_lens, kpool)
            topk_indices_buffer[start:end, : tokens.shape[1]] = tokens
    return topk_indices_buffer


@CustomOp.register("sparse_attn_indexer_ivf")
class SparseAttnIndexerIVF(CustomOp):
    """IVF (k-means) replacement for ``SparseAttnIndexerKpool`` (GLM-5.3-Flash)."""

    def __init__(
        self,
        k_cache,
        tail_cache,
        cid_cache,
        cen_cache,
        config: IvfIndexerConfig,
        topk_indices_buffer: torch.Tensor,
    ):
        super().__init__()
        self.k_cache = k_cache
        self.tail_cache = tail_cache
        self.cid_cache = cid_cache
        self.cen_cache = cen_cache
        self.config = config
        self.topk_indices_buffer = topk_indices_buffer
        cfg = get_current_vllm_config_or_none()
        self.topk_backend = (
            cfg.kernel_config.sparse_indexer_topk_backend if cfg is not None else "auto"
        )
        self.max_scratch_bytes = _select_scratch_bytes(config)

    def forward_native(self, *args, **kwargs):
        return self.forward_cuda(*args, **kwargs)

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        k: torch.Tensor,
        gate_score: torch.Tensor,
        compress_ape: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return sparse_attn_indexer_ivf(
            hidden_states,
            self.k_cache.prefix,
            self.tail_cache.prefix,
            self.cid_cache.prefix,
            self.cen_cache.prefix,
            self.k_cache.kv_cache,
            self.tail_cache.kv_cache,
            self.cid_cache.kv_cache,
            self.cen_cache.kv_cache,
            q_fp8,
            weights,
            k,
            gate_score,
            compress_ape,
            positions,
            self.config,
            self.topk_indices_buffer,
            self.topk_backend,
            self.max_scratch_bytes,
        )
