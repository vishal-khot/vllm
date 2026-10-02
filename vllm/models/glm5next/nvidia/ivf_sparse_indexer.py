# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""IVF (k-means) top-k selection for the glm5next DSA indexer.

The algorithm of ``ivf-dsa-kmeans/compare_decode_kmeans.py``: the (single-head,
MQA) keys are clustered by cosine k-means per request and per layer into a
fixed number of clusters. A query keeps its H fp8 indexer heads q_h and head
weights w_h (query scales folded in, as for the stock indexer). It probes the
fixed number of clusters with the best sum_h w_h ReLU(cos(q_h, c)), scores
every visible key of them by the DSA score sum_h w_h ReLU(q_h . k), and
selects the top-k.

Multi-token steps (prefill and its chunks) rebuild the request's index over
prefix + chunk; one-token steps assign the new key to the nearest fixed
centroid. Decode append and select take no host data, so the decode path is
safe under CUDA graph capture.
"""

import functools
from dataclasses import dataclass

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
from vllm.models.glm5next.nvidia.ops.ivf_index import (
    REC_BYTES,
    ivf_decode_append_kernel,
    ivf_emit_kernel,
    ivf_emit_runs_kernel,
    ivf_group_count_kernel,
    ivf_group_fill_kernel,
    ivf_group_offsets_kernel,
    ivf_grouped_score_kernel,
    ivf_kmeans_kernel,
    ivf_probe_score_kernel,
    ivf_probe_select_kernel,
    ivf_run_scan_kernel,
    ivf_run_scatter_kernel,
    ivf_scan_count_kernel,
    ivf_scan_write_kernel,
    ivf_score_kernel,
    ivf_write_keys_kernel,
)
from vllm.triton_utils import triton
from vllm.v1.worker.workspace import current_workspace_manager


@dataclass
class IvfKernelTuning:
    """Launch parameters. Unmeasured starting points; sweep on the target GPU."""

    centroid_block: int = 64
    score_block_n: int = 128
    score_num_warps: int = 4
    score_num_stages: int = 3
    # Score programs per SM to aim for; a row splits its tiles until reached.
    score_programs_per_sm: int = 4
    scan_block: int = 2048
    # Key tiles per step of the run-layout scan.
    scan_tiles: int = 16
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
    write_rows: int = 32
    # Prefill grouped GEMM: BLOCK_M keys of one cluster against tiles of
    # gemm_rows query rows (gemm_rows x heads MMA columns); a work unit
    # streams gemm_tiles tiles.
    gemm_block_m: int = 64
    gemm_rows: int = 8
    gemm_tiles: int = 8
    gemm_num_warps: int = 4
    gemm_num_stages: int = 3
    gemm_programs_per_sm: int = 2
    plan_block_r: int = 64
    plan_block_c: int = 64


TUNING = IvfKernelTuning()


@functools.cache
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@dataclass(frozen=True)
class IvfKeyView:
    """The per-token key cache of one layer, as ``[pages, PAGE, 132]`` bytes."""

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


def write_ivf_keys(key: IvfKeyView, k: torch.Tensor, slot_mapping: torch.Tensor):
    """FWHT-rotate, fp8-quantize and cache ``k`` (bf16 ``[T, 128]``) by slot."""
    n = k.shape[0]
    if n == 0:
        return
    rows = TUNING.write_rows
    ivf_write_keys_kernel[(triton.cdiv(n, rows),)](
        k.contiguous(),
        slot_mapping,
        key.u8,
        key.f32,
        key.page_stride,
        n,
        PAGE=key.page,
        BLOCK_R=rows,
        num_warps=2,
    )


def build_ivf_index(
    *,
    key: IvfKeyView,
    state: IvfStateView,
    batch_rows_cpu: list[int],
    n_keys_cpu: list[int],
    num_batch_rows: int,
    config: IvfIndexerConfig,
    n_prev_keys_cpu: list[int] | None = None,
) -> IvfRunLayout:
    """Cluster every listed request's cached keys, replacing its IVF state.

    ``n_prev_keys_cpu`` are the keys each request had before this step (0 for
    a request's first step): its previous chunk's centroids seed k-means.
    """
    if n_prev_keys_cpu is None:
        n_prev_keys_cpu = [0] * len(n_keys_cpu)
    device = key.u8.device
    C = state.num_clusters
    page = key.page
    num_items = len(n_keys_cpu)
    tune = TUNING
    tile_item, tile_start, tile_first, seg_first = [], [], [0], [0]
    for i, n in enumerate(n_keys_cpu):
        starts = range(0, n, page)
        tile_item.extend([i] * len(starts))
        tile_start.extend(starts)
        tile_first.append(len(tile_item))
        seg_tiles = max(
            triton.cdiv(len(starts), tune.kmeans_segments),
            tune.kmeans_min_segment_tiles,
        )
        seg_first.append(seg_first[-1] + triton.cdiv(len(starts), seg_tiles))
    cu_keys_cpu = [0]
    for n in n_keys_cpu:
        cu_keys_cpu.append(cu_keys_cpu[-1] + n)
    num_tiles, total_keys = len(tile_item), cu_keys_cpu[-1]
    host = torch.tensor(
        tile_item
        + tile_start
        + list(batch_rows_cpu)
        + list(n_keys_cpu)
        + list(n_prev_keys_cpu)
        + cu_keys_cpu[:-1]
        + tile_first
        + seg_first,
        dtype=torch.int32,
        pin_memory=True,
    ).to(device, non_blocking=True)
    (
        tile_item_t,
        tile_start_t,
        req_t,
        n_keys_t,
        n_prev_t,
        cu_keys_t,
        tile_first_t,
        seg_first_t,
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
        ]
    )

    num_segs = seg_first[-1]
    c_f32 = torch.empty((num_items, C, 128), dtype=torch.float32, device=device)
    part = torch.empty((num_segs, C, 128), dtype=torch.float32, device=device)
    packed_cid = torch.empty((total_keys,), dtype=torch.int32, device=device)
    tile_counts = torch.empty((num_tiles, C), dtype=torch.int32, device=device)
    num_phases = 2 * config.kmeans_iters + 2
    sync = torch.zeros(num_phases * (num_items + 3), dtype=torch.int32, device=device)
    units = num_items * triton.cdiv(C, tune.kmeans_update_block)
    programs = tune.kmeans_programs_per_sm * _num_sms(device.index or 0)
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
        num_items,
        config.num_clusters,
        config.kmeans_seed,
        config.kmeans_iters,
        int(config.kmeans_stop_fraction * total_keys),
        C=C,
        # Two halves of >= 16 centroids each (tl.dot's minimum width).
        C_PAD=max(triton.next_power_of_2(C), 32),
        BLOCK_U=tune.kmeans_update_block,
        BLOCK_S=tune.kmeans_update_segments,
        SEGS=tune.kmeans_segments,
        MIN_SEG_TILES=tune.kmeans_min_segment_tiles,
        PAGE=page,
        num_warps=tune.kmeans_num_warps,
    )

    # Stable counting sort of every request's keys into (request, cluster) runs.
    run_size = torch.zeros((num_batch_rows, C), dtype=torch.int32, device=device)
    run_start = torch.zeros((num_batch_rows, C), dtype=torch.int32, device=device)
    tile_off = torch.empty((num_tiles, C), dtype=torch.int32, device=device)
    ivf_run_scan_kernel[(num_items,)](
        tile_counts,
        tile_first_t,
        req_t,
        cu_keys_t,
        tile_off,
        run_start,
        run_size,
        C=C,
        C_PAD=triton.next_power_of_2(C),
        TILES=TUNING.scan_tiles,
    )
    run_pos = torch.empty((total_keys,), dtype=torch.int32, device=device)
    run_keys = torch.empty((total_keys, 128), dtype=torch.uint8, device=device)
    run_scales = torch.empty((total_keys,), dtype=torch.float32, device=device)
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
) -> None:
    """Top-k over IVF candidates, written in place into ``out`` [rows, W].

    q_fp8 [rows, H, 128] fp8 e4m3 are the rotated indexer query heads and
    weights [rows, H] fp32 their head weights with the query scales folded in,
    as the stock indexer takes them. Row r is a query at position query_pos[r]
    of batch row row_batch[r]. Prefill passes ``layout``; decode passes
    ``seq_lens`` (rows are its batch rows).

    Candidates are every visible key of the probed clusters, uncapped (at
    prefill, only the probed runs are searched for their visible prefix). Prefill
    sizes its score buffer by reading back the largest count (its one
    host sync; prefill runs eagerly) and, only if [rows, width] scratch would
    exceed ``max_scratch_bytes``, splits rows into passes that fit. Decode
    uses the requests' position capacity as the width, so it can be captured.
    """
    assert (layout is None) == (seq_lens is not None)
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
    vis_ws = torch.empty((num_rows, C), dtype=torch.int32, device=device)
    sel = torch.empty((num_rows, C), dtype=torch.int8, device=device)
    ends = torch.empty((num_rows, C), dtype=torch.int32, device=device)
    counts = torch.empty((num_rows,), dtype=torch.int32, device=device)
    totals = torch.empty((num_rows,), dtype=torch.int32, device=device)
    topk_lens = torch.empty((num_rows,), dtype=torch.int32, device=device)
    if is_decode:
        width = key.block_table.shape[1] * key.page
        cand = torch.empty((num_rows, width), dtype=torch.int32, device=device)
    else:
        width, cand = 0, counts  # prefill emits from the run layout instead

    run_args = (
        (counts, counts, counts)
        if is_decode
        else (layout.run_start, layout.run_pos, layout.run_size)
    )
    probe_grid = (
        triton.cdiv(num_rows, tune.probe_rows),
        triton.cdiv(C, tune.centroid_block),
    )
    ivf_probe_score_kernel[probe_grid](
        q_fp8,
        weights,
        row_batch,
        query_pos,
        *state.cen_args(),
        *run_args,
        score_ws,
        vis_ws,
        num_rows,
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
        config.num_probes,
        TOPK=topk,
        C=C,
        C_PAD=triton.next_power_of_2(C),
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

    width = int(counts.max())
    per_row = width * 4 + topk * 4
    rows_per_pass = num_rows
    if max_scratch_bytes is not None:
        rows_per_pass = max(1, min(num_rows, max_scratch_bytes // per_row))
    for start in range(0, num_rows, rows_per_pass):
        rows = slice(start, min(start + rows_per_pass, num_rows))
        n = rows.stop - rows.start
        scores = torch.empty((n, width), dtype=torch.float32, device=device)
        _score_grouped_gemm(
            q_fp8=q_fp8[rows],
            weights=weights[rows],
            row_batch=row_batch[rows],
            layout=layout,
            sel=sel[rows],
            vis_ws=vis_ws[rows],
            ends=ends[rows],
            topk_lens=topk_lens[rows],
            scores=scores,
            num_clusters=C,
        )
        _topk_and_emit(
            counts=counts[rows],
            topk_lens=topk_lens[rows],
            scores=scores,
            out=out[rows],
            topk=topk,
            topk_backend=topk_backend,
            runs=_EmitRuns(
                totals=totals[rows],
                row_batch=row_batch[rows],
                query_pos=query_pos[rows],
                ends=ends[rows],
                layout=layout,
            ),
        )


@dataclass(frozen=True)
class _EmitRuns:
    """What the prefill emit needs to map candidate slots to positions."""

    totals: torch.Tensor
    row_batch: torch.Tensor
    query_pos: torch.Tensor
    ends: torch.Tensor
    layout: IvfRunLayout


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
    grid = (num_rows, triton.cdiv(out.shape[1], block))
    if runs is not None:
        C = runs.ends.shape[1]
        ivf_emit_runs_kernel[grid](
            counts,
            runs.totals,
            selected,
            runs.row_batch,
            runs.query_pos,
            runs.ends,
            runs.layout.run_start,
            runs.layout.run_pos,
            out,
            out.stride(0),
            out.shape[1],
            TOPK=topk,
            C=C,
            LOG_C=C.bit_length(),
            BLOCK=block,
        )
        return
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
    tune = TUNING
    num_rows = q_fp8.shape[0]
    C = num_clusters
    device = q_fp8.device
    num_groups = layout.run_start.shape[0] * C
    rows_per_tile = tune.gemm_rows
    unit_rows = rows_per_tile * tune.gemm_tiles

    group_m = torch.zeros((num_groups,), dtype=torch.int32, device=device)
    group_vis = torch.zeros((num_groups,), dtype=torch.int32, device=device)
    group_off = torch.empty((num_groups,), dtype=torch.int32, device=device)
    cursor = torch.empty((num_groups,), dtype=torch.int32, device=device)
    unit_off = torch.empty((num_groups + 1,), dtype=torch.int32, device=device)
    row_list = torch.empty((num_rows * C,), dtype=torch.int32, device=device)
    plan_grid = (
        triton.cdiv(num_rows, tune.plan_block_r),
        triton.cdiv(C, tune.plan_block_c),
    )
    ivf_group_count_kernel[plan_grid](
        sel,
        vis_ws,
        topk_lens,
        row_batch,
        group_m,
        group_vis,
        num_rows,
        C=C,
        BLOCK_R=tune.plan_block_r,
        BLOCK_C=tune.plan_block_c,
    )
    ivf_group_offsets_kernel[(1,)](
        group_m,
        group_vis,
        group_off,
        cursor,
        unit_off,
        num_groups,
        BLOCK_M=tune.gemm_block_m,
        UNIT_ROWS=unit_rows,
        BLOCK=1024,
    )
    ivf_group_fill_kernel[plan_grid](
        sel,
        topk_lens,
        row_batch,
        cursor,
        row_list,
        num_rows,
        C=C,
        BLOCK_R=tune.plan_block_r,
        BLOCK_C=tune.plan_block_c,
    )
    num_programs = tune.gemm_programs_per_sm * _num_sms(device.index or 0)
    ivf_grouped_score_kernel[(num_programs,)](
        q_fp8,
        weights,
        layout.run_keys,
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
        C=C,
        H=q_fp8.shape[1],
        BLOCK_M=tune.gemm_block_m,
        ROWS=rows_per_tile,
        TILES=tune.gemm_tiles,
        LOG_G=num_groups.bit_length(),
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
    a quarter of the longest context as candidates (P = C/4 probes see ~n/4
    keys). ``select_ivf_topk`` splits rows into passes beyond it."""
    cfg = get_current_vllm_config_or_none()
    if cfg is None:
        return 1 << 30
    max_tokens = cfg.scheduler_config.max_num_batched_tokens
    quarter = triton.cdiv(cfg.model_config.max_model_len, 4)
    return max_tokens * (quarter * 4 + config.topk * 4)


def _reserve_profile_memory(config: IvfIndexerConfig, device: torch.device) -> None:
    """Make the profiler see the IVF workspaces' worst case."""
    current_workspace_manager().get_simultaneous(
        ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
    )
    cfg = get_current_vllm_config_or_none()
    max_reqs = cfg.scheduler_config.max_num_seqs if cfg is not None else 1
    max_tokens = cfg.scheduler_config.max_num_batched_tokens if cfg is not None else 1
    max_len = cfg.model_config.max_model_len if cfg is not None else 0
    C = config.max_clusters
    probe_bytes = max_tokens * (C * 17 + 16)
    prefill_bytes = _select_scratch_bytes(config)
    # Decode rows use the full position capacity as their candidate width.
    decode_bytes = max_reqs * (max_len * 8 + config.topk * 4)
    # Per centroid: the fp32 master.
    build_bytes = max_reqs * C * 128 * 4 + max_tokens * 16
    # Run layout of one maximal context: packed keys, scales, positions and
    # cluster ids (144 B/key), the per-tile count/offset tables and the
    # k-means segment sums.
    tiles = triton.cdiv(max_len, 64)
    segs = triton.cdiv(tiles, TUNING.kmeans_min_segment_tiles)
    build_bytes += max_len * 144 + tiles * C * 8 + segs * C * 128 * 4
    total = probe_bytes + max(prefill_bytes, decode_bytes) + build_bytes
    _ = torch.empty(total, dtype=torch.uint8, device=device)


@eager_break_during_capture
def sparse_attn_indexer_ivf(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    cid_prefix: str,
    cen_prefix: str,
    key_cache: torch.Tensor,
    cid_cache: torch.Tensor,
    cen_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    k: torch.Tensor,
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
    cid_meta = attn_metadata[cid_prefix]
    cen_meta = attn_metadata[cen_prefix]
    assert isinstance(cid_meta, IvfStateMetadata)
    assert isinstance(cen_meta, IvfStateMetadata)

    key = IvfKeyView.of(key_cache, meta.block_table)
    state = IvfStateView.of(
        cid_cache,
        cen_cache,
        cid_meta.block_table,
        cen_meta.block_table,
        config.max_clusters,
    )
    num_tokens = meta.slot_mapping.shape[0]
    write_ivf_keys(key, k[:num_tokens], meta.slot_mapping)
    topk_indices_buffer[: hidden_states.shape[0]] = -1

    nd = meta.num_decodes
    if nd > 0:
        seq_lens = meta.seq_lens[:nd]
        append_ivf_decode_keys(key=key, state=state, seq_lens=seq_lens)
        select_ivf_topk(
            key=key,
            state=state,
            q_fp8=q_fp8[:nd],
            weights=weights[:nd],
            row_batch=meta.req_index[:nd],
            query_pos=seq_lens - 1,
            config=config,
            out=topk_indices_buffer[:nd],
            topk_backend=topk_backend,
            seq_lens=seq_lens,
        )

    if meta.num_prefills > 0:
        device = q_fp8.device
        batch_rows = list(range(nd, meta.num_reqs))
        layout = build_ivf_index(
            key=key,
            state=state,
            batch_rows_cpu=batch_rows,
            n_keys_cpu=meta.prefill_seq_lens_cpu,
            num_batch_rows=meta.num_reqs,
            config=config,
            n_prev_keys_cpu=[
                n - q
                for n, q in zip(meta.prefill_seq_lens_cpu, meta.prefill_query_lens_cpu)
            ],
        )
        start, end = meta.num_decode_tokens, meta.num_decode_tokens
        end += meta.num_prefill_tokens
        num_rows = end - start
        qlens = torch.tensor(
            meta.prefill_query_lens_cpu, dtype=torch.int32, pin_memory=True
        ).to(device, non_blocking=True)
        row_batch = torch.repeat_interleave(
            torch.arange(nd, meta.num_reqs, dtype=torch.int32, device=device),
            qlens,
            output_size=num_rows,
        )
        # A query's position: its request's context start plus its offset.
        first_pos = meta.seq_lens[nd : meta.num_reqs] - qlens
        row_start = meta.query_start_loc[nd : meta.num_reqs] - start
        offsets = torch.arange(num_rows, dtype=torch.int32, device=device)
        rel = (row_batch - nd).long()
        query_pos = first_pos[rel] + (offsets - row_start[rel].to(torch.int32))
        select_ivf_topk(
            key=key,
            state=state,
            q_fp8=q_fp8[start:end],
            weights=weights[start:end],
            row_batch=row_batch,
            query_pos=query_pos,
            config=config,
            out=topk_indices_buffer[start:end],
            topk_backend=topk_backend,
            layout=layout,
            max_scratch_bytes=max_scratch_bytes,
        )
    return topk_indices_buffer


@CustomOp.register("sparse_attn_indexer_ivf")
class SparseAttnIndexerIVF(CustomOp):
    """IVF (k-means) replacement for ``SparseAttnIndexerKpool`` (GLM-5.3-Flash)."""

    def __init__(
        self,
        k_cache,
        cid_cache,
        cen_cache,
        config: IvfIndexerConfig,
        topk_indices_buffer: torch.Tensor,
    ):
        super().__init__()
        self.k_cache = k_cache
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
    ) -> torch.Tensor:
        return sparse_attn_indexer_ivf(
            hidden_states,
            self.k_cache.prefix,
            self.cid_cache.prefix,
            self.cen_cache.prefix,
            self.k_cache.kv_cache,
            self.cid_cache.kv_cache,
            self.cen_cache.kv_cache,
            q_fp8,
            weights,
            k,
            self.config,
            self.topk_indices_buffer,
            self.topk_backend,
            self.max_scratch_bytes,
        )
