# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill grouped score of the IVF indexer in CuTe DSL (sm90).

Same contract and work plan as ``ivf_grouped_score_kernel`` (ivf_index.py):
per work unit (``unit_off``: a group's ``kc`` chunks of 64 keys against up
to ``4 * rt`` of its rows) it writes

    score[row, first + m] = k_scale[m] * sum_h w[row, h] * ReLU(q[row, h] . k[m])

for each key m < vis[row] with first + m < cand_cap. Only the fp32 summation
order differs from the Triton kernel (Tier B).

One persistent CTA per SM, warp specialized:
- warp 0 (producer): claims units from the work counter ``unit_off[G + 1]``,
  finds the unit's group, skips the unit's leading row tiles whose rows see
  none of its keys, and fills a key stage: the unit's description, its rows'
  ids, visible counts and first slots, its key scales (shared memory) and
  its ``kc`` key chunks (TMA). Per row tile of 4 rows it TMA-loads the rows'
  32 fp8 heads (4 boxes of [32, 128], together the wgmma B tile) and their
  weights into a Q stage.
- 2 consumer warpgroups: row tile t of the CTA's sequence goes to warpgroup
  t % 2. Per key chunk, one m64n128k128 fp8 wgmma (keys x the tile's 4 rows x
  32 heads), then a register epilogue: ReLU, weight and head sum as 64
  FMNMX + 64 FFMA per thread, a quad reduce-scatter (6 shuffles), the key
  scale and 2 stores. Chunk c's epilogue runs while chunk c + 1's MMA does,
  and the two warpgroups take turns issuing their MMAs (ping-pong), so one's
  epilogue overlaps the other's MMA.
- One-chunk units use 8-row tiles instead (``tile_rows=8``): one m64n256k128
  wgmma per chunk, no ping-pong.
"""

import functools

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils

H = 32  # indexer heads
D = 128  # head dim
BN = 64  # keys per chunk (wgmma M)
TILE_ROWS = 4  # rows per tile: 4 x 32 heads = wgmma N 128
PP_BAR = 6  # ping-pong named barriers: PP_BAR + consumer warpgroup (0: __syncthreads)


class GroupedScore:
    def __init__(self, kc: int, rt: int, q_stages: int = 6, k_stages: int = 2,
                 regs: tuple[int, int] = (40, 232), tile_rows: int = TILE_ROWS):
        # The producer's tile skip looks at one tile per lane.
        assert rt <= 32
        # Rows per tile: 4 (wgmma N 128, ping-pong consumers) or 8 (N 256:
        # one warpgroup's MMA nearer peak).
        assert tile_rows in (4, 8)
        # Ping-pong needs two Q stages: a unit starting at an odd tile gives
        # warpgroup 0 tile 1, and it waits for tile 1's Q before warpgroup 1
        # may take tile 0; with one stage, tile 1 can't load until tile 0 is
        # released.
        assert tile_rows == 8 or q_stages >= 2
        num_consumers = 2
        self.kc, self.rt, self.nw, self.tr = kc, rt, num_consumers, tile_rows
        self.q_stages, self.k_stages = q_stages, k_stages
        self.load_regs, self.mma_regs = regs
        self.threads = 128 * (1 + num_consumers)
        self.unit_keys = BN * kc
        self.unit_rows = self.tr * rt

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,  # (rows * H, D) fp8
        w: cute.Tensor,  # (rows, H) f32
        keys: cute.Tensor,  # (n, D) fp8
        scales: cute.Tensor,  # (n,) f32
        run_start: cute.Tensor,  # (G,) i32
        group_m: cute.Tensor,
        group_vis: cute.Tensor,
        group_off: cute.Tensor,
        unit_off: cute.Tensor,  # (G + 2,)
        row_list: cute.Tensor,
        row_vis: cute.Tensor,
        row_first: cute.Tensor,
        scores: cute.Tensor,  # (rows, cand_cap) f32
        num_groups: cutlass.Int32,
        cand_cap: cutlass.Int32,
        grid_x: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        fp8 = q.element_type
        k_layout = utils.LayoutEnum.from_tensor(keys)
        q_layout = utils.LayoutEnum.from_tensor(q)
        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            fp8, fp8, k_layout.sm90_mma_major_mode(), q_layout.sm90_mma_major_mode(),
            cutlass.Float32, (1, 1, 1), tiler_mn=(BN, self.tr * H))
        # A second copy that always accumulates (the first zero-fills on the
        # first k block): set once each, never inside staged branches.
        self.tiled_mma_acc = sm90_utils.make_trivial_tiled_mma(
            fp8, fp8, k_layout.sm90_mma_major_mode(), q_layout.sm90_mma_major_mode(),
            cutlass.Float32, (1, 1, 1), tiler_mn=(BN, self.tr * H))
        mma_tiler = (BN, self.tr * H, D)
        # Keys: KS stages of KC chunks, each the wgmma A tile [64, 128].
        self.k_smem = sm90_utils.make_smem_layout_a(
            k_layout, mma_tiler, fp8, self.k_stages * self.kc)
        # Q: the wgmma B tile [128, 128] per stage, and the same memory as
        # 4 * QS TMA boxes of [32, 128] (the swizzled K-major layout stacks
        # 8-row atoms, so 4 boxes of 32 rows are the 128-row tile).
        self.q_smem = sm90_utils.make_smem_layout_b(
            q_layout, mma_tiler, fp8, self.q_stages)
        self.q_box_smem = sm90_utils.make_smem_layout_b(
            q_layout, (BN, H, D), fp8, self.q_stages * self.tr)
        self.w_smem = cute.make_layout((self.tr, H, self.q_stages),
                                       stride=(H, 1, self.tr * H))
        g2s = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        tma_k, tma_k_t = cute.nvgpu.cpasync.make_tiled_tma_atom(
            g2s, keys, cute.slice_(self.k_smem, (None, None, 0)), (BN, D))
        tma_q, tma_q_t = cute.nvgpu.cpasync.make_tiled_tma_atom(
            g2s, q, cute.slice_(self.q_box_smem, (None, None, 0)), (H, D))
        tma_w, tma_w_t = cute.nvgpu.cpasync.make_tiled_tma_atom(
            g2s, w, cute.make_layout((1, H), stride=(H, 1)), (1, H))
        self.k_bytes = BN * D * self.kc
        self.q_bytes = self.tr * H * D + self.tr * H * 4

        @cute.struct
        class Storage:
            k_bar: cute.struct.MemRange[cutlass.Int64, self.k_stages * 2]
            q_bar: cute.struct.MemRange[cutlass.Int64, self.q_stages * 2]
            info: cute.struct.MemRange[cutlass.Int32, self.k_stages * 8]
            urow: cute.struct.MemRange[
                cutlass.Int32, self.k_stages * 3 * self.unit_rows]
            sScale: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.k_stages * BN * self.kc], 16]
            sW: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.tr * H * self.q_stages], 128]
            sK: cute.struct.Align[
                cute.struct.MemRange[fp8, cute.cosize(self.k_smem)], 1024]
            sQ: cute.struct.Align[
                cute.struct.MemRange[fp8, cute.cosize(self.q_smem)], 1024]

        self.storage = Storage
        self.kernel(
            self.tiled_mma, self.tiled_mma_acc, self.k_smem, self.q_smem,
            self.q_box_smem, self.w_smem,
            tma_k, tma_k_t, tma_q, tma_q_t, tma_w, tma_w_t, scales, run_start, group_m,
            group_vis, group_off, unit_off, row_list, row_vis, row_first, scores,
            num_groups, cand_cap,
        ).launch(grid=[grid_x, 1, 1], block=[self.threads, 1, 1], stream=stream,
                 min_blocks_per_mp=1)

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_acc: cute.TiledMma,
        k_smem: cute.ComposedLayout,
        q_smem: cute.ComposedLayout,
        q_box_smem: cute.ComposedLayout,
        w_smem: cute.Layout,
        tma_k: cute.CopyAtom,
        mK: cute.Tensor,
        tma_q: cute.CopyAtom,
        mQ: cute.Tensor,
        tma_w: cute.CopyAtom,
        mW: cute.Tensor,
        scales: cute.Tensor,
        run_start: cute.Tensor,
        group_m: cute.Tensor,
        group_vis: cute.Tensor,
        group_off: cute.Tensor,
        unit_off: cute.Tensor,
        row_list: cute.Tensor,
        row_vis: cute.Tensor,
        row_first: cute.Tensor,
        scores: cute.Tensor,
        num_groups: cutlass.Int32,
        cand_cap: cutlass.Int32,
    ):
        KC = self.kc
        NW = self.nw
        UNIT_KEYS = self.unit_keys
        UNIT_ROWS = self.unit_rows
        tidx, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        wg = cute.arch.make_warp_uniform(tidx // 128)

        smem = utils.SmemAllocator()
        st = smem.allocate(self.storage)
        k_pipe = pipeline.PipelineTmaAsync.create(
            barrier_storage=st.k_bar.data_ptr(),
            num_stages=self.k_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 4 * NW),
            tx_count=self.k_bytes,
            defer_sync=True,
        )
        q_pipe = pipeline.PipelineTmaAsync.create(
            barrier_storage=st.q_bar.data_ptr(),
            num_stages=self.q_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 4),
            tx_count=self.q_bytes,
            defer_sync=True,
        )
        sK = st.sK.get_tensor(k_smem.outer, swizzle=k_smem.inner)
        sQ = st.sQ.get_tensor(q_smem.outer, swizzle=q_smem.inner)
        sQbox = cute.make_tensor(
            cute.recast_ptr(sQ.iterator, q_box_smem.inner), q_box_smem.outer)
        sW = st.sW.get_tensor(w_smem)
        info = st.info.get_tensor(cute.make_layout((8, self.k_stages), stride=(1, 8)))
        # Per key stage, the unit's rows from its first live tile on: id,
        # visible keys (0 past the unit's rows) and first candidate slot.
        meta = st.urow.get_tensor(cute.make_layout(
            (UNIT_ROWS, 3, self.k_stages), stride=(1, UNIT_ROWS, 3 * UNIT_ROWS)))
        sScale = st.sScale.get_tensor(
            cute.make_layout((BN * KC, self.k_stages), stride=(1, BN * KC)))

        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        if wg == 0:
            cute.arch.setmaxregister_decrease(self.load_regs)
            if warp == 0:
                cute.nvgpu.cpasync.prefetch_descriptor(tma_k)
                cute.nvgpu.cpasync.prefetch_descriptor(tma_q)
                cute.nvgpu.cpasync.prefetch_descriptor(tma_w)
                self.producer(
                    k_pipe, q_pipe, tma_k, mK, tma_q, mQ, tma_w, mW, sK, sQbox, sW,
                    info, meta, sScale, scales, run_start, group_m, group_vis,
                    group_off, unit_off, row_list, row_vis, row_first, num_groups,
                    lane, KC, UNIT_KEYS, UNIT_ROWS)
        else:
            cute.arch.setmaxregister_increase(self.mma_regs)
            if cutlass.const_expr(self.tr == 8):
                self.consumer_wide(
                    tiled_mma, tiled_mma_acc, k_pipe, q_pipe, sK, sQ, sW, info, meta,
                    sScale, scores, cand_cap, tidx % 128, wg - 1, KC, NW)
            else:
                self.consumer_pp(
                    tiled_mma, tiled_mma_acc, k_pipe, q_pipe, sK, sQ, sW, info, meta,
                    sScale, scores, cand_cap, tidx % 128, wg - 1, KC, NW)

    @cute.jit
    def producer(self, k_pipe, q_pipe, tma_k, mK, tma_q, mQ, tma_w, mW, sK, sQbox, sW,
                 info, meta, sScale, scales, run_start, group_m, group_vis, group_off,
                 unit_off, row_list, row_vis, row_first, num_groups, lane,
                 KC: cutlass.Constexpr,
                 UNIT_KEYS: cutlass.Constexpr, UNIT_ROWS: cutlass.Constexpr):
        kst = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.k_stages)
        qst = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.q_stages)
        total = unit_off[num_groups]
        counter = unit_off.iterator + (num_groups + 1)
        g = cutlass.Int32(0)
        u = cutlass.Int32(0)
        if lane == 0:
            u = cute.arch.atomic_add(
                counter, cutlass.Int32(1), sem="relaxed", scope="gpu")
        u = cute.arch.shuffle_sync(u, 0)
        while u < total:
            # The group holding unit u: the first g with unit_off[g + 1] > u.
            # A CTA's units ascend, so scan forward 32 groups at a time.
            found = cutlass.Int32(0)
            while found == 0:
                gl = g + lane
                ends = cutlass.Int32(0x7FFFFFFF)
                if gl < num_groups:
                    ends = unit_off[gl + 1]
                mask = cute.arch.vote_ballot_sync(ends > u)
                if mask != 0:
                    g = g + cute.arch.popc((mask & (-mask)) - 1)
                    found = cutlass.Int32(1)
                else:
                    g = g + 32
            rel = u - unit_off[g]
            vis_max = group_vis[g]
            n_chunks = (vis_max + UNIT_KEYS - 1) // UNIT_KEYS
            m0 = (rel % n_chunks) * UNIT_KEYS
            r_begin = (rel // n_chunks) * UNIT_ROWS
            r_end = cutlass.min(group_m[g], r_begin + UNIT_ROWS)
            row_base = group_off[g]
            # Tiles whose rows all see none of these keys lead the unit (a
            # group's rows ascend by visible count): skip them.
            tr = r_begin + self.tr * lane
            seen = cutlass.Int32(0)
            if tr < r_end:
                seen = row_vis[row_base + cutlass.min(tr + self.tr - 1, r_end - 1)]
            live = cute.arch.vote_ballot_sync(seen > m0)
            if live != 0:
                skip = cute.arch.popc((live & (-live)) - 1)
                rb = r_begin + self.tr * skip
                n_tiles = (r_end - rb + self.tr - 1) // self.tr
                key0 = run_start[g] + m0
                # Key stage: the unit's description, then its KC chunks.
                k_pipe.sync_object_empty.wait(kst.index, kst.phase)
                info[(0, kst.index)] = m0
                info[(1, kst.index)] = rb
                info[(2, kst.index)] = r_end
                info[(3, kst.index)] = row_base
                info[(4, kst.index)] = key0
                info[(5, kst.index)] = vis_max - m0
                info[(6, kst.index)] = n_tiles
                # The unit's key scales (0 past the cluster).
                for j in cutlass.range_constexpr(BN * KC // 32):
                    kk = j * 32 + lane
                    sc = cutlass.Float32(0.0)
                    if kk < vis_max - m0:
                        sc = scales[key0 + kk]
                    sScale[(kk, kst.index)] = sc
                for j in cutlass.range_constexpr(UNIT_ROWS // 32):
                    rr = rb + j * 32 + lane
                    slot = row_base + cutlass.min(rr, r_end - 1)
                    rv = row_vis[slot]
                    if rr >= r_end:
                        rv = cutlass.Int32(0)  # padding row: no stores
                    meta[(j * 32 + lane, 0, kst.index)] = row_list[slot]
                    meta[(j * 32 + lane, 1, kst.index)] = rv
                    meta[(j * 32 + lane, 2, kst.index)] = row_first[slot]
                # The elected lane's arrive publishes only its own writes:
                # order the warp's first.
                cute.arch.sync_warp()
                k_pipe.sync_object_full.arrive(kst.index, k_pipe.producer_mask)
                bar = k_pipe.producer_get_barrier(kst)
                for c in cutlass.range_constexpr(KC):
                    gK = cute.local_tile(
                        cute.domain_offset((key0 + c * BN, 0), mK), (BN, D), (0, 0))
                    tKs, tKg = cute.nvgpu.cpasync.tma_partition(
                        tma_k, 0, cute.make_layout(1), cute.group_modes(sK, 0, 2),
                        cute.group_modes(gK, 0, 2))
                    cute.copy(tma_k, tKg, tKs[(None, kst.index * KC + c)],
                              tma_bar_ptr=bar)
                ks = kst.index
                kst.advance()
                for t in cutlass.range(n_tiles, unroll=1):
                    q_pipe.producer_acquire(qst)
                    qbar = q_pipe.producer_get_barrier(qst)
                    for i in cutlass.range_constexpr(self.tr):
                        row = meta[(self.tr * t + i, 0, ks)]
                        gQ = cute.local_tile(
                            cute.domain_offset((row * H, 0), mQ), (H, D), (0, 0))
                        tQs, tQg = cute.nvgpu.cpasync.tma_partition(
                            tma_q, 0, cute.make_layout(1),
                            cute.group_modes(sQbox, 0, 2),
                            cute.group_modes(gQ, 0, 2))
                        cute.copy(tma_q, tQg, tQs[(None, qst.index * self.tr + i)],
                                  tma_bar_ptr=qbar)
                        gW = cute.local_tile(
                            cute.domain_offset((row, 0), mW), (1, H), (0, 0))
                        sWi = cute.local_tile(
                            sW[(None, None, qst.index)], (1, H), (i, 0))
                        tWs, tWg = cute.nvgpu.cpasync.tma_partition(
                            tma_w, 0, cute.make_layout(1), cute.group_modes(sWi, 0, 2),
                            cute.group_modes(gW, 0, 2))
                        cute.copy(tma_w, tWg, tWs, tma_bar_ptr=qbar)
                    qst.advance()
            if lane == 0:
                u = cute.arch.atomic_add(counter, cutlass.Int32(1), sem="relaxed",
                                         scope="gpu")
            u = cute.arch.shuffle_sync(u, 0)
        # End: a key stage with no unit (a plain arrive, no bytes).
        k_pipe.sync_object_empty.wait(kst.index, kst.phase)
        info[(0, kst.index)] = cutlass.Int32(-1)
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive(k_pipe.producer_get_barrier(kst))


    @cute.jit
    def issue2(self, mma0, mma1, tCrA, tCrB, acc, kidx, qidx):
        cute.nvgpu.warpgroup.fence()
        for kb in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
            mma = mma0 if kb == 0 else mma1
            cute.gemm(mma, acc, tCrA[(None, None, kb, kidx)],
                      tCrB[(None, None, kb, qidx)], acc)
        cute.nvgpu.warpgroup.commit_group()

    @cute.jit
    def chunk_epilogue(self, acc, wr, sScale, scores, cand_cap, qq, bit0, bit1, kidx, c,
                       m_lo, row, vis, first):
        """Chunk c of a 4-row tile: per thread, its 2 keys' head sums for the
        4 rows, a quad reduce-scatter (lane qq keeps row qq) and 2 stores."""
        mk0 = c * BN + m_lo
        sc0 = sScale[(mk0, kidx)]
        sc1 = sScale[(mk0 + 8, kidx)]
        s = cute.make_rmem_tensor((2, self.tr), cutlass.Float32)
        for v1 in cutlass.range_constexpr(2):
            for rr in cutlass.range_constexpr(self.tr):
                x = cutlass.Float32(0.0)
                for a in cutlass.range_constexpr(4):
                    for b in cutlass.range_constexpr(2):
                        v = acc[b + 2 * v1 + 4 * (4 * rr + a)]
                        x = cute.arch.fmax(v, 0.0) * wr[(b, a, rr)] + x
                s[(v1, rr)] = x
        fin = cute.make_rmem_tensor((2,), cutlass.Float32)
        for v1 in cutlass.range_constexpr(2):
            p0 = cutlass.Float32(0.0)
            p1 = cutlass.Float32(0.0)
            for j in cutlass.range_constexpr(2):
                lo = s[(v1, j)]
                hi = s[(v1, 2 + j)]
                mine = lo
                other = hi
                if bit1 == 1:
                    mine = hi
                    other = lo
                got = cute.arch.shuffle_sync_bfly(other, 2) + mine
                if j == 0:
                    p0 = got
                else:
                    p1 = got
            mine = p0
            other = p1
            if bit0 == 1:
                mine = p1
                other = p0
            fin[v1] = cute.arch.shuffle_sync_bfly(other, 1) + mine
        mk = mk0
        if (mk < vis) & (first + mk < cand_cap):
            scores[(row, first + mk)] = fin[0] * sc0
        mk = mk0 + 8
        if (mk < vis) & (first + mk < cand_cap):
            scores[(row, first + mk)] = fin[1] * sc1

    @cute.jit
    def pp_sync(self, wgi):
        # Wait for this warpgroup's turn (both consumers' 256 threads).
        cute.arch.barrier(barrier_id=PP_BAR + wgi, number_of_threads=256)

    @cute.jit
    def pp_arrive(self, wgi):
        # Hand the turn to the other consumer warpgroup.
        if cutlass.const_expr(hasattr(cute.arch, "barrier_arrive")):
            cute.arch.barrier_arrive(barrier_id=PP_BAR + 1 - wgi, number_of_threads=256)
        else:
            pipeline.NamedBarrier(barrier_id=PP_BAR + 1 - wgi, num_threads=256).arrive()

    @cute.jit
    def consumer_pp(self, tiled_mma, tiled_mma_acc, k_pipe, q_pipe, sK, sQ, sW, info,
                    meta, sScale, scores, cand_cap, tid, wgi, KC: cutlass.Constexpr,
                    NW: cutlass.Constexpr):
        """4-row tiles. Row tile t of the CTA's sequence goes to warpgroup
        t % 2; per tile, chunk c's epilogue runs while chunk c + 1's MMA does
        (two accumulators, the chunk loop unrolled over KC).

        Ping-pong turns (FA3's warp scheduler barrier): the two warpgroups
        alternate issuing a tile's first MMA, so one's epilogue overlaps the
        other's MMA instead of both running in lockstep. Per unit each
        warpgroup takes ceil(n_tiles / 2) tile slots (t = t0 + 2 p, t0 from
        the global tile parity so the Q stages stay aligned); a slot past the
        unit's tiles is a dummy that only takes its turn (no Q wait, MMA or
        epilogue), so both take the same number of turns."""
        tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
        tiled_mma_acc.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
        thr_mma = tiled_mma.get_slice(tid)
        tCrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sK))
        tCrB = tiled_mma.make_fragment_B(thr_mma.partition_B(sQ))
        shape_c = thr_mma.partition_shape_C((BN, self.tr * H))
        accs = [cute.make_rmem_tensor(shape_c, cutlass.Float32),
                cute.make_rmem_tensor(shape_c, cutlass.Float32)]
        lane = tid % 32
        warp = tid // 32
        qq = lane % 4
        bit0 = qq % 2
        bit1 = qq // 2
        m_lo = warp * 16 + lane // 4
        kst = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.k_stages)
        qst = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.q_stages)
        for _ in cutlass.range(wgi, unroll=1):
            qst.advance()
        # Warpgroup 0 takes the first turn.
        if wgi == 1:
            self.pp_arrive(wgi)
        qseq = cutlass.Int32(0)  # tiles before this unit (both warpgroups)
        running = cutlass.Int32(1)
        while running != 0:
            k_pipe.consumer_wait(kst)
            m0 = info[(0, kst.index)]
            if m0 < 0:
                running = cutlass.Int32(0)
            else:
                n_tiles = info[(6, kst.index)]
                ks = kst.index
                t0 = (wgi - qseq % NW + NW) % NW
                for p in cutlass.range((n_tiles + 1) // 2, unroll=1):
                    t = t0 + NW * p
                    r = self.tr * cutlass.min(t, n_tiles - 1)
                    row = meta[(r + qq, 0, ks)]
                    vis = meta[(r + qq, 1, ks)] - m0
                    first = meta[(r + qq, 2, ks)] + m0
                    seen = cutlass.max(
                        cutlass.max(meta[(r, 1, ks)], meta[(r + 1, 1, ks)]),
                        cutlass.max(meta[(r + 2, 1, ks)], meta[(r + 3, 1, ks)]))
                    n_ch = cutlass.min(KC, (seen - m0 + BN - 1) // BN)
                    if t >= n_tiles:
                        n_ch = cutlass.Int32(0)
                    if t < n_tiles:
                        q_pipe.consumer_wait(qst)
                    qs = qst.index
                    # A dummy slot (t >= n_tiles) reads sW unwaited; the
                    # values are unused (n_ch = 0, nothing stored).
                    wr = cute.make_rmem_tensor((2, 4, self.tr), cutlass.Float32)
                    for rr in cutlass.range_constexpr(self.tr):
                        for a in cutlass.range_constexpr(4):
                            for b in cutlass.range_constexpr(2):
                                wr[(b, a, rr)] = sW[(rr, 8 * a + 2 * qq + b, qs)]
                    self.pp_sync(wgi)
                    if n_ch > 0:
                        self.issue2(tiled_mma, tiled_mma_acc, tCrA, tCrB, accs[0],
                                    ks * KC, qs)
                    self.pp_arrive(wgi)
                    # Chunk c uses accumulator c % 2. Issue c + 1 before c's
                    # epilogue; wait(1) retires c when c + 1 is in flight,
                    # wait(0) otherwise (a wait on both arms of every branch,
                    # so ptxas keeps the MMAs asynchronous).
                    for c in cutlass.range_constexpr(KC):
                        if cutlass.const_expr(c + 1 < KC):
                            if c + 1 < n_ch:
                                self.issue2(tiled_mma, tiled_mma_acc, tCrA, tCrB,
                                            accs[(c + 1) % 2], ks * KC + c + 1, qs)
                                cute.nvgpu.warpgroup.wait_group(1)
                            else:
                                cute.nvgpu.warpgroup.wait_group(0)
                        else:
                            cute.nvgpu.warpgroup.wait_group(0)
                        if c < n_ch:
                            if c == n_ch - 1:
                                q_pipe.consumer_release(qst)
                            self.chunk_epilogue(
                                accs[c % 2], wr, sScale, scores, cand_cap, qq, bit0,
                                bit1, ks, c, m_lo, row, vis, first)
                    if t < n_tiles:
                        for _ in cutlass.range_constexpr(NW):
                            qst.advance()
                qseq = qseq + n_tiles
                k_pipe.consumer_release(kst)
            kst.advance()
        # Warpgroup 1's first arrive is one more than warpgroup 0's turns
        # consume: take it, so no barrier is left half-arrived at exit.
        if wgi == 0:
            self.pp_sync(wgi)

    @cute.jit
    def consumer_wide(self, tiled_mma, tiled_mma_acc, k_pipe, q_pipe, sK, sQ, sW, info,
                      meta, sScale, scores, cand_cap, tid, wgi, KC: cutlass.Constexpr,
                      NW: cutlass.Constexpr):
        """8-row tiles: one m64n256k128 MMA per chunk. After the quad
        reduce-scatter each thread holds rows 2 qq and 2 qq + 1 of its 2 keys."""
        TR = self.tr
        tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
        tiled_mma_acc.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
        thr_mma = tiled_mma.get_slice(tid)
        tCrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sK))
        tCrB = tiled_mma.make_fragment_B(thr_mma.partition_B(sQ))
        acc = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((BN, TR * H)), cutlass.Float32)
        lane = tid % 32
        warp = tid // 32
        qq = lane % 4
        bit0 = qq % 2
        bit1 = qq // 2
        m_lo = warp * 16 + lane // 4
        kst = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.k_stages)
        qst = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.q_stages)
        t_seq = cutlass.Int32(0)
        running = cutlass.Int32(1)
        while running != 0:
            k_pipe.consumer_wait(kst)
            m0 = info[(0, kst.index)]
            if m0 < 0:
                running = cutlass.Int32(0)
            else:
                n_tiles = info[(6, kst.index)]
                ks = kst.index
                for t in cutlass.range(n_tiles, unroll=1):
                    if t_seq % NW == wgi:
                        r = TR * t
                        row0 = meta[(r + 2 * qq, 0, ks)]
                        vis0 = meta[(r + 2 * qq, 1, ks)] - m0
                        first0 = meta[(r + 2 * qq, 2, ks)] + m0
                        row1 = meta[(r + 2 * qq + 1, 0, ks)]
                        vis1 = meta[(r + 2 * qq + 1, 1, ks)] - m0
                        first1 = meta[(r + 2 * qq + 1, 2, ks)] + m0
                        seen = meta[(r, 1, ks)]
                        for i in cutlass.range_constexpr(1, TR):
                            seen = cutlass.max(seen, meta[(r + i, 1, ks)])
                        n_ch = cutlass.min(KC, (seen - m0 + BN - 1) // BN)
                        q_pipe.consumer_wait(qst)
                        qs = qst.index
                        for c in cutlass.range(n_ch, unroll=1):
                            self.issue2(tiled_mma, tiled_mma_acc, tCrA, tCrB, acc,
                                        ks * KC + c, qs)
                            cute.nvgpu.warpgroup.wait_group(0)
                            mk0 = c * BN + m_lo
                            sc0 = sScale[(mk0, ks)]
                            sc1 = sScale[(mk0 + 8, ks)]
                            s = cute.make_rmem_tensor((2, TR), cutlass.Float32)
                            for v1 in cutlass.range_constexpr(2):
                                for rr in cutlass.range_constexpr(TR):
                                    x = cutlass.Float32(0.0)
                                    for a in cutlass.range_constexpr(4):
                                        for b in cutlass.range_constexpr(2):
                                            v = acc[b + 2 * v1 + 4 * (4 * rr + a)]
                                            wv = sW[(rr, 8 * a + 2 * qq + b, qs)]
                                            x = cute.arch.fmax(v, 0.0) * wv + x
                                    s[(v1, rr)] = x
                            if c == n_ch - 1:
                                q_pipe.consumer_release(qst)
                            # Quad reduce-scatter: lane qq keeps rows 2 qq, 2 qq + 1.
                            p = cute.make_rmem_tensor((2, 4), cutlass.Float32)
                            for v1 in cutlass.range_constexpr(2):
                                for j in cutlass.range_constexpr(4):
                                    mine = s[(v1, j)]
                                    other = s[(v1, 4 + j)]
                                    if bit1 == 1:
                                        mine = s[(v1, 4 + j)]
                                        other = s[(v1, j)]
                                    p[(v1, j)] = (
                                        cute.arch.shuffle_sync_bfly(other, 2) + mine)
                            f = cute.make_rmem_tensor((2, 2), cutlass.Float32)
                            for v1 in cutlass.range_constexpr(2):
                                for i in cutlass.range_constexpr(2):
                                    mine = p[(v1, i)]
                                    other = p[(v1, 2 + i)]
                                    if bit0 == 1:
                                        mine = p[(v1, 2 + i)]
                                        other = p[(v1, i)]
                                    f[(v1, i)] = (
                                        cute.arch.shuffle_sync_bfly(other, 1) + mine)
                            for v1 in cutlass.range_constexpr(2):
                                mk = mk0 + 8 * v1
                                sc = sc0 if v1 == 0 else sc1
                                if (mk < vis0) & (first0 + mk < cand_cap):
                                    scores[(row0, first0 + mk)] = f[(v1, 0)] * sc
                                if (mk < vis1) & (first1 + mk < cand_cap):
                                    scores[(row1, first1 + mk)] = f[(v1, 1)] * sc
                    qst.advance()
                    t_seq = t_seq + 1
                k_pipe.consumer_release(kst)
            kst.advance()


@functools.cache
def _compiled(kc: int, rt: int, num_sms: int, tile_rows: int, q_stages: int):
    """The kernel compiled for TVM-FFI calls with torch tensors; shapes are
    runtime values (one compile per unit shape and SM count)."""
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

    def sym():
        return cute.sym_int32()

    fp8, f32, i32 = cutlass.Float8E4M3FN, cutlass.Float32, cutlass.Int32
    rows, n, g, nr = sym(), sym(), sym(), sym()
    args = [
        make_fake_tensor(fp8, (sym(), D), (D, 1), assumed_align=16),
        make_fake_tensor(f32, (sym(), H), (H, 1), assumed_align=16),
        make_fake_tensor(fp8, (n, D), (D, 1), assumed_align=16),
        make_fake_tensor(f32, (sym(),), (1,), assumed_align=4),
        make_fake_tensor(i32, (g,), (1,), assumed_align=4),
        make_fake_tensor(i32, (sym(),), (1,), assumed_align=4),
        make_fake_tensor(i32, (sym(),), (1,), assumed_align=4),
        make_fake_tensor(i32, (sym(),), (1,), assumed_align=4),
        make_fake_tensor(i32, (sym(),), (1,), assumed_align=4),
        make_fake_tensor(i32, (nr,), (1,), assumed_align=4),
        make_fake_tensor(i32, (sym(),), (1,), assumed_align=4),
        make_fake_tensor(i32, (sym(),), (1,), assumed_align=4),
        # A 32-bit row stride (a 64-bit one costs 3-5%): callers keep
        # rows * stride below 2^31 (``supported``).
        make_fake_tensor(f32, (rows, sym()), (sym(), 1), assumed_align=4),
        cutlass.Int32(1),
        cutlass.Int32(1),
    ]
    kernel = GroupedScore(kc, rt, q_stages=q_stages, tile_rows=tile_rows)
    return cute.compile(kernel, *args, num_sms,
                        make_fake_stream(use_tvm_ffi_env_stream=False),
                        options="--enable-tvm-ffi")


def supported(q_fp8: torch.Tensor, scores: torch.Tensor, row_tiles: int,
              mma_cols: int) -> bool:
    return (q_fp8.shape[1:] == (H, D) and mma_cols == TILE_ROWS * H and row_tiles <= 32
            and q_fp8.is_contiguous() and scores.stride(1) == 1
            and scores.shape[0] * scores.stride(0) < 2**31)


def grouped_score(*, q_fp8, weights, run_keys, run_scales, run_start, group_m,
                  group_vis, group_off, unit_off, row_list, row_vis, row_first,
                  scores, num_groups: int, key_chunks: int, row_tiles: int,
                  num_sms: int, tile_rows: int = TILE_ROWS, q_stages: int = 6) -> None:
    """Launch on the current stream; arguments as the Triton kernel's."""
    fn = _compiled(key_chunks, row_tiles, num_sms, tile_rows, q_stages)
    fn(q_fp8.view(-1, D), weights, run_keys, run_scales, run_start.view(-1), group_m,
       group_vis, group_off, unit_off, row_list, row_vis, row_first, scores, num_groups,
       scores.shape[1], cuda.CUstream(torch.cuda.current_stream().cuda_stream))
