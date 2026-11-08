# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill per-row top-k of the IVF indexer in CuTe DSL (sm90).

Contract of the shared ``per_row`` top-k as the IVF prefill uses it: for each
row with ``lens[row] > 0`` (only rows over budget have one), write into
``out[row, :K]`` the slots of the row's K largest scores among
``scores[row, :lens[row]]``, in no particular order. Rows with ``lens[row] ==
0`` are not written (the emit keeps all of their candidates). Exact ties at
the cut are taken in arrival order, as in the shared kernel (Tier B).

Rows are at most ``cap`` scores long (the host sends longer rows to the
shared kernel). A group of ``group`` warps owns a row at a time (no
block-wide barriers): the row's scores fill a shared-memory buffer of
(order-preserving key, slot) and a coarse histogram, read once; then the K
best of the buffer are written (``select``):
1. A 1024-bin histogram of a coarse, monotone key (the sign, and the
   magnitude clamped to 32 binades from 2^-14 with 4 mantissa bits). A group
   scan finds the cut bin: ``a1`` entries lie above it, ``n1`` in it.
2. A pass writes the entries above the cut bin and stages the bin's.
3. Up to ``ecap`` staged entries get their rank (greater keys, then earlier
   equal ones) and the ranks below the need are written; past that, radix
   rounds (10, 10, 10, 2 bits) over the buffer find the exact cut key, then
   a last pass writes the keys above it and as many ties as fit.
"""

import functools

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils

NB = 1024  # histogram bins per group
E_LO = 113  # coarse key: magnitudes from 2^-14 (biased fp32 exponent 113)
MB = 4  # coarse key mantissa bits: 32 binades x 16 = 512 bins per sign
SH = 23 - MB
SIGN = -2147483648  # 0x80000000 as int32
LOW = 0x7FFFFFFF


class IvfTopK:
    def __init__(self, k: int = 512, warps: int = 8, group: int = 8, cap: int = 2048,
                 ecap: int = 512, unroll: int = 8, rank: int = 64, fcap: int = 256):
        assert warps % group == 0 and NB % (32 * group) == 0
        assert group == 1 or warps // group <= 15  # named barriers 1..15
        self.k, self.warps, self.g = k, warps, group
        self.cap, self.ecap, self.unroll = cap, ecap, unroll
        # Cut bins up to rank entries are resolved by ranking them; larger
        # ones by a second histogram (the next 10 key bits) whose cut bin,
        # up to fcap entries, is then ranked.
        self.rank, self.fcap = rank, fcap
        self.tg = 32 * group
        assert cap >= k and cap % (self.tg * unroll) == 0
        self.ngroups = warps // group
        self.bpt = NB // self.tg

    @cute.jit
    def __call__(
        self,
        scores: cute.Tensor,  # (rows, width) i32: the fp32 score bits
        lens: cute.Tensor,  # (rows,) i32
        out: cute.Tensor,  # (rows, K) i32
        num_rows: cutlass.Int32,
        grid_x: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        G = self.ngroups

        @cute.struct
        class Storage:
            hist: cute.struct.MemRange[cutlass.Int32, NB * G]
            wsum: cute.struct.MemRange[cutlass.Int32, 32 * G]
            var: cute.struct.MemRange[cutlass.Int32, 8 * G]
            akey: cute.struct.MemRange[cutlass.Int32, self.cap * G]
            aidx: cute.struct.MemRange[cutlass.Int32, self.cap * G]
            ekey: cute.struct.MemRange[cutlass.Int32, self.ecap * G]
            eidx: cute.struct.MemRange[cutlass.Int32, self.ecap * G]
            fkey: cute.struct.MemRange[cutlass.Int32, self.fcap * G]
            fidx: cute.struct.MemRange[cutlass.Int32, self.fcap * G]

        self.storage = Storage
        self.kernel(scores, lens, out, num_rows).launch(
            grid=[grid_x, 1, 1], block=[32 * self.warps, 1, 1], stream=stream)

    # -- keys ------------------------------------------------------------

    @cute.jit
    def skey(self, b):
        """Order-preserving key, compared as a signed int32 (an involution:
        it maps keys back to fp32 bits too)."""
        return b ^ ((b >> 31) & LOW)

    @cute.jit
    def coarse(self, s):
        """Monotone 10-bit bin of the key ``s`` (higher bin, larger)."""
        b = self.skey(s)
        q = ((b & LOW) >> SH) - E_LO * (1 << MB)
        q = cutlass.min(cutlass.max(q, 0), 511)
        r = 512 + q
        if b < 0:
            r = 511 - q
        return r

    @cute.jit
    def lower(self, c):
        """The smallest key in coarse bin c (0 <= c < NB)."""
        r = cutlass.Int32(0)
        if c >= 512:
            q = c - 512
            if q > 0:
                r = (q + E_LO * (1 << MB)) << SH
        else:
            q = 511 - c
            m = cutlass.Int32(LOW)
            if q < 511:
                m = ((q + 1 + E_LO * (1 << MB)) << SH) - 1
            r = (m | SIGN) ^ LOW
        return r

    @cute.jit
    def hi(self, u, s: cutlass.Constexpr):
        """The unsigned key's bits from ``s`` up, as a non-negative int32."""
        r = u
        if cutlass.const_expr(s > 0):
            r = (u >> s) & ((1 << (32 - s)) - 1)
        return r

    # -- group helpers ---------------------------------------------------

    @cute.jit
    def gsync(self, grp):
        if cutlass.const_expr(self.g == 1):
            cute.arch.sync_warp()
        else:
            cute.arch.barrier(barrier_id=1 + grp, number_of_threads=self.tg)

    @cute.jit
    def wagg(self, pred, counter, lane):
        """Warp-aggregated slot: one shared atomic per warp. Every lane calls."""
        mask = cute.arch.vote_ballot_sync(pred)
        base = cutlass.Int32(0)
        if mask != 0:
            leader = cute.arch.popc((mask & (-mask)) - 1)
            got = cutlass.Int32(0)
            if lane == leader:
                got = cute.arch.atomic_add(counter, cute.arch.popc(mask), sem="relaxed",
                                           scope="cta")
            base = cute.arch.shuffle_sync(got, leader)
        below = mask & ((cutlass.Int32(1) << lane) - 1)
        return base + cute.arch.popc(below)

    @cute.jit
    def cut(self, hist, wsum, var, need, lt, lane, wig, grp):
        """The bin holding the need-th largest entry: (bin, count above it,
        its count). Thread lt holds bins NB-1-BPT*lt down."""
        s = cutlass.Int32(0)
        for j in cutlass.range_constexpr(self.bpt):
            s = s + hist[NB - 1 - (lt * self.bpt + j)]
        x = s
        for d in cutlass.range_constexpr(5):
            y = cute.arch.shuffle_sync(x, cutlass.max(lane - (1 << d), 0))
            if lane >= (1 << d):
                x = x + y
        pre = cutlass.Int32(0)
        if cutlass.const_expr(self.g > 1):
            if lane == 31:
                wsum[wig] = x
            self.gsync(grp)
            for w in cutlass.range_constexpr(self.g):
                if w < wig:
                    pre = pre + wsum[w]
        c = pre + x - s
        if (c < need) & (c + s >= need):
            found = cutlass.Int32(0)
            for j in cutlass.range_constexpr(self.bpt):
                bb = NB - 1 - (lt * self.bpt + j)
                h = hist[bb]
                if found == 0:
                    if c + h >= need:
                        var[0] = bb
                        var[1] = c
                        var[2] = h
                        found = cutlass.Int32(1)
                    else:
                        c = c + h
        self.gsync(grp)
        return var[0], var[1], var[2]

    @cute.jit
    def zero_hist(self, hist, lt):
        for j in cutlass.range_constexpr(self.bpt):
            hist[lt + j * self.tg] = cutlass.Int32(0)

    # -- selection of the buffer's K best ---------------------------------

    @cute.jit
    def select(self, sm, cnt, out, row, lt, lane, wig, grp):
        """Write the K largest of buffer A[:cnt] (cnt >= K) to out[row], from
        the buffer's coarse histogram. Ends with a group barrier."""
        (hist, wsum, var, akey, aidx, ekey, eidx, fkey, fidx) = sm
        TG = self.tg
        K = self.k
        n_out = var.iterator + 3
        n_eq = var.iterator + 4
        n_tie = var.iterator + 5
        if lt == 0:
            var[3] = cutlass.Int32(0)
            var[4] = cutlass.Int32(0)
            var[5] = cutlass.Int32(0)
        self.gsync(grp)
        b1, a1, n1 = self.cut(hist, wsum, var, cutlass.Int32(K), lt, lane, wig, grp)
        need = K - a1
        direct = n1 == need
        split = n1 != need
        staged = n1 <= self.ecap
        lo1 = self.lower(b1)
        lo2 = cutlass.Int32(LOW)
        any2 = b1 < NB - 1
        if any2:
            lo2 = self.lower(b1 + 1)
        for base in cutlass.range(0, cnt, TG, unroll=1):
            j = base + lt
            live = j < cnt
            s = cutlass.Int32(SIGN)
            idx = cutlass.Int32(0)
            if live:
                s = akey[j]
                idx = aidx[j]
            ge = live & (s >= lo1)
            gt = ge & any2 & (s >= lo2)
            sel = gt | (direct & ge)
            pos = self.wagg(sel, n_out, lane)
            if sel:
                out[(row, pos)] = idx
            eq = (ge != gt) & split & staged
            pe = self.wagg(eq, n_eq, lane)
            if eq:
                ekey[pe] = s
                eidx[pe] = idx
        self.gsync(grp)
        if split:
            # Bins 0, 511, 512 and 1023 are clamped (mixed exponents): the
            # next key bits don't order them.
            ordered = (b1 != 0) & (b1 != 511) & (b1 != 512) & (b1 != NB - 1)
            if staged & (n1 <= self.rank):
                self.rank_emit(ekey, eidx, n1, need, a1, out, row, lt)
            elif staged & ordered:
                self.level2(sm, cnt, out, row, n1, lo1, lo2, any2, a1, need, lt, lane,
                            wig, grp)
            else:
                self.refine(sm, cnt, out, row, lo1, lo2, any2, a1, need, lt, lane, wig,
                            grp)
        self.gsync(grp)

    @cute.jit
    def rank_emit(self, keys, idxs, m, need, base, out, row, lt):
        """Each of m entries' rank (greater keys, then earlier entries with an
        equal key) is its slot after base, if below need."""
        for j in cutlass.range(lt, m, self.tg, unroll=1):
            u = keys[j]
            r = cutlass.Int32(0)
            for i in cutlass.range(m, unroll=4):
                w = keys[i]
                if (w > u) | ((w == u) & (i < j)):
                    r = r + 1
            if r < need:
                out[(row, base + r)] = idxs[j]

    @cute.jit
    def level2(self, sm, cnt, out, row, n1, lo1, lo2, any2, a1, need, lt, lane, wig,
               grp):
        """A staged cut bin over rank entries: histogram of the next 10 key
        bits (its entries share the bits above), then rank the second cut bin."""
        (hist, wsum, var, akey, aidx, ekey, eidx, fkey, fidx) = sm
        TG = self.tg
        n_out = var.iterator + 3
        n_f = var.iterator + 4
        self.zero_hist(hist, lt)
        if lt == 0:
            var[4] = cutlass.Int32(0)
        self.gsync(grp)
        for j in cutlass.range(lt, n1, TG, unroll=4):
            cute.arch.atomic_add(hist.iterator + (((ekey[j] ^ SIGN) >> 9) & (NB - 1)),
                                 cutlass.Int32(1), sem="relaxed", scope="cta")
        self.gsync(grp)
        c2, a2, n2 = self.cut(hist, wsum, var, need, lt, lane, wig, grp)
        if n2 > self.fcap:
            self.refine(sm, cnt, out, row, lo1, lo2, any2, a1, need, lt, lane, wig, grp)
        else:
            for base in cutlass.range(0, n1, TG, unroll=1):
                j = base + lt
                live = j < n1
                s = cutlass.Int32(SIGN)
                idx = cutlass.Int32(0)
                if live:
                    s = ekey[j]
                    idx = eidx[j]
                f = ((s ^ SIGN) >> 9) & (NB - 1)
                up = live & (f > c2)
                pos = self.wagg(up, n_out, lane)
                if up:
                    out[(row, pos)] = idx
                eq = live & (f == c2)
                pe = self.wagg(eq, n_f, lane)
                if eq:
                    fkey[pe] = s
                    fidx[pe] = idx
            self.gsync(grp)
            self.rank_emit(fkey, fidx, n2, need - a2, a1 + a2, out, row, lt)

    @cute.jit
    def refine(self, sm, cnt, out, row, lo1, lo2, any2, a1, need, lt, lane, wig, grp):
        """A cut bin over ecap: radix rounds over the buffer, filtered to the
        bin, then the keys above the cut and as many ties as fit."""
        (hist, wsum, var, akey, aidx, ekey, eidx, fkey, fidx) = sm
        TG = self.tg
        n_out = var.iterator + 3
        n_tie = var.iterator + 5
        rem = need
        pre = cutlass.Int32(0)  # the cut's unsigned key bits so far
        level = cutlass.Int32(0)
        done = cutlass.Int32(0)
        for r in cutlass.range_constexpr(4):
            sh = (22, 12, 2, 0)[r]
            width = (10, 10, 10, 2)[r]
            psh = (32, 22, 12, 2)[r]
            if done == 0:
                self.zero_hist(hist, lt)
                self.gsync(grp)
                for j in cutlass.range(lt, cnt, TG, unroll=1):
                    s = akey[j]
                    ok = s >= lo1
                    if any2:
                        ok = ok & (s < lo2)
                    u = s ^ SIGN
                    if cutlass.const_expr(r > 0):
                        ok = ok & (self.hi(u, psh) == pre)
                    if ok:
                        cute.arch.atomic_add(
                            hist.iterator + (self.hi(u, sh) & ((1 << width) - 1)),
                            cutlass.Int32(1), sem="relaxed", scope="cta")
                self.gsync(grp)
                c, above, cnt_c = self.cut(hist, wsum, var, rem, lt, lane, wig, grp)
                pre = (pre << width) | c
                rem = rem - above
                level = cutlass.Int32(r + 1)
                if cnt_c == rem:
                    done = cutlass.Int32(1)
        tie0 = a1 + need - rem
        for base in cutlass.range(0, cnt, TG, unroll=1):
            j = base + lt
            ok = j < cnt
            s = cutlass.Int32(SIGN)
            idx = cutlass.Int32(0)
            if ok:
                s = akey[j]
                idx = aidx[j]
                ok = s >= lo1
                if any2:
                    ok = ok & (s < lo2)
            u = s ^ SIGN
            h = u
            if level == 1:
                h = self.hi(u, 22)
            elif level == 2:
                h = self.hi(u, 12)
            elif level == 3:
                h = self.hi(u, 2)
            up = ok & (h > pre)
            tie = ok & (h == pre)
            if level == 4:  # whole keys compare signed, as keys
                up = ok & (s > (pre ^ SIGN))
            pos = self.wagg(up, n_out, lane)
            if up:
                out[(row, pos)] = idx
            pt = self.wagg(tie, n_tie, lane)
            if tie & (pt < rem):
                out[(row, tie0 + pt)] = idx

    # -- kernel ----------------------------------------------------------

    @cute.kernel
    def kernel(self, scores: cute.Tensor, lens: cute.Tensor, out: cute.Tensor,
               num_rows: cutlass.Int32):
        TG = self.tg
        U = self.unroll
        CAP = self.cap
        tidx, _, _ = cute.arch.thread_idx()
        bidx_, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        lane = cute.arch.lane_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        grp = warp // self.g
        wig = warp % self.g
        lt = tidx % TG

        smem = utils.SmemAllocator()
        st = smem.allocate(self.storage)

        def part(mr, size):
            return cute.make_tensor(mr.data_ptr() + grp * size, cute.make_layout(size))

        sm = (part(st.hist, NB), part(st.wsum, 32), part(st.var, 8),
              part(st.akey, CAP), part(st.aidx, CAP), part(st.ekey, self.ecap),
              part(st.eidx, self.ecap), part(st.fkey, self.fcap),
              part(st.fidx, self.fcap))
        (hist, wsum, var, akey, aidx, ekey, eidx, fkey, fidx) = sm

        gid = bidx_ * self.ngroups + grp
        for row in cutlass.range(gid, num_rows, gdim * self.ngroups, unroll=1):
            n = lens[row]
            if n > 0:
                # The row (n <= CAP) fills the buffer, blocks of B scores.
                B = U * TG
                self.zero_hist(hist, lt)
                self.gsync(grp)
                for j in cutlass.range_constexpr(CAP // B):
                    v = cute.make_rmem_tensor((U,), cutlass.Int32)
                    for k in cutlass.range_constexpr(U):
                        i = j * B + k * TG + lt
                        v[k] = cutlass.Int32(0)
                        if i < n:
                            v[k] = scores[(row, i)]
                    for k in cutlass.range_constexpr(U):
                        i = j * B + k * TG + lt
                        if i < n:
                            sk = self.skey(v[k])
                            akey[i] = sk
                            aidx[i] = i
                            cute.arch.atomic_add(
                                hist.iterator + self.coarse(sk), cutlass.Int32(1),
                                sem="relaxed", scope="cta")
                self.gsync(grp)
                self.select(sm, n, out, row, lt, lane, wig, grp)


@functools.cache
def _compiled(k: int, warps: int, group: int, cap: int, ecap: int, unroll: int,
              rank: int, grid: int):
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

    def sym():
        return cute.sym_int32()

    i32 = cutlass.Int32
    args = [
        make_fake_tensor(i32, (sym(), sym()), (sym(), 1), assumed_align=4),
        make_fake_tensor(i32, (sym(),), (1,), assumed_align=4),
        make_fake_tensor(i32, (sym(), k), (k, 1), assumed_align=4),
        cutlass.Int32(1),
    ]
    return cute.compile(IvfTopK(k, warps, group, cap, ecap, unroll, rank), *args, grid,
                        make_fake_stream(use_tvm_ffi_env_stream=False),
                        options="--enable-tvm-ffi")


def supported(scores: torch.Tensor, out: torch.Tensor, k: int) -> bool:
    return (scores.dtype == torch.float32 and scores.stride(1) == 1
            and scores.shape[0] * scores.stride(0) < 2**31
            and out.is_contiguous() and out.shape[1] == k)


def topk(*, scores: torch.Tensor, lens: torch.Tensor, out: torch.Tensor, k: int,
         num_sms: int, warps: int = 8, group: int = 8, cap: int = 4096, ecap: int = 512,
         unroll: int = 8, rank: int = 64, ctas_per_sm: int = 4) -> None:
    """Launch on the current stream (see the module docstring). Every row's
    length must be at most ``cap``."""
    fn = _compiled(k, warps, group, cap, ecap, unroll, rank, num_sms * ctas_per_sm)
    fn(scores.view(torch.int32), lens, out, scores.shape[0],
       cuda.CUstream(torch.cuda.current_stream().cuda_stream))
