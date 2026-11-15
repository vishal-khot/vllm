# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E-means centroids for the IVF indexer's full builds.

E-means (Bessa et al., "E-Means: Scaling Laws for Mini-Batch K-Means",
arXiv 2611.09501; reference code: faiss/contrib/e_means, MIT) is mini-batch
k-means with an EMA update whose batch size and step come from fitted scaling
laws. Per cluster j it keeps a mass N_j and a position c_j = S_j / N_j. Each
batch assigns its rows to the nearest unit direction of c, collects per-cluster
counts n_j and sums s_j, and with alpha = alpha0 cosine-annealed to alpha0 / 10:

    N_j <- (1 - alpha) N_j + alpha n_j
    c_j <- c_j + (alpha n_j / N_j) (s_j / n_j - c_j)      (clusters with n_j > 0)

Every 5 batches a cluster with N_j < 0.1 * mean(N) is revived by splitting a
donor drawn in proportion to N (Gumbel top-k): c(1 -/+ 1e-4 * [1, -1, ...]),
masses halved. Every pass visits the rows in a fresh random order; the closing
centroids are the unit means of the last pass's batch assignments. This module
mirrors the reference's spherical torch path (``Emeans(spherical=True)``,
throughput batch law since d = 128 > 16): constants, schedule, revival and
closing.

The IVF state is written as a warm build expects it: ``emeans_seed_centroids``
stores the centroids in the request's records, and the fused k-means kernel,
run with no iteration and every item marked warm, then assigns every key and
writes cluster ids, sizes and page counts. ``emeans_store_lengths`` afterwards
records each centroid's member-sum length, which appends fold new keys into.
Rows are the scaled keys (fp8 x ue8m0 scale, exact in bf16), as the fused
kernel's sums use them.
"""

from __future__ import annotations

import math

import torch

from vllm.models.glm5next.nvidia.ops.ivf_index import (
    _key_page,
    _load_centroids,
    _load_key_scales,
    _load_keys,
    _rec,
)
from vllm.triton_utils import tl, triton

HEAD_DIM = 128
ALPHA_MIN_FACTOR = 0.1  # cosine floor: alpha ends at alpha0 / 10
CHECK_UNUSED_EVERY = 5  # batches between revival checks
INIT_USAGE = 2.0**-16  # initial mass N_j (the paper uses 1e-10)
SPLIT_EPSILON = 1e-4  # relative perturbation of a split's two children
UNUSED_THRESHOLD = 0.1  # revive N_j < tau * mean(N)
# B = C * k^a * n^b * T^c (reference "throughput" law, used for d > 16).
BATCH_LAW = (0.035134, 0.134749, 0.876473, 0.813344)
# alpha0 = C * k^a * B^b * n^c * T^d, clamped to [1e-4, 1].
ALPHA_LAW = (0.8244, -0.4617, 0.9003, -0.4498, -0.1057)


def _round_power_of_two(value: float) -> int:
    if value <= 1:
        return 1
    low = 2 ** int(math.floor(math.log2(value)))
    return low if value / low < 2 * low / value else 2 * low


def recipe(n: int, k: int, nepochs: float) -> dict:
    """Batch size, initial step and per-pass rows for ``n`` rows into ``k``."""
    c, bk, bn, bt = BATCH_LAW
    batch = _round_power_of_two(c * k**bk * n**bn * nepochs**bt)
    batch = max(1, min(batch, _round_power_of_two(n / 2), n))
    ca, ak, ab, an, at = ALPHA_LAW
    alpha0 = min(max(ca * k**ak * batch**ab * n**an * nepochs**at, 1e-4), 1.0)
    whole = math.floor(nepochs)
    rem = int(round((nepochs - whole) * n))
    rows = [rem] if whole == 0 else [n] * (whole - 1) + ([rem] if rem else []) + [n]
    steps = sum(-(-r // batch) for r in rows)
    return {"batch": batch, "alpha0": alpha0, "epoch_rows": rows, "total_steps": steps}


def _alpha(alpha0: float, step: int, total: int) -> float:
    frac = min(step, total) / max(1, total)
    cos = 0.5 * (1.0 + math.cos(math.pi * frac))
    return alpha0 * (ALPHA_MIN_FACTOR + (1.0 - ALPHA_MIN_FACTOR) * cos)


def _generator(device: torch.device, seed: int) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(seed % 2**63)
    return g


def _unit(x: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(x, dim=1, keepdim=True)
    return x / torch.where(norm > 0, norm, 1)


def _ema(c, usage, sums, counts, alpha: float) -> None:
    """One EMA step on positions ``c`` and masses ``usage`` (in place)."""
    replace = alpha == 1.0
    touched = counts > 0
    mean = (
        sums.masked_fill(~touched[:, None], 0)
        / torch.where(touched, counts, 1)[:, None]
    )
    if not replace:
        usage.mul_(1.0 - alpha)
    retained = torch.zeros_like(usage) if replace else usage
    mass = retained + alpha * counts
    weight = alpha * counts / torch.where(touched, mass, 1)
    pos = (mean - c) * weight[:, None] + c
    pos = torch.where((weight == 1)[:, None], mean, pos)
    c.copy_(torch.where(touched[:, None], pos, c))
    usage.copy_(torch.where(touched, mass, usage))


def _revive(c, usage, rng: torch.Generator) -> int:
    """Split donors into clusters whose mass fell below tau * mean mass."""
    k = usage.numel()
    unused = usage < usage.sum() * UNUSED_THRESHOLD / k
    donors = (~unused) & (usage > 0)
    n_unused, n_donors = torch.stack((unused.sum(), donors.sum())).tolist()
    nsplit = min(n_unused, n_donors)
    if nsplit == 0:
        return 0
    uniform = torch.rand(k, dtype=usage.dtype, device=usage.device, generator=rng)
    uniform = uniform.clamp_min(torch.finfo(usage.dtype).tiny)
    keys = torch.where(donors, usage.log() - (-uniform.log()).log(), float("-inf"))
    source = torch.topk(keys, nsplit).indices
    target = torch.sort(unused.byte(), descending=True, stable=True).indices[:nsplit]
    donor = c[source]
    sign = torch.ones(c.shape[1], dtype=c.dtype, device=c.device)
    sign[1::2] = -1.0
    offset = SPLIT_EPSILON * sign
    half = usage[source] / 2.0
    usage[source] = half
    c[source] = donor * (1.0 - offset)
    usage[target] = half
    c[target] = donor * (1.0 + offset)
    return nsplit


def emeans_spherical(
    x: torch.Tensor,
    k: int,
    nepochs: float,
    seed: int,
    init: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Spherical E-means of rows ``x [n, d]``.

    Returns unit centroids ``[k, d]`` fp32, the last pass's member-sum lengths
    ``[k]`` fp32 and run statistics. ``init`` (any scale) replaces the default
    initialization, ``k`` rows drawn without replacement.
    """
    n, d = x.shape
    dev = x.device
    r = recipe(n, k, nepochs)
    if init is None:
        idx = (
            torch.arange(n, device=dev)
            if n == k
            else torch.randperm(n, generator=_generator(dev, seed), device=dev)[:k]
        )
        init = x[idx]
    c = _unit(init.float())
    usage = torch.full((k,), INIT_USAGE, dtype=torch.float32, device=dev)
    split_rng = _generator(dev, seed)
    step = splits = 0
    batch = r["batch"]
    last = len(r["epoch_rows"]) - 1
    pass_sums = pass_counts = None
    for epoch, rows in enumerate(r["epoch_rows"]):
        order = torch.randperm(
            n, generator=_generator(dev, (seed * 1000003 + epoch) % 2**32), device=dev
        )
        if epoch == last:
            pass_sums = torch.zeros((k, d), dtype=torch.float32, device=dev)
            pass_counts = torch.zeros((k,), dtype=torch.float32, device=dev)
        for b0 in range(0, rows, batch):
            xb = x[order[b0 : min(b0 + batch, rows)]].float()
            labels = torch.argmax(xb @ _unit(c).T, dim=1)
            sums = torch.zeros((k, d), dtype=torch.float32, device=dev)
            sums.index_add_(0, labels, xb)
            counts = torch.bincount(labels, minlength=k).float()
            if epoch == last:
                pass_sums += sums
                pass_counts += counts
            _ema(c, usage, sums, counts, _alpha(r["alpha0"], step, r["total_steps"]))
            if (step + 1) % CHECK_UNUSED_EVERY == 0:
                splits += _revive(c, usage, split_rng)
            step += 1
    mean = pass_sums / pass_counts.clamp_min(1)[:, None]
    mean = torch.where((pass_counts > 0)[:, None], mean, c)
    stats = {**r, "steps": step, "splits": splits}
    return _unit(mean), torch.linalg.vector_norm(pass_sums, dim=1), stats


@triton.jit
def ivf_gather_keys_kernel(
    k_u8_ptr,
    k_f32_ptr,
    k_page_stride,
    k_bt_ptr,
    k_bt_stride,
    b,
    n_keys,
    out_ptr,
    BLOCK: tl.constexpr,
    PAGE: tl.constexpr,
):
    """Scaled keys ``[n_keys, 128]`` bf16 of batch row ``b``, in position order."""
    pos = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = pos < n_keys
    page = _key_page(k_bt_ptr, k_bt_stride, b, pos, PAGE, mask)
    k = _load_keys(k_u8_ptr, k_page_stride, page, pos, PAGE, mask).to(tl.float32)
    s = _load_key_scales(k_f32_ptr, k_page_stride, page, pos, PAGE, mask)
    offs_d = tl.arange(0, 128)
    tl.store(
        out_ptr + pos.to(tl.int64)[:, None] * 128 + offs_d[None, :],
        (k * s[:, None]).to(tl.bfloat16),
        mask=mask[:, None],
    )


@triton.jit
def ivf_centroid_records_kernel(
    cen_ptr,
    cen_i32_ptr,
    cen_stride,
    cen_bt_ptr,
    cen_bt_stride,
    cpb,
    b,
    k,
    unit_ptr,
    len_ptr,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Centroid records of batch row ``b``, clusters ``[0, k)``.

    MODE 0 reads the bf16 directions into ``unit [k, 128]`` fp32; MODE 1
    writes ``unit`` as the directions; MODE 2 writes ``len [k]`` as the
    member-sum lengths.
    """
    c = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = c < k
    rec = _rec(cen_bt_ptr, cen_bt_stride, b, c, cpb, cen_stride, mask)
    offs_d = tl.arange(0, 128)
    rows = c.to(tl.int64)[:, None] * 128 + offs_d[None, :]
    if MODE == 0:
        tl.store(
            unit_ptr + rows,
            _load_centroids(cen_ptr, rec, mask).to(tl.float32),
            mask=mask[:, None],
        )
    elif MODE == 1:
        x = tl.load(unit_ptr + rows, mask=mask[:, None], other=0.0)
        tl.store(
            cen_ptr + (rec // 2)[:, None] + offs_d[None, :],
            x.to(tl.bfloat16),
            mask=mask[:, None],
        )
    else:
        length = tl.load(len_ptr + c, mask=mask, other=0.0)
        tl.store(
            cen_i32_ptr + (rec + 260) // 4, length.to(tl.int32, bitcast=True), mask=mask
        )


def gather_keys(key, b: int, n: int) -> torch.Tensor:
    out = torch.empty((n, HEAD_DIM), dtype=torch.bfloat16, device=key.u8.device)
    block = 64
    ivf_gather_keys_kernel[(triton.cdiv(n, block),)](
        key.u8,
        key.f32,
        key.page_stride,
        key.block_table,
        key.block_table.stride(0),
        b,
        n,
        out,
        BLOCK=block,
        PAGE=key.page,
    )
    return out


def _records(state, b: int, k: int, unit, length, mode: int) -> None:
    block = 32
    ivf_centroid_records_kernel[(triton.cdiv(k, block),)](
        *state.cen_args(), b, k, unit, length, MODE=mode, BLOCK=block
    )


def emeans_seed_centroids(
    *,
    key,
    state,
    batch_rows: list[int],
    n_keys: list[int],
    warm: list[bool],
    clusters: list[int],
    epochs: float,
    warm_epochs: float,
    seed: int,
) -> list[tuple[int, int, torch.Tensor]]:
    """Fit and store each full build's centroids; return (row, k, lengths).

    A warm item starts from its stored centroids, a cold one from k of its
    keys. Lengths are stored after the assigning kernel, which zeroes them.
    """
    pending = []
    for b, n, w, k in zip(batch_rows, n_keys, warm, clusters):
        x = gather_keys(key, b, n)
        init = None
        if w:
            init = torch.empty((k, HEAD_DIM), dtype=torch.float32, device=x.device)
            _records(state, b, k, init, init, mode=0)
        unit, length, _ = emeans_spherical(
            x, k, warm_epochs if w else epochs, seed, init
        )
        _records(state, b, k, unit.contiguous(), length, mode=1)
        pending.append((b, k, length.contiguous()))
    return pending


def emeans_store_lengths(state, pending: list[tuple[int, int, torch.Tensor]]) -> None:
    for b, k, length in pending:
        _records(state, b, k, length, length, mode=2)
