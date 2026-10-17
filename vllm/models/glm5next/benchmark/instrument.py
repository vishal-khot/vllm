# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-layer DSA-indexer timing, peak memory, and resident-state probe.

Monkeypatches ``forward_cuda`` on both indexer classes so a run captures the
same metrics whichever arm is active:

- ``SparseAttnIndexerKpool`` (stock k-pool), and
- ``SparseAttnIndexerIVF`` (k-means IVF).

Keyed by layer index (parsed from the indexer's ``k_cache`` prefix) and phase
(prefill vs decode). Imported from ``sitecustomize`` when ``GLM5_INSTRUMENT=1``
so it also installs inside every TP worker. Each process flushes its own JSON to
``GLM5_TIMING_DIR`` on exit; the aggregator merges across ranks.

The indexer exists only on the 11 DSA layers (3, 7, ..., 43 for GLM-5.3-Flash);
no other layer is patched because no other layer has this op.

Env:
    GLM5_INSTRUMENT=1       enable (sitecustomize checks this).
    GLM5_TIMING_DIR=/path   where each process writes its JSON (required).
    GLM5_MEM_PROBE=1        also record indexer transient peak memory (perturbs
                            global peak stats via reset; run as its own pass).
    GLM5_RESIDENT_PROBE=1   also record resident indexer-state MiB per layer
                            (k_cache + cluster-id + centroid caches).
"""

import atexit
import json
import os
import re
import time

_LAYER_RE = re.compile(r"layers\.(\d+)")


def _tensor_bytes(x) -> int:
    """Total bytes of a tensor, or of all tensors in a (nested) list/tuple."""
    import torch

    if torch.is_tensor(x):
        return x.element_size() * x.numel()
    if isinstance(x, (list, tuple)):
        return sum(_tensor_bytes(y) for y in x)
    return 0


def _resident_bytes(indexer) -> int:
    """Steady-state bytes the indexer holds: its key / cluster / centroid caches.

    k-pool carries only the pooled fp8 ``k_cache``; IVF adds the per-request
    cluster-id map (``cid_cache``) and centroids (``cen_cache``).
    """
    total = 0
    for name in ("k_cache", "cid_cache", "cen_cache", "tail_cache"):
        cache = getattr(indexer, name, None)
        if cache is not None:
            total += _tensor_bytes(getattr(cache, "kv_cache", None))
    return total


def _indexer_classes():
    """Import whichever indexer classes are available on this build."""
    classes = []
    try:
        from vllm.models.glm5next.nvidia.sparse_indexer import (
            SparseAttnIndexerKpool,
        )

        classes.append(SparseAttnIndexerKpool)
    except Exception as e:  # pragma: no cover - import guard
        print(f"[glm5-instrument] no k-pool indexer: {e}", flush=True)
    try:
        from vllm.models.glm5next.nvidia.ivf_sparse_indexer import (
            SparseAttnIndexerIVF,
        )

        classes.append(SparseAttnIndexerIVF)
    except Exception as e:  # pragma: no cover - import guard
        print(f"[glm5-instrument] no IVF indexer: {e}", flush=True)
    return classes


def _install():
    import torch

    timing_dir = os.environ.get("GLM5_TIMING_DIR")
    if not timing_dir:
        print("[glm5-instrument] GLM5_TIMING_DIR unset; not installing", flush=True)
        return
    classes = _indexer_classes()
    if not classes:
        return
    os.makedirs(timing_dir, exist_ok=True)

    mem_probe = os.environ.get("GLM5_MEM_PROBE") == "1"
    resident_probe = os.environ.get("GLM5_RESIDENT_PROBE") == "1"

    # entries[layer][phase] = {n, ms_sum, tok_sum, peak_mib, resident_mib}
    entries: dict[int, dict[str, dict]] = {}
    last_flush = [0.0]
    pending: list = []  # (layer, phase, n_tok, start_event, end_event)

    def _layer_of(indexer) -> int:
        prefix = getattr(getattr(indexer, "k_cache", None), "prefix", "") or ""
        m = _LAYER_RE.search(str(prefix))
        return int(m.group(1)) if m else -1

    def _slot(layer, phase):
        return entries.setdefault(layer, {}).setdefault(
            phase,
            {"n": 0, "ms_sum": 0.0, "tok_sum": 0, "peak_mib": 0.0, "resident_mib": 0.0},
        )

    def _drain():
        if not pending:
            return
        torch.cuda.synchronize()
        for layer, phase, n_tok, start, end in pending:
            s = _slot(layer, phase)
            s["n"] += 1
            s["ms_sum"] += start.elapsed_time(end)
            s["tok_sum"] += n_tok
        pending.clear()

    def _make_timed(orig):
        def timed_forward_cuda(self, *args, **kw):
            layer = _layer_of(self)
            # args[0] is hidden_states for both indexer classes.
            n_tok = int(args[0].shape[0])
            # Decode carries ~one query token per running sequence (+MTP spec);
            # prefill carries the whole chunk. 8 covers single-stream decode.
            phase = "decode" if n_tok <= 8 else "prefill"

            if mem_probe:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                before = torch.cuda.memory_allocated()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = orig(self, *args, **kw)
            end.record()

            if mem_probe:
                torch.cuda.synchronize()
                s = _slot(layer, phase)
                s["n"] += 1
                s["ms_sum"] += start.elapsed_time(end)
                s["tok_sum"] += n_tok
                peak = (torch.cuda.max_memory_allocated() - before) / (1024**2)
                s["peak_mib"] = max(s["peak_mib"], peak)
            else:
                pending.append((layer, phase, n_tok, start, end))
                if len(pending) >= 2048:
                    _drain()

            if resident_probe:
                # Constant once allocated; record on the slot (cheap, no sync).
                mib = _resident_bytes(self) / (1024**2)
                if mib:
                    _slot(layer, phase)["resident_mib"] = mib

            now = time.time()
            if now - last_flush[0] > 5.0:
                _drain()
                _flush()
                last_flush[0] = now
            return out

        return timed_forward_cuda

    patched = []
    for cls in classes:
        cls.forward_cuda = _make_timed(cls.forward_cuda)
        patched.append(cls.__name__)

    def _flush():
        try:
            rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", -1)))
        except ValueError:
            rank = -1
        report = {
            "rank": rank,
            "pid": os.getpid(),
            "mem_probe": mem_probe,
            "resident_probe": resident_probe,
            "layers": {},
        }
        for layer in sorted(entries):
            report["layers"][str(layer)] = {}
            for phase, s in entries[layer].items():
                report["layers"][str(layer)][phase] = {
                    "calls": s["n"],
                    "mean_ms": s["ms_sum"] / s["n"] if s["n"] else 0.0,
                    "ms_per_tok": s["ms_sum"] / s["tok_sum"] if s["tok_sum"] else 0.0,
                    "peak_mib": round(s["peak_mib"], 2),
                    "resident_mib": round(s["resident_mib"], 2),
                }
        fn = os.path.join(timing_dir, f"indexer_timing_pid{os.getpid()}_r{rank}.json")
        with open(fn, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[glm5-instrument] wrote {fn}", flush=True)

    def _drain_and_flush():
        # atexit runs after stdout may be closed; never raise from here.
        try:
            _drain()
            _flush()
        except Exception:
            pass

    atexit.register(_drain_and_flush)
    print(
        f"[glm5-instrument] installed on {patched} "
        f"(mem_probe={mem_probe} resident_probe={resident_probe}) -> {timing_dir}",
        flush=True,
    )


_install()
