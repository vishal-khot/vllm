# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lost-attention measurement for the glm5next IVF (k-means) DSA indexer.

The IVF indexer probes ``num_probes`` of a request's clusters and selects the
top-k keys among them. Probing fewer clusters than exist can miss keys that the
exact top-k (every cluster probed) would have kept. This module quantifies that
loss: it re-runs the same selection with every cluster probed and measures what
fraction of the exact top-k keys the real P-probe selection recovered.

key recall = |selected ∩ exact| / |exact|, per query row.  1 - recall is the
attention lost to probing. The exact re-run reuses the identical scoring path,
so the fp8 precision cancels and only the probing approximation is measured.

Gated by ``VLLM_DSA_IVF_MEASURE_LOST_ATTENTION``; expensive (it ~doubles the
selection cost), so it is meant for select measurement runs, not production.
"""

from __future__ import annotations

import atexit
import json
import os
import time
from dataclasses import dataclass, field

import torch

from vllm import envs
from vllm.logger import init_logger

logger = init_logger(__name__)

# Log a running summary every this many measured forwards (per phase).
_LOG_EVERY = 50
# Rewrite the dump at most this often (seconds). The dump runs on the hot path,
# so it is throttled; it also means the file survives a worker kill, which skips
# atexit (TP workers are terminated, not exited, on serve shutdown).
_FLUSH_SECONDS = 5.0


@dataclass
class _Stat:
    """Running key-recall and attention-mass stats for one (layer, phase)."""

    rows: int = 0
    recall_sum: float = 0.0
    recall_min: float = 1.0
    mass_sum: float = 0.0
    mass_min: float = 1.0
    real_mass_sum: float = 0.0
    exact_mass_sum: float = 0.0

    def add(
        self,
        recall: torch.Tensor,
        mass: torch.Tensor,
        real_mass: torch.Tensor,
        exact_mass: torch.Tensor,
    ) -> None:
        n = recall.numel()
        if n == 0:
            return
        self.rows += n
        self.recall_sum += float(recall.sum())
        self.recall_min = min(self.recall_min, float(recall.min()))
        self.mass_sum += float(mass.sum())
        self.mass_min = min(self.mass_min, float(mass.min()))
        self.real_mass_sum += float(real_mass.sum())
        self.exact_mass_sum += float(exact_mass.sum())

    def summary(self) -> dict:
        n = self.rows
        recall_mean = self.recall_sum / n if n else 0.0
        mass_mean = self.mass_sum / n if n else 0.0
        return {
            "rows": n,
            "recall_mean": recall_mean,
            "recall_min": self.recall_min if n else 0.0,
            "key_lost_mean": 1.0 - recall_mean,
            "mass_captured_mean": mass_mean,
            "mass_captured_min": self.mass_min if n else 0.0,
            "mass_lost_mean": 1.0 - mass_mean,
            # Raw summed DSA-score mass each selection kept, for debugging the
            # captured ratio above.
            "real_mass_mean": self.real_mass_sum / n if n else 0.0,
            "exact_mass_mean": self.exact_mass_sum / n if n else 0.0,
        }


@dataclass
class _Recorder:
    stats: dict[tuple[str, str], _Stat] = field(default_factory=dict)
    _last_flush: float = 0.0

    def add(
        self,
        layer: str,
        phase: str,
        recall: torch.Tensor,
        mass: torch.Tensor,
        real_mass: torch.Tensor,
        exact_mass: torch.Tensor,
    ) -> None:
        key = (layer, phase)
        stat = self.stats.get(key)
        if stat is None:
            stat = self.stats[key] = _Stat()
        before = stat.rows
        stat.add(recall, mass, real_mass, exact_mass)
        if before // _LOG_EVERY != stat.rows // _LOG_EVERY:
            s = stat.summary()
            logger.info(
                "IVF lost-attention %s/%s: key_lost=%.4f mass_lost=%.4f "
                "recall_min=%.4f mass_min=%.4f rows=%d",
                layer,
                phase,
                s["key_lost_mean"],
                s["mass_lost_mean"],
                s["recall_min"],
                s["mass_captured_min"],
                s["rows"],
            )
        now = time.time()
        if now - self._last_flush > _FLUSH_SECONDS:
            self.dump()
            self._last_flush = now

    def dump(self) -> None:
        if not self.stats:
            return
        out = {
            f"{layer}|{phase}": stat.summary()
            for (layer, phase), stat in self.stats.items()
        }
        directory = envs.VLLM_DSA_IVF_LOST_ATTENTION_DIR or "."
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"lost_attention_pid{os.getpid()}.json")
        tmp = f"{path}.tmp{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(out, f, indent=2)
        os.replace(tmp, path)  # atomic, so a kill mid-write can't truncate it


_RECORDER: _Recorder | None = None


def enabled() -> bool:
    return envs.VLLM_DSA_IVF_MEASURE_LOST_ATTENTION


def _safe_dump() -> None:
    # atexit runs after logging/stdout may be closed; never raise from here.
    try:
        if _RECORDER is not None:
            _RECORDER.dump()
    except Exception:
        pass


def _recorder() -> _Recorder:
    global _RECORDER
    if _RECORDER is None:
        _RECORDER = _Recorder()
        atexit.register(_safe_dump)
    return _RECORDER


def key_recall(selected: torch.Tensor, exact: torch.Tensor) -> torch.Tensor:
    """Per-row |selected ∩ exact| / |exact| over top-k key indices.

    Both tensors are [rows, K] int32 key positions with unused slots set to -1.
    """
    rows = selected.shape[0]
    if rows == 0:
        return selected.new_empty((0,), dtype=torch.float32)
    # Shift by 1 so padding (-1) lands in a sentinel column 0 we then ignore.
    width = int(max(int(selected.max()), int(exact.max())) + 2)
    mark = torch.zeros((rows, width), dtype=torch.bool, device=selected.device)
    mark.scatter_(1, (exact + 1).long(), True)
    mark[:, 0] = False
    hit = mark.gather(1, (selected + 1).long())
    inter = hit.sum(dim=1).float()
    exact_cnt = (exact >= 0).sum(dim=1).clamp(min=1).float()
    return inter / exact_cnt


def mass_captured(real_mass: torch.Tensor, exact_mass: torch.Tensor) -> torch.Tensor:
    """Per-row retained DSA-score mass as a fraction of the exact top-k mass.

    ``exact`` probes every cluster, so its summed top-k score is the maximum
    attainable; the ratio is the attention mass the P-probe selection kept.
    """
    return (real_mass / exact_mass.clamp(min=1e-9)).clamp(max=1.0)


def record(
    layer: str,
    phase: str,
    selected: torch.Tensor,
    exact: torch.Tensor,
    real_mass: torch.Tensor,
    exact_mass: torch.Tensor,
) -> None:
    """Accumulate key recall and attention-mass captured vs the exact top-k."""
    _recorder().add(
        layer,
        phase,
        key_recall(selected, exact),
        mass_captured(real_mass, exact_mass),
        real_mass,
        exact_mass,
    )
