# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correct the lost-attention measurement for the fidelity sweep.

A row at or under top-k keeps every candidate, so the indexer never scores it. Its
``score_sum_out`` then reads 0, and any row under budget reports all of its mass
lost. Scoring those rows too fixes that without changing the selection: they keep
their candidates by count, and the top-k and emit steps never read their scores.
No kernel changes; imported from ``sitecustomize`` in the fidelity serve only.

A row with nothing to lose (no exact keys, or no exact mass) counts as nothing
lost, not as everything lost.
"""

import torch

from vllm.models.glm5next.nvidia import ivf_fidelity, ivf_sparse_indexer

_score_per_row = ivf_sparse_indexer._score_per_row
_score_grouped_gemm = ivf_sparse_indexer._score_grouped_gemm
_key_recall = ivf_fidelity.key_recall
_mass_captured = ivf_fidelity.mass_captured


def score_every_row(**kwargs) -> None:
    _score_per_row(**{**kwargs, "topk": 0})


def score_every_group(*, topk_lens: torch.Tensor, **kwargs) -> None:
    _score_grouped_gemm(topk_lens=torch.ones_like(topk_lens), **kwargs)


def key_recall(selected: torch.Tensor, exact: torch.Tensor) -> torch.Tensor:
    recall = _key_recall(selected, exact)
    return torch.where((exact >= 0).any(dim=1), recall, recall.new_ones(()))


def mass_captured(real_mass: torch.Tensor, exact_mass: torch.Tensor) -> torch.Tensor:
    ratio = _mass_captured(real_mass, exact_mass)
    return torch.where(exact_mass > 0, ratio, ratio.new_ones(()))


ivf_sparse_indexer._score_per_row = score_every_row
ivf_sparse_indexer._score_grouped_gemm = score_every_group
ivf_fidelity.key_recall = key_recall
ivf_fidelity.mass_captured = mass_captured
print("[glm5-fidelity] measurement patch installed", flush=True)
