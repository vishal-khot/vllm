# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RULER task definitions and recall scoring for the DSA-indexer benchmark.

Each task scores by recall: the fraction of gold answer strings that appear
(case-insensitive) in the generation. Single-answer NIAH/QA tasks collapse to
hit/miss; aggregation tasks (cwe/fwe/vt/multivalue/multiquery) get partial
recall, matching RULER's ``string_match_all``.
"""

# max_new_tokens per task. GLM-5.3-Flash reasoning is always-on and consumes the
# budget before the answer, so caps are generous. Overridable per run.
TASKS = {
    "niah_single_1": {"group": "niah", "max_tokens": 1024},
    "niah_single_2": {"group": "niah", "max_tokens": 1024},
    "niah_single_3": {"group": "niah", "max_tokens": 1024},
    "niah_multikey_1": {"group": "niah", "max_tokens": 1024},
    "niah_multikey_2": {"group": "niah", "max_tokens": 1024},
    "niah_multikey_3": {"group": "niah", "max_tokens": 1024},
    "niah_multivalue": {"group": "niah", "max_tokens": 2048},
    "niah_multiquery": {"group": "niah", "max_tokens": 2048},
    "vt": {"group": "track", "max_tokens": 2048},
    "cwe": {"group": "aggr", "max_tokens": 2048},
    "fwe": {"group": "aggr", "max_tokens": 2048},
    "qa_1": {"group": "qa", "max_tokens": 2048},
    "qa_2": {"group": "qa", "max_tokens": 2048},
}

ALL_TASKS = tuple(TASKS)


def recall(gold, text) -> float:
    """Fraction of gold strings present in text (case-insensitive)."""
    golds = gold if isinstance(gold, list) else [gold]
    if not golds:
        return 0.0
    low = text.lower()
    hit = sum(1 for g in golds if str(g).lower() in low)
    return hit / len(golds)
