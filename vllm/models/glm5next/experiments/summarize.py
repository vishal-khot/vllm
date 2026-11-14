# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Roll the reorder records up into ``<reorder>/summary.json``.

One row per (kind, length, probe share, rows): the median over inputs (prompts
x layers) of each step's median ms, the share of inputs where the reorder pays
(a + b < c) and where IVF's scoring beats kpool's (a + b < d), the median
break-even row count, and the lowest top-k agreement between b and c.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics


def _med(v):
    v = [x for x in v if x is not None]
    return round(statistics.median(v), 4) if v else None


def summarize_reorder(root: str) -> dict:
    recs = []
    for path in sorted(glob.glob(os.path.join(root, "results", "*.jsonl"))):
        with open(path) as f:
            for line in f:
                try:
                    recs.append(json.loads(line))
                except ValueError:
                    pass
    groups: dict[tuple, list] = {}
    for r in recs:
        groups.setdefault(
            (r["kind"], r["length"], r["probe_share"], r["rows"]), []).append(r)
    rows = []
    for (kind, length, share, n_rows), rs in sorted(groups.items()):
        pays = [r["reorder_pays"] for r in rs if "reorder_pays" in r]
        beats = [r["ivf_beats_kpool_scoring"] for r in rs]
        matches = [r["match"] for r in rs if "match" in r]
        rows.append({
            "kind": kind, "length": length, "probe_share": share, "rows": n_rows,
            "inputs": len(rs),
            "a_reorder_ms": _med([r["a_reorder_ms"] for r in rs]),
            "b_contiguous_ms": _med([r["b_contiguous_ms"] for r in rs]),
            "a_plus_b_ms": _med([r["a_reorder_ms"] + r["b_contiguous_ms"]
                                 for r in rs if r["b_contiguous_ms"] is not None]),
            "c_scattered_ms": _med([r["c_scattered_ms"] for r in rs]),
            "d_kpool_ms": _med([r["d_kpool_ms"] for r in rs]),
            "reorder_pays_share": (round(sum(pays) / len(pays), 3) if pays else None),
            "ivf_beats_kpool_share": round(sum(beats) / len(beats), 3),
            "breakeven_rows": _med([r.get("breakeven_rows") for r in rs]),
            "min_match": min(matches) if matches else None,
        })
    out = {"records": len(recs), "rows": rows}
    with open(os.path.join(root, "summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reorder", required=True, help="the reorder output root")
    args = p.parse_args()
    out = summarize_reorder(args.reorder)
    print(f"{out['records']} records -> {os.path.join(args.reorder, 'summary.json')}")


if __name__ == "__main__":
    main()
