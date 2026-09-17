"""Prior bits, derived from pool composition (TECHNICAL_SPEC.md §4.4).

    true pairs = sum over offenders of C(their case count, 2)
    all pairs  = C(pool size, 2)
    prior      = log2(true_pairs / (all_pairs - true_pairs))

This uses ground-truth offender_id from the synthetic corpus — a
measurement only possible because this data is synthetic (CLAUDE.md
"Honesty requirements"; TECHNICAL_SPEC.md §11.4). A real deployment has no
such labels; there the prior is a stated modelling assumption; it does not
fall out of a query the way it does here.

Mirrors linkage/generate/report.py's pair_stats (which produced the numbers
in DATASET.md and manifest.json) but returns just the priors the scorer
needs, keyed by pair_class.
"""
from __future__ import annotations

from math import comb, log2

import pandas as pd

from linkage import schema


def _bits(true_pairs: int, total_pairs: int) -> float | None:
    if not 0 < true_pairs < total_pairs:
        return None
    return log2(true_pairs / (total_pairs - true_pairs))


def compute(truth: pd.DataFrame) -> dict:
    """truth: one row per ingested case, with offender_id, is_serial, crime_type.

    Returns {'same_type': {crime_type: bits}, 'cross_type': bits, 'all_property': bits}.
    """
    ing = truth
    ser = ing[ing["is_serial"]]
    n_type = ing["crime_type"].value_counts()
    by_offender = ser.groupby("offender_id").size()
    by_offender_type = ser.groupby(["offender_id", "crime_type"]).size()

    true_all = int(sum(comb(int(n), 2) for n in by_offender))
    all_pairs = comb(len(ing), 2)

    same_type: dict[str, float | None] = {}
    same_true_total = 0
    for ct in schema.CRIME_TYPES:
        n = int(n_type.get(ct, 0))
        sizes = (by_offender_type.xs(ct, level=1)
                 if ct in by_offender_type.index.get_level_values(1) else [])
        t = int(sum(comb(int(k), 2) for k in sizes))
        same_true_total += t
        same_type[ct] = _bits(t, comb(n, 2))

    cross_true = true_all - same_true_total
    cross_pairs = all_pairs - sum(comb(int(n), 2) for n in n_type)

    return {
        "same_type": same_type,
        "cross_type": _bits(cross_true, cross_pairs),
        "all_property": _bits(true_all, all_pairs),
    }
