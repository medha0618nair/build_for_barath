"""Per-field, per-value frequencies (u) — TECHNICAL_SPEC.md §4.5.

`u` is the population frequency of a value: how often two unrelated cases
agree on it. It's computed within the relevant pool:

- same-type pairs use frequencies computed within that crime_type,
- same-family mo_ext fields (both burglary types share mo_ext vocabulary,
  see linkage/schema.py) are pooled across the family,
- cross-type mo_core fields are pooled across all ingested property crime.

Sentinel tokens (__MISSING__, __ABSENT__, __UNKNOWABLE__) are excluded from
both the count and the denominator — a value that was never recorded says
nothing about its population frequency.

Computed locally over the normalised corpus via pandas/plain Python. The
equivalent Athena SQL (for the real pipeline, where cases live in S3/Glue,
not a local parquet) is infra/sql/frequencies.sql.
"""
from __future__ import annotations

from collections import Counter, defaultdict

from linkage import schema
from linkage.features import SENTINELS


def _value_counts(cases: list[dict], field: str, section: str) -> tuple[Counter, int]:
    """Counts categorical values, or tag presence for TAG_FIELDS. Returns
    (counts, n_recorded), n_recorded excluding sentinel tokens."""
    counts: Counter = Counter()
    n = 0
    tag = field in schema.TAG_FIELDS
    for case in cases:
        value = case[section].get(field)
        if value is None or (isinstance(value, str) and value in SENTINELS):
            continue
        n += 1
        if tag:
            counts.update(value)
        else:
            counts[value] += 1
    return counts, n


def field_frequencies(cases: list[dict], field: str, section: str) -> dict[str, float]:
    counts, n = _value_counts(cases, field, section)
    if n == 0:
        return {}
    return {v: c / n for v, c in counts.items()}


def compute(cases: list[dict]) -> dict:
    """{'same_type': {crime_type: {field: {value: u}}},
        'same_family': {family: {field: {value: u}}},
        'cross_type': {field: {value: u}}}"""
    by_type: dict[str, list[dict]] = defaultdict(list)
    by_family: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        by_type[case["crime_type"]].append(case)
        by_family[schema.FAMILY[case["crime_type"]]].append(case)

    same_type = {
        ct: {f: field_frequencies(by_type.get(ct, []), f, "mo_core") for f in schema.MO_CORE}
        for ct in schema.CRIME_TYPES
    }
    same_family = {
        family: {f: field_frequencies(by_family.get(family, []), f, "mo_ext") for f in fields}
        for family, fields in schema.MO_EXT.items()
    }
    cross_type = {f: field_frequencies(cases, f, "mo_core") for f in schema.MO_CORE}

    return {"same_type": same_type, "same_family": same_family, "cross_type": cross_type}
