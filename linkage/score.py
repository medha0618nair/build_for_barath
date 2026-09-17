"""Pair scoring: total bits = prior + sum of evidence weights (TECHNICAL_SPEC.md §4).

    python -m linkage.score --worked-example
        Reproduces TECHNICAL_SPEC.md §4.3's burglary example field by field.

    python -m linkage.score --pair <case_id_a> <case_id_b> [--scope same|all]
        Scores a real pair from data/final/, same as the batch linker will.

Same-type pairs (same crime_type) score mo_core + mo_ext and use that
crime_type's own prior. Cross-type pairs (different crime_type) use the
cross-type prior. mo_ext is scored whenever the two crime types share the
same MO_EXT vocabulary (schema.FAMILY) — both burglary types, even when
`--scope` puts them in the cross-type prior bucket (see linkage/schema.py's
comment: "both burglary types score against each other on the full
burglary field set").
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from math import log2
from pathlib import Path

import pandas as pd

from linkage import config as config_mod
from linkage import features, frequencies as frequencies_mod, prior as prior_mod, schema
from linkage.features import SENTINELS, Contribution
from linkage.normalise import normalise_all

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "final"


def _score_fields(fields: tuple[str, ...], section: str, case_a: dict, case_b: dict,
                   u_pool: dict, alpha: float) -> list[Contribution]:
    prov_a, prov_b = case_a["field_provenance"], case_b["field_provenance"]
    out: list[Contribution] = []
    for f in fields:
        va, vb = case_a[section].get(f), case_b[section].get(f)
        u_dist = u_pool.get(f, {})
        if f in schema.TAG_FIELDS:
            out += features.score_tag(f, va, vb, u_dist, alpha, prov_a.get(f), prov_b.get(f))
        else:
            c = features.score_categorical(f, va, vb, u_dist, alpha, prov_a.get(f), prov_b.get(f))
            if c is not None:
                out.append(c)
    return out


def score_pair(case_a: dict, case_b: dict, freqs: dict, priors: dict,
               alpha_same: float, alpha_cross: float) -> dict:
    ct_a, ct_b = case_a["crime_type"], case_b["crime_type"]
    same_type = ct_a == ct_b
    pair_class = "same_type" if same_type else "cross_type"
    alpha = alpha_same if same_type else alpha_cross

    u_core = freqs["same_type"][ct_a] if same_type else freqs["cross_type"]
    contributions = _score_fields(schema.MO_CORE, "mo_core", case_a, case_b, u_core, alpha)

    if schema.FAMILY[ct_a] == schema.FAMILY[ct_b]:
        family = schema.FAMILY[ct_a]
        u_ext = freqs["same_family"][family]
        contributions += _score_fields(schema.MO_EXT[family], "mo_ext", case_a, case_b, u_ext, alpha)

    contributions.sort(key=lambda c: -abs(c.bits))
    prior_bits = priors["same_type"].get(ct_a) if same_type else priors["cross_type"]
    total = (prior_bits or 0.0) + sum(c.bits for c in contributions)
    return {"pair_class": pair_class, "prior_bits": prior_bits, "total_bits": total,
            "contributions": contributions}


# --- fitting alpha for cross-type pairs (TECHNICAL_SPEC.md §4.5) -----------

def fit_cross_type_repeat_rate(cases: list[dict], truth: pd.DataFrame) -> float | None:
    """P(agree | same offender, different crime_type) on mo_core's
    categorical fields, measured from labelled cross-type pairs — the only
    kind of "fit" possible before Phase 3's retrieval/training exists.
    Tag fields are skipped; the categorical fields alone are enough to fit
    one alpha, and tags need a different (per-tag) aggregation.
    """
    offender = truth.set_index("case_id")["offender_id"]
    by_offender: dict = defaultdict(list)
    for case in cases:
        off = offender.get(case["case_id"])
        if off is not None and pd.notna(off):
            by_offender[off].append(case)

    categorical_core = [f for f in schema.MO_CORE if f not in schema.TAG_FIELDS]
    agree, total = 0, 0
    for group in by_offender.values():
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if a["crime_type"] == b["crime_type"]:
                    continue
                for f in categorical_core:
                    va, vb = a["mo_core"].get(f), b["mo_core"].get(f)
                    if va is None or vb is None or va in SENTINELS or vb in SENTINELS:
                        continue
                    total += 1
                    agree += va == vb
    return agree / total if total else None


# --- worked example (TECHNICAL_SPEC.md §4.3) --------------------------------

# (field, value_a==value_b or None if disagreeing, u, spec_bits, is_tag)
WORKED_EXAMPLE_ALPHA = 7.05
WORKED_EXAMPLE_ROWS = [
    ("entry_point", "roof", 0.06, 1.559, False),
    ("entry_method", "lock_broken", 0.45, 0.204, False),
    ("premise", "independent_house", 0.42, 0.228, False),
    ("occupancy", None, 0.58, -0.191, False),       # *disagrees* — table gives the field-aggregate u/m
    ("time_band", "night", 0.52, 0.157, False),
    ("search_pattern", "selective", 0.28, 0.400, False),
    ("counter_forensic", "cctv_disabled", 0.03, 2.327, False),
    ("group_size_est", "2-3", 0.48, 0.182, False),
    ("tools", "cutter", 0.16, 0.724, True),
    ("property_taken", "gold", 0.61, 0.110, True),
    ("property_taken", "cash", 0.54, 0.145, True),
]


def worked_example() -> list[dict]:
    rows = []
    for field, value, u, spec_bits, is_tag in WORKED_EXAMPLE_ROWS:
        m = features.m_of(u, WORKED_EXAMPLE_ALPHA)
        if value is None:
            bits = features.disagreement_bits(m, u)
            display_value = "*disagrees*"
        else:
            bits = features.agreement_bits(m, u)
            display_value = value
        rows.append({"field": field, "value": display_value, "u": u, "m": round(m, 3),
                     "my_bits": bits, "spec_bits": spec_bits, "diff": bits - spec_bits})
    return rows


def print_worked_example() -> bool:
    rows = worked_example()
    print(f"{'field':<18} {'value':<20} {'u':>6} {'m':>6} {'my bits':>9} {'spec bits':>10} {'diff':>8}")
    for r in rows:
        print(f"{r['field']:<18} {r['value']:<20} {r['u']:>6.2f} {r['m']:>6.3f} "
              f"{r['my_bits']:>+9.3f} {r['spec_bits']:>+10.3f} {r['diff']:>+8.4f}")
    my_total = sum(r["my_bits"] for r in rows)
    spec_total = 5.845
    print(f"{'TOTAL':<18} {'':<20} {'':>6} {'':>6} {my_total:>+9.3f} {spec_total:>+10.3f} "
          f"{my_total - spec_total:>+8.4f}")
    ok = all(abs(r["diff"]) <= 0.05 for r in rows) and abs(my_total - spec_total) <= 0.05
    print(f"\n{'PASS' if ok else 'FAIL'} (within 0.05 bits per row and total)")
    return ok


# --- CLI ----------------------------------------------------------------------

def _load_context(data_dir: Path, cfg: dict):
    from linkage.extract import LocalStubExtractor
    cases = normalise_all(cfg, data_dir, LocalStubExtractor())
    truth = pd.read_parquet(data_dir / "truth.parquet")
    freqs = frequencies_mod.compute(cases)
    priors = prior_mod.compute(truth[truth["ingested"]])
    rr = cfg["corpus"]["repeat_rate"]
    alpha_same = config_mod.alpha(rr["value"], rr["reference_frequency"])
    cross_rr = fit_cross_type_repeat_rate(cases, truth)
    alpha_cross = (config_mod.alpha(cross_rr, rr["reference_frequency"])
                   if cross_rr and cross_rr > rr["reference_frequency"] else alpha_same)
    return cases, freqs, priors, alpha_same, alpha_cross, cross_rr


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m linkage.score", description=__doc__.splitlines()[0])
    ap.add_argument("--worked-example", action="store_true")
    ap.add_argument("--pair", nargs=2, metavar=("CASE_A", "CASE_B"))
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--config-dir", type=Path, default=config_mod.CONFIG_DIR)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    if args.worked_example:
        return 0 if print_worked_example() else 1

    if args.pair:
        cfg = config_mod.load(args.config_dir)
        cases, freqs, priors, alpha_same, alpha_cross, cross_rr = _load_context(args.data_dir, cfg)
        by_id = {c["case_id"]: c for c in cases}
        a_id, b_id = args.pair
        if a_id not in by_id or b_id not in by_id:
            print(f"case not found: {a_id if a_id not in by_id else b_id}")
            return 1
        print(f"alpha_same={alpha_same:.3f}  alpha_cross={alpha_cross:.3f} "
              f"(fit from cross-type repeat rate {cross_rr})\n")
        result = score_pair(by_id[a_id], by_id[b_id], freqs, priors, alpha_same, alpha_cross)
        print(f"pair_class={result['pair_class']}  prior={result['prior_bits']:+.3f} bits")
        for c in result["contributions"]:
            print(f"  {c.field:<28} {str(c.value_a):<18} {str(c.value_b):<18} "
                  f"u={c.u:.3f} bits={c.bits:+.3f}")
        print(f"\nTOTAL {result['total_bits']:+.3f} bits")
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
