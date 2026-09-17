"""Pick τ by measurement: python -m linkage.generate.tau

For each τ on the grid and each seed, generate truth and the recorded table
(no rendering) and apply validators 2 and 3. The rule, fixed before any
sweep ran: choose the SMALLEST τ at which both pass on EVERY seed — enough
cross-field correlation to break the scorer's independence assumption, and
enough of it surviving corruption, at the least marginal drift.

Writes config/tau_selection.json so the chosen value has its reason attached.
Copy the value into corpus.yaml by hand.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone

import numpy as np

from linkage import config as config_mod
from linkage import schema
from linkage.generate import corrupt, sample, validate
from linkage.generate import rng as streams

GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
SEEDS = [7, 11, 23]


def expected_drift(cfg: dict, tau: float, n: int = 20000) -> tuple[float, str]:
    """Jensen drift of serial crimes' expected marginals, free of sampling noise.

    E over s ~ N(0, sd²) of each field's tilted q, weighted by the offender's
    propensity for the crime type (q_ct(s)[type]), against config p. Largest
    relative drift among values with p >= 0.02; implied fields skipped.
    """
    model = sample.build_model(cfg)
    styles = np.random.default_rng(0).normal(0.0, cfg["corpus"]["style"]["sd"], (n, len(sample.AXES)))
    type_weight = model.crime_type.tilt(styles, tau)
    worst, name = 0.0, None
    for j, ct in enumerate(schema.CRIME_TYPES):
        w = type_weight[:, j] / type_weight[:, j].sum()
        for f in schema.mo_fields(ct):
            if (ct, f) in sample.IMPLIED:
                continue
            fm = model.fields[ct, f]
            expected = w @ fm.tilt(styles, tau)
            for v, p, e in zip(fm.values, fm.p, expected):
                if 0.02 <= p < 1 and abs(e / p - 1) > abs(worst):
                    worst, name = float(e / p - 1), f"{ct}.{f}.{v}"
    return worst, name


RULE = ("smallest tau > 0 on the grid at which validator 2 (truth MI above all permutation nulls) "
        "and validator 3 (recorded MI above all nulls, >= 50% of truth excess kept) pass on every seed")


HISTORY = [
    "Sweep 1 (discarded): MI validators pooled every crime of a series. A series shares one theta, so rows "
    "were not independent and the row-level permutation null was invalid: tau = 0 showed ~1.5 bits of "
    "'excess' MI. Fixed by using one crime per offender per crime type.",
    "Sweep 2 (discarded for power): at the 10k dev size the corrected test was underpowered — excess-kept "
    "ratios ranged from -7.1 to 3.9 and only the grid edge (1.5) passed. Tau is chosen for the 45k final "
    "corpus, so the sweep runs at that size. The rule itself is unchanged.",
]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="python -m linkage.generate.tau")
    ap.add_argument("--n-cases", type=int, default=None, help="corpus size for the sweep (default: corpus.yaml)")
    args = ap.parse_args()
    base = config_mod.load()
    if args.n_cases is not None:
        base["corpus"]["n_cases"] = args.n_cases
    rows = []
    for tau in GRID:
        per_seed = []
        for seed in SEEDS:
            cfg = copy.deepcopy(base)
            cfg["corpus"]["tau"] = tau
            truth, _ = sample.generate(cfg, seed)
            cases = truth.merge(corrupt.record(truth, cfg, seed, sample.build_model(cfg)), on="case_uid")
            vrng = streams.stream(seed, 99)
            v2 = validate.truth_mi(truth, vrng)
            v3 = validate.recorded_mi(cases, v2, vrng)
            per_seed.append({
                "seed": seed, "truth_mi_pass": v2["passed"], "recorded_mi_pass": v3["passed"],
                "truth_excess_bits": v2["excess_bits"], "recorded_excess_bits": v3["excess_bits"],
                "excess_kept": v3["excess_kept_vs_truth"],
            })
        ok = all(s["truth_mi_pass"] and s["recorded_mi_pass"] for s in per_seed)
        drift, drift_value = expected_drift(base, tau)
        rows.append({"tau": tau, "all_pass": ok, "expected_max_drift": round(drift, 3),
                     "expected_max_drift_value": drift_value, "seeds": per_seed})
        print(f"tau {tau:<5} {'PASS' if ok else 'fail'}  truth excess "
              f"{[s['truth_excess_bits'] for s in per_seed]}  kept {[s['excess_kept'] for s in per_seed]}  "
              f"expected drift {drift:+.1%} ({drift_value})")

    chosen = next((r["tau"] for r in rows if r["tau"] > 0 and r["all_pass"]), None)
    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "rule": RULE, "n_cases": base["corpus"]["n_cases"], "seeds": SEEDS,
           "chosen_tau": chosen, "history": HISTORY, "grid": rows}
    path = config_mod.CONFIG_DIR / "tau_selection.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nchosen tau: {chosen}  → {path}")
    return 0 if chosen is not None else 1


if __name__ == "__main__":
    sys.exit(main())
