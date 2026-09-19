"""Build frontend/fixtures/ from real data/final/links.parquet rows, so the
UI (Phase 6) can be built against the exact shape api/openapi.yaml defines,
with no API server required yet.

    python -m linkage.link_batch                    # writes data/final/links.parquet first
    python -m scripts.build_fixtures

Picks: the single highest-bits link in the whole corpus, one same-type and
one cross-type example, one relocated (cross-state) same-offender pair for
the demo (Phase 6's "Pune/Indore-style pair" toggle), and a low-evidence
example so the UI doesn't only ever see confident cases.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from linkage import config as config_mod
from linkage.extract import LocalStubExtractor
from linkage.normalise import normalise_all

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "final"
FIXTURES_DIR = Path(__file__).resolve().parent.parent / "frontend" / "fixtures"


def case_record(case: dict) -> dict:
    return {
        "case_id": case["case_id"],
        "state_code": case["state_code"],
        "district": case["district"],
        "crime_type": case["crime_type"],
        "occurred_from": case["occurred_from"].isoformat(),
        "occurred_to": case["occurred_to"].isoformat(),
        "registered_at": case["registered_at"].isoformat(),
        "mo_core": case["mo_core"],
        "mo_ext": case["mo_ext"],
        "narrative_text": case["narrative_text"],
        "narrative_lang": case["narrative_lang"],
        "field_provenance": case["field_provenance"],
    }


def driven_by(contributions: list[dict], n: int = 3) -> list[str]:
    top = sorted(contributions, key=lambda c: -abs(c["bits"]))[:n]
    return [f"{c['field']}={c['value_a']}" if c["value_a"] == c["value_b"] else
            f"{c['field']} disagrees" for c in top]


def links_response(links_df: pd.DataFrame, case_id: str, scope: str) -> dict:
    rows = links_df[links_df["case_id"] == case_id]
    if scope == "same":
        rows = rows[rows["pair_class"] == "same_type"]
    links = []
    for _, row in rows.sort_values("rank").iterrows():
        contributions = list(row["contributions"])
        links.append({
            "partner_id": row["partner_id"],
            "rank": int(row["rank"]),
            "total_bits": float(row["total_bits"]),
            "prior_bits": None if pd.isna(row["prior_bits"]) else float(row["prior_bits"]),
            "pair_class": row["pair_class"],
            "driven_by": driven_by(contributions),
            "contributions": [
                {"field": c["field"], "value_a": c["value_a"], "value_b": c["value_b"],
                 "u": float(c["u"]), "bits": float(c["bits"]),
                 "provenance": {"a": c["provenance_a"], "b": c["provenance_b"]}}
                for c in contributions
            ],
        })
    return {"case_id": case_id, "scope": scope, "links": links}


def find_relocated_pair(truth: pd.DataFrame) -> tuple[str, str] | None:
    ing = truth[truth["ingested"] & truth["is_serial"]]
    by_off = ing.groupby("offender_id")
    for offender_id, group in by_off:
        states = group["state_code"].unique()
        if len(states) > 1:
            a = group[group["state_code"] == states[0]].iloc[0]
            b = group[group["state_code"] == states[1]].iloc[0]
            return a["case_id"], b["case_id"]
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scripts.build_fixtures", description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--config-dir", type=Path, default=config_mod.CONFIG_DIR)
    ap.add_argument("--out", type=Path, default=FIXTURES_DIR)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    links_path = args.data_dir / "links.parquet"
    if not links_path.exists():
        print(f"{links_path} not found — run `python -m linkage.link_batch` first")
        return 1

    cfg = config_mod.load(args.config_dir)
    cases = normalise_all(cfg, args.data_dir, LocalStubExtractor())
    by_id = {c["case_id"]: c for c in cases}
    links_df = pd.read_parquet(links_path)
    truth = pd.read_parquet(args.data_dir / "truth.parquet")

    top = links_df.loc[links_df["total_bits"].idxmax(), "case_id"]
    same_type_example = links_df[links_df["pair_class"] == "same_type"]["case_id"].iloc[0]
    cross_rows = links_df[links_df["pair_class"] == "cross_type"]
    cross_type_example = cross_rows["case_id"].iloc[0] if len(cross_rows) else same_type_example
    low_evidence = links_df.loc[links_df["total_bits"].idxmin(), "case_id"]
    relocated = find_relocated_pair(truth)

    example_ids = {top, same_type_example, cross_type_example, low_evidence}
    if relocated:
        example_ids.update(relocated)

    (args.out / "cases").mkdir(parents=True, exist_ok=True)
    (args.out / "links").mkdir(parents=True, exist_ok=True)

    for case_id in example_ids:
        case = by_id.get(case_id)
        if case is None:
            continue
        (args.out / "cases" / f"{case_id}.json").write_text(
            json.dumps(case_record(case), indent=2), encoding="utf-8")
        for scope in ("same", "all"):
            (args.out / "links" / f"{case_id}_{scope}.json").write_text(
                json.dumps(links_response(links_df, case_id, scope), indent=2), encoding="utf-8")

    manifest = {
        "top_link_case": top,
        "same_type_example": same_type_example,
        "cross_type_example": cross_type_example,
        "low_evidence_example": low_evidence,
        "relocated_demo_pair": list(relocated) if relocated else None,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    feedback_example = {
        "case_id_a": top,
        "case_id_b": links_response(links_df, top, "all")["links"][0]["partner_id"],
        "status": "confirmed",
        "reason": "entry method and counter-forensic evidence match; analyst confirmed via FIR review.",
        "actor_id": "officer.demo@example.gov.in",
        "pair_id": f"{top}#confirmed-demo",
        "submitted_at": "2026-01-15T10:30:00+05:30",
    }
    (args.out / "feedback_example.json").write_text(json.dumps(feedback_example, indent=2), encoding="utf-8")

    print(f"wrote fixtures for {len(example_ids)} cases to {args.out}")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
