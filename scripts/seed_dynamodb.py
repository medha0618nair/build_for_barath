"""Seeds the deployed `cases` and `links` DynamoDB tables directly from the
already-computed local artifacts (data/final/, data/final/links.parquet),
for a fast, working demo — without waiting for a full Normalise->Enrich->
Index->Link Step Functions run over all 44,533 cases.

    python -m linkage.link_batch                         # if links.parquet is stale
    python -m scripts.seed_dynamodb --cases-table crime-linkage-dev-cases \
        --links-table crime-linkage-dev-links

By default seeds the same handful of example cases frontend/fixtures/ uses
(the highest-bits link in the corpus, a same-type/cross-type/low-evidence
example, a relocated cross-state pair) plus every case that appears as one
of their link partners — enough for the deploy gate's
`curl .../cases/{id}` to return something real. Pass --all to seed the
full corpus instead (slow: ~44k case writes, ~2.2M link writes).
"""
from __future__ import annotations

import argparse
import sys
from decimal import Decimal
from pathlib import Path

import boto3

from linkage import config as config_mod
from linkage.extract import LocalStubExtractor
from linkage.normalise import normalise_all
from scripts.build_fixtures import find_relocated_pair

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "final"


def _num(x):
    return None if x is None else Decimal(str(round(float(x), 6)))


def _case_item(case: dict) -> dict:
    return {
        "case_id": case["case_id"], "sk": "META", "state_code": case["state_code"],
        "district": case["district"], "crime_type": case["crime_type"],
        "occurred_from": case["occurred_from"].isoformat(),
        "occurred_to": case["occurred_to"].isoformat(),
        "registered_at": case["registered_at"].isoformat(),
        "mo_core": case["mo_core"], "mo_ext": case["mo_ext"],
        "narrative_text": case["narrative_text"], "narrative_lang": case["narrative_lang"],
        "field_provenance": case["field_provenance"], "pii_ref": None,
    }


def _bits_desc_sk(total_bits: float, partner_id: str) -> str:
    # Lexicographic SK order must equal descending total_bits (TECHNICAL_SPEC.md
    # §3.2: `links` PK case_id_a SK bits_desc#case_id_b). Offset into a fixed
    # positive range so string sort == numeric descending sort.
    encoded = 999999 - int(round((total_bits + 100) * 1000))
    return f"{max(0, min(999999, encoded)):06d}#{partner_id}"


def _links_items(case_id: str, links_df) -> list[dict]:
    rows = links_df[links_df["case_id"] == case_id]
    items = []
    for _, row in rows.iterrows():
        items.append({
            "case_id_a": case_id,
            "sk": _bits_desc_sk(float(row["total_bits"]), row["partner_id"]),
            "partner_id": row["partner_id"], "rank": int(row["rank"]),
            "total_bits": _num(row["total_bits"]), "prior_bits": _num(row["prior_bits"]),
            "pair_class": row["pair_class"],
            "contributions": [
                {"field": c["field"], "value_a": c["value_a"], "value_b": c["value_b"],
                 "u": _num(c["u"]), "bits": _num(c["bits"]),
                 "provenance": {"a": c["provenance_a"], "b": c["provenance_b"]}}
                for c in row["contributions"]
            ],
        })
    return items


def main(argv: list[str] | None = None) -> int:
    import pandas as pd

    ap = argparse.ArgumentParser(prog="python -m scripts.seed_dynamodb", description=__doc__.splitlines()[0])
    ap.add_argument("--cases-table", required=True)
    ap.add_argument("--links-table", required=True)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--config-dir", type=Path, default=config_mod.CONFIG_DIR)
    ap.add_argument("--all", action="store_true", help="seed the full corpus, not just the demo subset")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    links_path = args.data_dir / "links.parquet"
    if not links_path.exists():
        print(f"{links_path} not found — run `python -m linkage.link_batch` first")
        return 1
    links_df = pd.read_parquet(links_path)

    cfg = config_mod.load(args.config_dir)
    cases = normalise_all(cfg, args.data_dir, LocalStubExtractor())
    by_id = {c["case_id"]: c for c in cases}

    if args.all:
        seed_ids = set(by_id)
    else:
        truth = pd.read_parquet(args.data_dir / "truth.parquet")
        top = links_df.loc[links_df["total_bits"].idxmax(), "case_id"]
        same_type_example = links_df[links_df["pair_class"] == "same_type"]["case_id"].iloc[0]
        cross_rows = links_df[links_df["pair_class"] == "cross_type"]
        cross_type_example = cross_rows["case_id"].iloc[0] if len(cross_rows) else same_type_example
        low_evidence = links_df.loc[links_df["total_bits"].idxmin(), "case_id"]
        relocated = find_relocated_pair(truth) or ()
        seed_ids = {top, same_type_example, cross_type_example, low_evidence, *relocated}
        # pull in every partner of every seeded case too, so their links pages aren't empty
        seed_ids |= set(links_df[links_df["case_id"].isin(seed_ids)]["partner_id"])

    dynamodb = boto3.resource("dynamodb")
    cases_table = dynamodb.Table(args.cases_table)
    links_table = dynamodb.Table(args.links_table)

    n_cases = n_links = 0
    with cases_table.batch_writer() as batch:
        for case_id in seed_ids:
            case = by_id.get(case_id)
            if case is None:
                continue
            batch.put_item(Item=_case_item(case))
            n_cases += 1

    with links_table.batch_writer() as batch:
        for case_id in seed_ids:
            for item in _links_items(case_id, links_df):
                batch.put_item(Item=item)
                n_links += 1

    print(f"seeded {n_cases} cases into {args.cases_table}, {n_links} links into {args.links_table}")
    if not args.all:
        print(f"try: curl https://<api-url>/v1/cases/{sorted(seed_ids)[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
