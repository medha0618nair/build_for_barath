"""Batch linker: for every case, retrieve + score, write links.parquet
(TECHNICAL_SPEC.md §3.2 `links` table, §6 LinkBatch).

    python -m linkage.link_batch --data-dir data/final --out data/final/links.parquet

Every case links against the *whole* corpus (no offender split — that's an
evaluation-only concept from linkage/train.py's Context). Retrieval stays
narrative-only, chunked, 1,000 rows at a time (CLAUDE.md hard rule 1 and
"Conventions"); each case keeps its own top-50 retrieved candidates,
re-ranked by score — one row per (case_id, partner_id) directed edge, which
is what the DynamoDB `links` table's PK case_id_a / SK bits_desc#case_id_b
layout expects: "top N links for this case" is a per-case range query, not
a deduplicated undirected pair.

Ranking is by the raw Fellegi-Sunter total (prior + contributions), not the
logistic-regression residual correction from linkage/train.py's
weights.json — the contribution breakdown the UI renders (CLAUDE.md: "rank,
bits, driving features", never a probability) only has a clean per-field
story for the FS total; the LR layer is a calibration/evaluation tool, not
a display quantity.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from linkage import config as config_mod
from linkage import frequencies as frequencies_mod
from linkage import prior as prior_mod
from linkage.extract import LocalStubExtractor
from linkage.features import Contribution
from linkage.normalise import normalise_all
from linkage.retrieve import HashingTfidfEmbedder, embed_corpus, top_k
from linkage.score import fit_cross_type_repeat_rate, score_pair

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "final"
TOP_K = 50
CHUNK_SIZE = 1000
EMBED_DIM = 512


def _contribution_row(c: Contribution) -> dict:
    return {
        "field": c.field,
        "value_a": None if c.value_a is None else str(c.value_a),
        "value_b": None if c.value_b is None else str(c.value_b),
        "u": round(c.u, 6),
        "bits": round(c.bits, 6),
        "provenance_a": c.provenance_a,
        "provenance_b": c.provenance_b,
    }


def link_batch(data_dir: Path, cfg: dict, embed_dim: int = EMBED_DIM) -> pd.DataFrame:
    cases = normalise_all(cfg, data_dir, LocalStubExtractor())
    by_id = {c["case_id"]: c for c in cases}
    truth = pd.read_parquet(data_dir / "truth.parquet")
    ing = truth[truth["ingested"]]

    freqs = frequencies_mod.compute(cases)
    priors = prior_mod.compute(ing)
    rr = cfg["corpus"]["repeat_rate"]
    alpha_same = config_mod.alpha(rr["value"], rr["reference_frequency"])
    cross_rr = fit_cross_type_repeat_rate(cases, truth)
    alpha_cross = (config_mod.alpha(cross_rr, rr["reference_frequency"])
                   if cross_rr and cross_rr > rr["reference_frequency"] else alpha_same)

    ids = [c["case_id"] for c in cases]
    texts = [c["narrative_text"] for c in cases]
    vectors, _ = embed_corpus(ids, texts, HashingTfidfEmbedder(dim=embed_dim))
    neighbours = top_k(vectors, ids, k=TOP_K, chunk_size=CHUNK_SIZE)

    rows = []
    for case_id, candidates in neighbours.items():
        case_a = by_id[case_id]
        scored = [(partner_id, score_pair(case_a, by_id[partner_id], freqs, priors,
                                          alpha_same, alpha_cross))
                  for partner_id, _sim in candidates]
        scored.sort(key=lambda t: -t[1]["total_bits"])
        for rank, (partner_id, result) in enumerate(scored, start=1):
            rows.append({
                "case_id": case_id,
                "partner_id": partner_id,
                "rank": rank,
                "total_bits": round(result["total_bits"], 6),
                "prior_bits": None if result["prior_bits"] is None else round(result["prior_bits"], 6),
                "pair_class": result["pair_class"],
                "contributions": [_contribution_row(c) for c in result["contributions"]],
            })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m linkage.link_batch", description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--config-dir", type=Path, default=config_mod.CONFIG_DIR)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    out = args.out or (args.data_dir / "links.parquet")
    cfg = config_mod.load(args.config_dir)
    print(f"linking {args.data_dir}...")
    df = link_batch(args.data_dir, cfg)
    df.to_parquet(out, index=False)
    print(f"wrote {out}: {len(df)} rows, {df['case_id'].nunique()} cases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
