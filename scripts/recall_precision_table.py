"""recall@50, recall@200, and precision@10 per pair_class, on the real
data/final/ corpus, using the default embedder (tfidf_original) and the
already-trained weights.json.

    python -m scripts.recall_precision_table
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from linkage import config as config_mod
from linkage.evaluate import precision_at_k, recall_at_k, scored_candidates
from linkage.retrieve import HashingTfidfEmbedder, embed_corpus, top_k
from linkage.train import DATA_DIR, Context

PAIR_CLASSES = ("same_type", "cross_type")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scripts.recall_precision_table",
                                 description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--config-dir", type=Path, default=config_mod.CONFIG_DIR)
    ap.add_argument("--weights", type=Path, default=Path(__file__).resolve().parent.parent / "weights.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    cfg = config_mod.load(args.config_dir)
    print("building context (data/final/, unmodified — repeat_rate as committed)...")
    ctx = Context(args.data_dir, cfg, seed=args.seed, build_retrieval=False)
    test_cases = {cid for cid in ctx.by_id if ctx.split.get(ctx.offender(cid)) == "test"}

    ids = [c["case_id"] for c in ctx.cases]
    texts = [c["narrative_text"] for c in ctx.cases]
    print("embedding + retrieving top-200 (default tfidf_original embedder)...")
    vectors, _ = embed_corpus(ids, texts, HashingTfidfEmbedder(dim=512))
    neighbours_200 = top_k(vectors, ids, k=200)

    recall_50 = recall_at_k(ctx, test_cases, neighbours_200, k=50)
    recall_200 = recall_at_k(ctx, test_cases, neighbours_200, k=200)

    # precision@10 needs scored candidates — reuse the top-50 slice (production shortlist size)
    ctx.neighbours = {cid: nbrs[:50] for cid, nbrs in neighbours_200.items()}
    weights = json.loads(args.weights.read_text())
    scored = scored_candidates(ctx, weights, ("test",))
    precision_10 = precision_at_k(scored)

    print(f"\n{len(test_cases)} test cases\n")
    print(f"{'metric':<16} {'same_type':>12} {'cross_type':>12}")
    for label, vals in (("recall@50", recall_50), ("recall@200", recall_200),
                        ("precision@10", precision_10)):
        row = " ".join(f"{(vals[pc]*100 if vals[pc] is not None else float('nan')):>11.1f}%"
                       for pc in PAIR_CLASSES)
        print(f"{label:<16} {row}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
