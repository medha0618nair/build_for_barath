"""recall@50 / @200 / @500 for tuned TF-IDF, MiniLM, and the (rule-violating,
diagnostic-only) oracle MO-field embedder — one table, same held-out split.

    python -m scripts.multi_k_recall --data-dir data/final
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from linkage import config as config_mod
from linkage.evaluate import recall_at_k
from linkage.retrieve import HashingTfidfEmbedder, SentenceTransformerEmbedder, embed_corpus, top_k
from linkage.train import DATA_DIR, Context
from scripts.oracle_embedder import oracle_vectors

K_VALUES = (50, 200, 500)
MAX_K = max(K_VALUES)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scripts.multi_k_recall", description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--config-dir", type=Path, default=config_mod.CONFIG_DIR)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    cfg = config_mod.load(args.config_dir)
    print("building context...")
    ctx = Context(args.data_dir, cfg, seed=args.seed, build_retrieval=False)
    test_cases = {cid for cid in ctx.by_id if ctx.split.get(ctx.offender(cid)) == "test"}
    ids = [c["case_id"] for c in ctx.cases]
    texts = [c["narrative_text"] for c in ctx.cases]

    rows: dict[str, dict[int, dict[str, float | None]]] = {}

    print("oracle (structured MO fields — diagnostic only, rule-violating)...")
    vecs = oracle_vectors(ctx.cases)
    neighbours = top_k(vecs, ids, k=MAX_K)
    rows["oracle (rule-violating)"] = {k: recall_at_k(ctx, test_cases, neighbours, k) for k in K_VALUES}

    print("tfidf_tuned...")
    vecs, _ = embed_corpus(ids, texts, HashingTfidfEmbedder(dim=512, ngram_range=(1, 2),
                                                            sublinear_tf=True, max_df=0.5))
    neighbours = top_k(vecs, ids, k=MAX_K)
    rows["tfidf_tuned"] = {k: recall_at_k(ctx, test_cases, neighbours, k) for k in K_VALUES}

    print("minilm...")
    vecs, _ = embed_corpus(ids, texts, SentenceTransformerEmbedder())
    neighbours = top_k(vecs, ids, k=MAX_K)
    rows["minilm"] = {k: recall_at_k(ctx, test_cases, neighbours, k) for k in K_VALUES}

    print(f"\n{len(test_cases)} test cases\n")
    header = f"{'embedder':<24}" + "".join(f"{'@'+str(k)+' same':>12}{'@'+str(k)+' cross':>13}" for k in K_VALUES)
    print(header)
    for name, by_k in rows.items():
        cells = []
        for k in K_VALUES:
            r = by_k[k]
            same = f"{r['same_type']*100:.1f}%" if r["same_type"] is not None else "n/a"
            cross = f"{r['cross_type']*100:.1f}%" if r["cross_type"] is not None else "n/a"
            cells.append(f"{same:>12}{cross:>13}")
        print(f"{name:<24}" + "".join(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
