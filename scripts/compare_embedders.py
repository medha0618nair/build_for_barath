"""Compare retrieval embedders on recall@50, per pair_class, on one Context
(normalised corpus + offender split held fixed, only the embedder changes).

    python -m scripts.compare_embedders --data-dir data/final

Prints a before/after table and returns the winning embedder's name so
sweep.py / train.py can be pointed at it.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from linkage import config as config_mod
from linkage.evaluate import recall_at_k
from linkage.retrieve import HashingTfidfEmbedder, SentenceTransformerEmbedder
from linkage.train import DATA_DIR, Context

EMBEDDERS = {
    "tfidf_original": lambda: HashingTfidfEmbedder(dim=512),
    "tfidf_tuned": lambda: HashingTfidfEmbedder(dim=512, ngram_range=(1, 2),
                                                sublinear_tf=True, max_df=0.5),
    "minilm": lambda: SentenceTransformerEmbedder("all-MiniLM-L6-v2"),
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scripts.compare_embedders", description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--config-dir", type=Path, default=config_mod.CONFIG_DIR)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="*", choices=list(EMBEDDERS), default=list(EMBEDDERS))
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    cfg = config_mod.load(args.config_dir)
    print("building context (normalise, frequencies, priors, offender split)...")
    ctx = Context(args.data_dir, cfg, seed=args.seed, build_retrieval=False)
    test_cases = {cid for cid in ctx.by_id if ctx.split.get(ctx.offender(cid)) == "test"}
    print(f"{len(test_cases)} test cases\n")

    rows = []
    for name in args.only:
        t0 = time.time()
        ctx.set_embedder(EMBEDDERS[name]())
        elapsed = time.time() - t0
        recall = recall_at_k(ctx, test_cases)
        rows.append((name, recall["same_type"], recall["cross_type"], elapsed))
        print(f"{name:<16} same_type={recall['same_type']}  cross_type={recall['cross_type']}  "
              f"({elapsed:.1f}s)")

    print(f"\n{'embedder':<16} {'same_type':>12} {'cross_type':>12}")
    for name, same, cross, _ in rows:
        same_s = f"{same*100:.1f}%" if same is not None else "n/a"
        cross_s = f"{cross*100:.1f}%" if cross is not None else "n/a"
        print(f"{name:<16} {same_s:>12} {cross_s:>12}")

    best = max(rows, key=lambda r: (r[1] or 0.0))
    print(f"\nbest same_type recall@50: {best[0]} ({(best[1] or 0)*100:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
