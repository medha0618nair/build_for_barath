"""recall@50, precision@10, PR-AUC — reported per pair_class, never blended
(CLAUDE.md hard rule 7), plus per-offender metrics (DATASET.md: the top 10%
of offenders hold half the true pairs, so per-pair metrics alone hide how
concentrated the problem is).

    python -m linkage.evaluate --data-dir data/final

Everything here runs on the TEST offender split only (linkage/train.py's
Context.split) — no offender whose cases trained the model or fit its
calibration contributes to a metric.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import auc, precision_recall_curve

from linkage import config as config_mod
from linkage import schema
from linkage.train import Context, DATA_DIR, build_training_table, candidate_pairs, is_true_pair

PAIR_CLASSES = ("same_type", "cross_type")


def _lr_score(x: np.ndarray, weights: dict) -> float:
    w, b = np.array(weights["coef"]), weights["bias"]
    return float(w @ x + b)


def raw_fs_score(x: np.ndarray) -> float:
    """The FS total (prior + sum of per-field bits) before any LR correction —
    exactly summing FEATURE_NAMES's slots, since that's how they were built."""
    return float(x.sum())


# --- ground truth: all same-offender pairs among a case set, no retrieval involved ---

def true_pairs(ctx: Context, cases: set[str]) -> dict[str, set[tuple[str, str]]]:
    """Every unordered same-offender pair with both endpoints in `cases`,
    split by pair_class. This is the actual denominator recall@k measures
    against — a direct count from offender_id, untouched by retrieval."""
    by_offender: dict[str, list[str]] = defaultdict(list)
    for cid in cases:
        by_offender[ctx.offender(cid)].append(cid)
    out: dict[str, set[tuple[str, str]]] = {"same_type": set(), "cross_type": set()}
    for case_ids in by_offender.values():
        if len(case_ids) < 2:
            continue
        for i in range(len(case_ids)):
            for j in range(i + 1, len(case_ids)):
                a, b = sorted((case_ids[i], case_ids[j]))
                pc = ("same_type" if ctx.by_id[a]["crime_type"] == ctx.by_id[b]["crime_type"]
                      else "cross_type")
                out[pc].add((a, b))
    return out


# --- recall@k (retrieval stage) -----------------------------------------------

def recall_at_k(ctx: Context, test_cases: set[str], neighbours: dict | None = None,
                 k: int = 50) -> dict[str, float | None]:
    """For each query case with >=1 true partner (of a given pair_class) also
    in the test set, recall_i = |true partners found in top-k| / |true partners|.
    recall@k is the mean over such queries, per pair_class.

    `neighbours` defaults to ctx.neighbours (the corpus-wide top-50 index);
    pass a different {case_id: [(case_id, sim), ...]} map (e.g. a top-500
    index) to evaluate recall@k for k > 50, or for a different embedder
    entirely. The denominator here (`relevant`, built from `test_cases`
    directly via true_pairs-style offender grouping) is NOT filtered by
    retrieval — see true_pairs() for the same computation exposed directly."""
    neighbours = ctx.neighbours if neighbours is None else neighbours
    by_offender: dict[str, list[str]] = defaultdict(list)
    for cid in test_cases:
        by_offender[ctx.offender(cid)].append(cid)

    hits = {pc: [] for pc in PAIR_CLASSES}
    for cid in test_cases:
        case = ctx.by_id[cid]
        partners = [p for p in by_offender[ctx.offender(cid)] if p != cid]
        if not partners:
            continue
        retrieved = {b for b, _ in neighbours.get(cid, [])[:k]}
        for pc in PAIR_CLASSES:
            same = pc == "same_type"
            relevant = [p for p in partners
                        if (ctx.by_id[p]["crime_type"] == case["crime_type"]) == same]
            if not relevant:
                continue
            found = sum(1 for p in relevant if p in retrieved)
            hits[pc].append(found / len(relevant))
    return {pc: (float(np.mean(v)) if v else None) for pc, v in hits.items()}


# --- precision@10 and PR-AUC (scoring stage) ---------------------------------

def scored_candidates(ctx: Context, weights: dict, splits: tuple[str, ...]):
    """(pair_class -> list of (case_a, case_b, x, y, raw_score, lr_score))."""
    pairs = candidate_pairs(ctx, splits)
    X, y = build_training_table(ctx, pairs)
    out: dict[str, list] = {pc: [] for pc in PAIR_CLASSES}
    idx = {pc: 0 for pc in PAIR_CLASSES}
    from linkage.train import pair_features
    for a_id, b_id in pairs:
        x, pc = pair_features(ctx.by_id[a_id], ctx.by_id[b_id], ctx)
        label = int(is_true_pair(a_id, b_id, ctx))
        raw = raw_fs_score(x)
        lr = _lr_score(x, weights[pc]) if "coef" in weights[pc] else raw
        out[pc].append((a_id, b_id, label, raw, lr))
    return out


def precision_at_k(scored: dict[str, list], k: int = 10) -> dict[str, float | None]:
    result = {}
    for pc, rows in scored.items():
        by_query: dict[str, list] = defaultdict(list)
        for a_id, b_id, label, raw, lr in rows:
            by_query[a_id].append((lr, label))
            by_query[b_id].append((lr, label))
        precisions = []
        for cid, cands in by_query.items():
            if not any(label for _, label in cands):
                continue
            top = sorted(cands, key=lambda t: -t[0])[:k]
            precisions.append(sum(label for _, label in top) / len(top))
        result[pc] = float(np.mean(precisions)) if precisions else None
    return result


def pr_auc(scored: dict[str, list]) -> dict[str, float | None]:
    result = {}
    for pc, rows in scored.items():
        if not rows:
            result[pc] = None
            continue
        y = np.array([r[2] for r in rows])
        s = np.array([r[4] for r in rows])
        if len(np.unique(y)) < 2:
            result[pc] = None
            continue
        precision, recall, _ = precision_recall_curve(y, s)
        result[pc] = float(auc(recall, precision))
    return result


# --- per-offender metrics ------------------------------------------------------

def per_offender_recall(ctx: Context, test_cases: set[str]) -> dict[str, float | None]:
    """Same as recall@50, but averaged per OFFENDER (one number per serial
    offender) instead of per case — offenders with more cases don't get more
    weight in the average. Only offenders with >=2 test cases count."""
    by_offender: dict[str, list[str]] = defaultdict(list)
    for cid in test_cases:
        by_offender[ctx.offender(cid)].append(cid)

    hits = {pc: [] for pc in PAIR_CLASSES}
    for off, case_ids in by_offender.items():
        if len(case_ids) < 2:
            continue
        for pc in PAIR_CLASSES:
            same = pc == "same_type"
            found, total = 0, 0
            for cid in case_ids:
                case = ctx.by_id[cid]
                retrieved = {b for b, _ in ctx.neighbours.get(cid, [])}
                relevant = [p for p in case_ids
                            if p != cid and (ctx.by_id[p]["crime_type"] == case["crime_type"]) == same]
                for p in relevant:
                    total += 1
                    found += p in retrieved
            if total:
                hits[pc].append(found / total)
    return {pc: (float(np.mean(v)) if v else None) for pc, v in hits.items()}


def evaluate(ctx: Context, weights: dict) -> dict:
    test_cases = {cid for cid in ctx.by_id if ctx.split.get(ctx.offender(cid)) == "test"}
    scored = scored_candidates(ctx, weights, ("test",))
    gt_pairs = true_pairs(ctx, test_cases)
    return {
        "n_test_cases": len(test_cases),
        "recall_at_50": recall_at_k(ctx, test_cases),
        "precision_at_10": precision_at_k(scored),
        "pr_auc": pr_auc(scored),
        "per_offender_recall_at_50": per_offender_recall(ctx, test_cases),
        # candidate pairs that survived retrieval — precision@10/PR-AUC's pool, NOT recall@50's denominator
        "n_candidate_pairs": {pc: len(rows) for pc, rows in scored.items()},
        "n_candidate_true_pairs": {pc: sum(r[2] for r in rows) for pc, rows in scored.items()},
        # the actual ground-truth denominator recall@50 is measured against
        "n_true_pairs": {pc: len(gt_pairs[pc]) for pc in PAIR_CLASSES},
    }


def print_table(results: dict) -> None:
    print(f"{'metric':<26} {'same_type':>12} {'cross_type':>12}")
    for key, label in (("recall_at_50", "recall@50"), ("precision_at_10", "precision@10"),
                       ("pr_auc", "PR-AUC"), ("per_offender_recall_at_50", "per-offender recall@50")):
        vals = results[key]
        row = " ".join(f"{(vals[pc]*100 if vals[pc] is not None else float('nan')):>11.1f}%"
                       for pc in PAIR_CLASSES)
        print(f"{label:<26} {row}")
    print(f"{'n true pairs (ground truth)':<26} {results['n_true_pairs']['same_type']:>12} "
          f"{results['n_true_pairs']['cross_type']:>12}")
    print(f"{'n candidate pairs':<26} {results['n_candidate_pairs']['same_type']:>12} "
          f"{results['n_candidate_pairs']['cross_type']:>12}")
    print(f"{'  of which true':<26} {results['n_candidate_true_pairs']['same_type']:>12} "
          f"{results['n_candidate_true_pairs']['cross_type']:>12}")
    print(f"\n{results['n_test_cases']} test cases (held out by offender)")


def main(argv: list[str] | None = None) -> int:
    import json
    ap = argparse.ArgumentParser(prog="python -m linkage.evaluate", description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--config-dir", type=Path, default=config_mod.CONFIG_DIR)
    ap.add_argument("--weights", type=Path, default=Path(__file__).resolve().parent.parent / "weights.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    cfg = config_mod.load(args.config_dir)
    ctx = Context(args.data_dir, cfg, seed=args.seed)
    weights = json.loads(args.weights.read_text())
    results = evaluate(ctx, weights)
    print_table(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
