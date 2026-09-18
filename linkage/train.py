"""Logistic regression as a residual correction on top of FS weights
(TECHNICAL_SPEC.md §4.7), fit locally, exported to weights.json.

    python -m linkage.train --data-dir data/final --out weights.json

Split by offender (CLAUDE.md hard rule 6): every case's offender_id is
assigned to exactly one of train/calibration/test up front, and a candidate
pair is only used by a split if *both* endpoints belong to it — a pair that
crosses the boundary is dropped rather than leaked.

One model per pair_class (CLAUDE.md hard rule 7: same-type and cross-type
are never blended) — same feature layout for both (one bits-per-field slot,
tag fields collapsed to one slot each, plus prior_bits), since a cross-type
pair only ever activates the mo_core slots plus, for same-family pairs like
residential/commercial burglary, that family's mo_ext slots too.

Negatives are downweighted (positive-unlabelled learning, CLAUDE.md /
DATASET.md limitations): an "unlabelled" retrieved candidate is not a
confirmed non-link, just an unconfirmed one, so it should not out-vote a
confirmed positive as if the label were trustworthy.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from linkage import config as config_mod
from linkage import frequencies as frequencies_mod
from linkage import prior as prior_mod
from linkage import schema
from linkage.extract import LocalStubExtractor
from linkage.normalise import normalise_all
from linkage.retrieve import HashingTfidfEmbedder, embed_corpus, top_k
from linkage.score import fit_cross_type_repeat_rate, score_pair

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "final"
WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "weights.json"

NEGATIVE_WEIGHT = 0.2   # positive-unlabelled downweight; see module docstring
TOP_K = 50
CHUNK_SIZE = 1000

# One feature slot per MO field (tag sub-contributions collapsed into their
# field's slot) plus the prior. Same layout for both pair_class models —
# whichever slots don't apply to a given pair are just zero.
FEATURE_FIELDS = schema.ALL_MO_FIELDS
FEATURE_NAMES = (*FEATURE_FIELDS, "prior_bits")


class Context:
    """Everything training and evaluation both need: normalised cases,
    frequencies, priors, alphas, a retrieval index, and the offender split.
    Building this once and sharing it keeps train/evaluate/sweep consistent."""

    def __init__(self, data_dir: Path, cfg: dict, seed: int = 0, embed_dim: int = 512):
        self.cfg = cfg
        self.cases = normalise_all(cfg, data_dir, LocalStubExtractor())
        self.by_id = {c["case_id"]: c for c in self.cases}
        truth = pd.read_parquet(data_dir / "truth.parquet")
        self.truth = truth[truth["ingested"]].set_index("case_id")

        self.freqs = frequencies_mod.compute(self.cases)
        self.priors = prior_mod.compute(truth[truth["ingested"]])
        rr = cfg["corpus"]["repeat_rate"]
        self.alpha_same = config_mod.alpha(rr["value"], rr["reference_frequency"])
        cross_rr = fit_cross_type_repeat_rate(self.cases, truth)
        self.alpha_cross = (config_mod.alpha(cross_rr, rr["reference_frequency"])
                             if cross_rr and cross_rr > rr["reference_frequency"] else self.alpha_same)

        ids = [c["case_id"] for c in self.cases]
        texts = [c["narrative_text"] for c in self.cases]
        vectors, _ = embed_corpus(ids, texts, HashingTfidfEmbedder(dim=embed_dim))
        self.neighbours = top_k(vectors, ids, k=TOP_K, chunk_size=CHUNK_SIZE)

        self.split = offender_split(self.truth, seed=seed)

    def offender(self, case_id: str) -> str:
        return self.truth.loc[case_id, "offender_id"]


def offender_split(truth: pd.DataFrame, train: float = 0.7, calib: float = 0.15,
                    seed: int = 0) -> dict[str, str]:
    """offender_id -> 'train' | 'calib' | 'test', a random partition seeded
    for reproducibility. Every case of a given offender lands in one split."""
    offenders = sorted(truth["offender_id"].unique())
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(offenders))
    n_train = int(len(offenders) * train)
    n_calib = int(len(offenders) * calib)
    split = {}
    for rank, idx in enumerate(perm):
        bucket = "train" if rank < n_train else ("calib" if rank < n_train + n_calib else "test")
        split[offenders[idx]] = bucket
    return split


def pair_features(case_a: dict, case_b: dict, ctx: Context) -> tuple[np.ndarray, str]:
    """Feature vector in FEATURE_NAMES order, and the pair_class."""
    result = score_pair(case_a, case_b, ctx.freqs, ctx.priors, ctx.alpha_same, ctx.alpha_cross)
    by_field: dict[str, float] = {f: 0.0 for f in FEATURE_FIELDS}
    for c in result["contributions"]:
        field = c.field.split(":", 1)[0]  # collapse tools:crowbar etc. into "tools"
        by_field[field] += c.bits
    x = np.array([by_field[f] for f in FEATURE_FIELDS] + [result["prior_bits"] or 0.0], dtype=np.float64)
    return x, result["pair_class"]


def candidate_pairs(ctx: Context, splits: tuple[str, ...]) -> list[tuple[str, str]]:
    """Deduplicated (case_a, case_b) pairs from the retrieval index, both
    endpoints' offenders in one of `splits`."""
    allowed_cases = {cid for cid in ctx.by_id if ctx.split.get(ctx.offender(cid)) in splits}
    seen: set[tuple[str, str]] = set()
    pairs = []
    for a_id, neighbours in ctx.neighbours.items():
        if a_id not in allowed_cases:
            continue
        for b_id, _sim in neighbours:
            if b_id not in allowed_cases:
                continue
            key = (a_id, b_id) if a_id < b_id else (b_id, a_id)
            if key in seen:
                continue
            seen.add(key)
            pairs.append(key)
    return pairs


def is_true_pair(a_id: str, b_id: str, ctx: Context) -> bool:
    return ctx.offender(a_id) == ctx.offender(b_id)


def build_training_table(ctx: Context, pairs: list[tuple[str, str]]):
    X_by_class: dict[str, list[np.ndarray]] = {"same_type": [], "cross_type": []}
    y_by_class: dict[str, list[int]] = {"same_type": [], "cross_type": []}
    for a_id, b_id in pairs:
        x, pair_class = pair_features(ctx.by_id[a_id], ctx.by_id[b_id], ctx)
        X_by_class[pair_class].append(x)
        y_by_class[pair_class].append(int(is_true_pair(a_id, b_id, ctx)))
    return ({k: np.array(v) for k, v in X_by_class.items()},
            {k: np.array(v) for k, v in y_by_class.items()})


def fit_pair_class(X: np.ndarray, y: np.ndarray) -> dict:
    if len(np.unique(y)) < 2:
        return {"coef": [0.0] * X.shape[1], "bias": 0.0, "calibration": None}
    sample_weight = np.where(y == 1, 1.0, NEGATIVE_WEIGHT)
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y, sample_weight=sample_weight)
    return {"model": model}


def calibrate(model: LogisticRegression, X_calib: np.ndarray, y_calib: np.ndarray) -> IsotonicRegression | None:
    if len(X_calib) == 0 or len(np.unique(y_calib)) < 2:
        return None
    scores = model.decision_function(X_calib)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(scores, y_calib)
    return iso


def to_json(pair_class: str, model, iso: IsotonicRegression | None) -> dict:
    if isinstance(model, dict):  # degenerate: no positives to fit against
        return {"feature_names": list(FEATURE_NAMES), **model}
    out = {
        "feature_names": list(FEATURE_NAMES),
        "coef": model.coef_[0].tolist(),
        "bias": float(model.intercept_[0]),
    }
    if iso is not None:
        out["calibration"] = {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}
    else:
        out["calibration"] = None
    return out


def train(ctx: Context) -> dict:
    train_pairs = candidate_pairs(ctx, ("train",))
    calib_pairs = candidate_pairs(ctx, ("calib",))
    X_train, y_train = build_training_table(ctx, train_pairs)
    X_calib, y_calib = build_training_table(ctx, calib_pairs)

    weights = {"feature_names": list(FEATURE_NAMES), "alpha_same": ctx.alpha_same,
               "alpha_cross": ctx.alpha_cross, "negative_weight": NEGATIVE_WEIGHT}
    for pair_class in ("same_type", "cross_type"):
        fit = fit_pair_class(X_train[pair_class], y_train[pair_class])
        if "model" in fit:
            iso = calibrate(fit["model"], X_calib.get(pair_class, np.empty((0, len(FEATURE_NAMES)))),
                             y_calib.get(pair_class, np.empty((0,))))
            weights[pair_class] = to_json(pair_class, fit["model"], iso)
        else:
            weights[pair_class] = to_json(pair_class, fit, None)
        n_pos = int(y_train[pair_class].sum()) if len(y_train[pair_class]) else 0
        print(f"{pair_class}: {len(y_train[pair_class])} training pairs, {n_pos} positive")
    return weights


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m linkage.train", description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--config-dir", type=Path, default=config_mod.CONFIG_DIR)
    ap.add_argument("--out", type=Path, default=WEIGHTS_PATH)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    cfg = config_mod.load(args.config_dir)
    ctx = Context(args.data_dir, cfg, seed=args.seed)
    weights = train(ctx)
    args.out.write_text(json.dumps(weights, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
