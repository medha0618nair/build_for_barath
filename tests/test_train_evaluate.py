from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from linkage import config as config_mod
from linkage.evaluate import evaluate, pr_auc, precision_at_k, recall_at_k
from linkage.sweep import _generate
from linkage.train import Context, offender_split, pair_features, train


@pytest.fixture(scope="module")
def small_ctx():
    """A tiny (600-case) corpus, generated fresh, so tests don't depend on
    data/final/ and stay fast. Exercises the exact same code path (render.py's
    NaN fix, normalise, retrieve, train) the sweep uses."""
    cfg = config_mod.load()
    cfg = {**cfg, "corpus": {**cfg["corpus"], "n_cases": 600}}
    with tempfile.TemporaryDirectory(prefix="linkage_test_") as tmp:
        out = Path(tmp)
        _generate(cfg, repeat_rate=0.6, seed=3, out=out)
        ctx = Context(out, {**cfg, "corpus": {**cfg["corpus"], "repeat_rate":
                      {**cfg["corpus"]["repeat_rate"], "value": 0.6}}}, seed=0, embed_dim=64)
        yield ctx


def test_offender_split_is_a_partition():
    import pandas as pd
    truth = pd.DataFrame({"offender_id": [f"o{i}" for i in range(100)]})
    split = offender_split(truth, train=0.7, calib=0.15, seed=0)
    assert set(split.values()) <= {"train", "calib", "test"}
    assert len(split) == 100


def test_offender_split_reproducible_with_same_seed():
    import pandas as pd
    truth = pd.DataFrame({"offender_id": [f"o{i}" for i in range(50)]})
    a = offender_split(truth, seed=1)
    b = offender_split(truth, seed=1)
    assert a == b


def test_context_builds_and_every_case_has_a_split(small_ctx):
    assert len(small_ctx.cases) > 0
    for case in small_ctx.cases:
        off = small_ctx.offender(case["case_id"])
        assert small_ctx.split[off] in ("train", "calib", "test")


def test_pair_features_shape_and_pair_class(small_ctx):
    from linkage.train import FEATURE_NAMES

    a, b = small_ctx.cases[0], small_ctx.cases[1]
    x, pair_class = pair_features(a, b, small_ctx)
    assert x.shape == (len(FEATURE_NAMES),)
    assert pair_class == ("same_type" if a["crime_type"] == b["crime_type"] else "cross_type")


def test_train_produces_weights_with_both_pair_classes(small_ctx):
    weights = train(small_ctx)
    assert "same_type" in weights and "cross_type" in weights
    for pc in ("same_type", "cross_type"):
        assert "feature_names" in weights[pc]


def test_evaluate_reports_both_pair_classes_never_blended(small_ctx):
    weights = train(small_ctx)
    results = evaluate(small_ctx, weights)
    assert set(results["recall_at_50"]) == {"same_type", "cross_type"}
    assert set(results["precision_at_10"]) == {"same_type", "cross_type"}
    assert set(results["pr_auc"]) == {"same_type", "cross_type"}
    # no metric silently combines the two pools into one number
    for metric in ("recall_at_50", "precision_at_10", "pr_auc"):
        assert len(results[metric]) == 2


def test_recall_at_k_is_between_0_and_1(small_ctx):
    test_cases = {cid for cid in small_ctx.by_id if small_ctx.split.get(small_ctx.offender(cid)) == "test"}
    r = recall_at_k(small_ctx, test_cases)
    for v in r.values():
        assert v is None or 0.0 <= v <= 1.0
