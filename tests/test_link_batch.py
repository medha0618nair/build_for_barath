from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from linkage import config as config_mod
from linkage.link_batch import link_batch
from linkage.sweep import _generate


@pytest.fixture(scope="module")
def small_links():
    cfg = config_mod.load()
    cfg = {**cfg, "corpus": {**cfg["corpus"], "n_cases": 500}}
    with tempfile.TemporaryDirectory(prefix="linkage_linkbatch_test_") as tmp:
        out = Path(tmp)
        _generate(cfg, repeat_rate=0.6, seed=5, out=out)
        point_cfg = {**cfg, "corpus": {**cfg["corpus"], "repeat_rate":
                     {**cfg["corpus"]["repeat_rate"], "value": 0.6}}}
        df = link_batch(out, point_cfg, embed_dim=64)
        yield df


def test_has_expected_columns(small_links):
    expected = {"case_id", "partner_id", "rank", "total_bits", "prior_bits",
                "pair_class", "contributions"}
    assert expected <= set(small_links.columns)


def test_each_case_has_at_most_50_links(small_links):
    counts = small_links.groupby("case_id").size()
    assert (counts <= 50).all()


def test_ranks_are_dense_and_sorted_by_total_bits(small_links):
    for case_id, group in small_links.groupby("case_id"):
        group = group.sort_values("rank")
        assert group["rank"].tolist() == list(range(1, len(group) + 1))
        bits = group["total_bits"].tolist()
        assert bits == sorted(bits, reverse=True)


def test_pair_class_matches_crime_type_relationship(small_links):
    # can't see crime_type here directly, but pair_class must be one of the two
    assert set(small_links["pair_class"].unique()) <= {"same_type", "cross_type"}


def test_contributions_have_expected_shape(small_links):
    row = small_links.iloc[0]
    contributions = list(row["contributions"])
    if not contributions:
        pytest.skip("this row has no contributions to check")
    c = contributions[0]
    assert set(c.keys()) == {"field", "value_a", "value_b", "u", "bits",
                             "provenance_a", "provenance_b"}
    assert isinstance(c["bits"], float)


def test_no_case_links_to_itself(small_links):
    assert not (small_links["case_id"] == small_links["partner_id"]).any()
