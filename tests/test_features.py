from __future__ import annotations

import math

import pytest

from linkage.features import (
    agreement_bits,
    disagreement_bits,
    field_aggregate,
    m_of,
    score_categorical,
    score_tag,
)
from linkage.generate.corrupt import ABSENT, MISSING, UNKNOWABLE


def test_m_of_limits():
    # alpha -> inf: offender behaves like the population (m_v -> p_v)
    assert m_of(0.3, alpha=1e9) == pytest.approx(0.3, abs=1e-6)
    # alpha -> 0: offender is deterministic (m_v -> 1)
    assert m_of(0.3, alpha=1e-9) == pytest.approx(1.0, abs=1e-6)


def test_m_categorical_and_m_tag_are_the_same_formula():
    from linkage.features import m_categorical, m_tag
    assert m_categorical(0.16, 7.05) == pytest.approx(m_tag(0.16, 7.05))


def test_agreement_bits_positive_for_rare_value():
    # rare shared value: strong positive evidence
    assert agreement_bits(m_of(0.03, 7.05), 0.03) > 2.0


def test_disagreement_bits_negative():
    m = m_of(0.58, 7.05)
    assert disagreement_bits(m, 0.58) < 0


def test_field_aggregate_matches_worked_example_occupancy_row():
    # TECHNICAL_SPEC.md §4.3: occupancy disagrees, u=0.58, m=0.632, -0.191 bits.
    # temporarily_away (p=0.58) dominates the occupancy distribution closely
    # enough that the field aggregate lands within 0.05 bits of that row.
    u_dist = {"occupied": 0.22, "temporarily_away": 0.58, "vacant_extended": 0.20}
    u_field, m_field = field_aggregate(u_dist, alpha=7.05)
    bits = disagreement_bits(m_field, u_field)
    assert bits == pytest.approx(-0.191, abs=0.05)


# --- zero evidence on sentinels ----------------------------------------------

@pytest.mark.parametrize("sentinel", [MISSING, ABSENT, UNKNOWABLE])
def test_categorical_sentinel_is_zero_evidence(sentinel):
    assert score_categorical("f", sentinel, "roof", {"roof": 0.1}, 5.0) is None
    assert score_categorical("f", "roof", sentinel, {"roof": 0.1}, 5.0) is None


@pytest.mark.parametrize("sentinel", [MISSING, ABSENT, UNKNOWABLE])
def test_tag_sentinel_is_zero_evidence(sentinel):
    assert score_tag("f", sentinel, ["gold"], {"gold": 0.1}, 5.0) == []
    assert score_tag("f", ["gold"], sentinel, {"gold": 0.1}, 5.0) == []


def test_unknown_value_not_in_pool_is_zero_evidence():
    # value not seen in the frequency pool at all (u undefined) -> no contribution
    assert score_categorical("f", "roof", "roof", {}, 5.0) is None


# --- categorical agreement / disagreement ------------------------------------

def test_categorical_agreement_contribution():
    c = score_categorical("entry_point", "roof", "roof", {"roof": 0.06, "door": 0.5}, 7.05,
                          prov_a="source", prov_b="source")
    assert c is not None
    assert c.bits == pytest.approx(1.559, abs=0.05)
    assert c.provenance_a == "source" and c.provenance_b == "source"


def test_categorical_disagreement_contribution():
    u_dist = {"occupied": 0.22, "temporarily_away": 0.58, "vacant_extended": 0.20}
    c = score_categorical("occupancy", "occupied", "temporarily_away", u_dist, 7.05)
    assert c is not None
    assert c.bits < 0


# --- tag fields ---------------------------------------------------------------

def test_tag_mutual_presence_is_agreement():
    contribs = score_tag("property_taken", ["gold", "cash"], ["gold"], {"gold": 0.61, "cash": 0.54}, 7.05)
    by_field = {c.field: c for c in contribs}
    assert by_field["property_taken:gold"].bits == pytest.approx(0.110, abs=0.05)
    assert by_field["property_taken:gold"].bits > 0


def test_tag_present_in_one_only_is_disagreement():
    contribs = score_tag("property_taken", ["gold", "cash"], ["gold"], {"gold": 0.61, "cash": 0.54}, 7.05)
    by_field = {c.field: c for c in contribs}
    assert by_field["property_taken:cash"].bits < 0


def test_tag_absent_from_both_is_skipped():
    # electronics present in neither list, and not even in the u_dist -> no row
    contribs = score_tag("property_taken", ["gold"], ["gold"], {"gold": 0.61}, 7.05)
    assert len(contribs) == 1
    assert contribs[0].field == "property_taken:gold"
