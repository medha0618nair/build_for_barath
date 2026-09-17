from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from linkage import config as config_mod
from linkage import schema
from linkage.extract import LocalStubExtractor
from linkage.generate.corrupt import ABSENT, MISSING, UNKNOWABLE
from linkage.normalise import (
    normalise,
    normalise_feed,
    parse_window,
    read_field,
    reverse_crime_type,
    reverse_vocab,
)


@pytest.fixture(scope="module")
def cfg():
    cfg = config_mod.load()
    readiness = config_mod.check(cfg)
    assert not readiness.errors, readiness.errors
    return cfg


@pytest.fixture(scope="module")
def rev(cfg):
    return reverse_vocab(cfg["vocab"]), reverse_crime_type(cfg["vocab"])


# --- case_id -----------------------------------------------------------------

def test_case_id_matches_dataset_formula(cfg, rev):
    rev_fields, rev_ct = rev
    row = {
        "FIR_NO": "MH-KOL-04/0001/2021", "DISTRICT": "Kolhapur", "POLICE_STATION": "MH-KOL-04",
        "CRIME_HEAD": "SHOP BREAKING",
        "OCCURRENCE_FROM": "01/01/2021 02:54", "OCCURRENCE_TO": "01/01/2021 09:30",
        "REGISTRATION_DATE": "01/01/2021 21:52",
        "NO_OF_ACCUSED": "2 TO 3", "WEAPONS_TOOLS": "NIL", "CONCEALMENT": "",
        "TARGET_TYPE": "INSIDE INFORMATION", "PROPERTY_STOLEN": "DOCUMENTS",
        "MODE_OF_ARRIVAL": "TWO WHEELER", "MODE_OF_ESCAPE": "TWO WHEELER",
        "PLACE_OF_ENTRY": "SHUTTER", "MODE_OF_ENTRY": "",
        "PREMISES_TYPE": "GODOWN", "PREMISES_STATUS": "CLOSED FOR NIGHT", "SEARCH_TYPE": "RANSACKED",
        "VEHICLE_TYPE": "", "VEHICLE_START_METHOD": "", "PARKING_PLACE": "",
        "OFFENDER_VEHICLE": "", "VICTIM_ACTIVITY": "", "ESCAPE_ROUTE": "",
        "ATM_SITE_TYPE": "", "ATM_ATTACK": "", "ALARM_STATUS": "",
        "FIR_CONTENTS": "some narrative",
    }
    case = normalise(row, "MH", cfg, rev_fields, rev_ct)
    expected = hashlib.sha1(b"MHMH-KOL-04/0001/20212021").hexdigest()[:16]
    assert case["case_id"] == expected
    assert case["crime_type"] == "BURGLARY_COMMERCIAL"


def test_out_of_scope_crime_type_is_dropped(cfg, rev):
    rev_fields, rev_ct = rev
    row = {"FIR_NO": "x", "CRIME_HEAD": "not a real label", "DISTRICT": "Pune",
           "POLICE_STATION": "MH-PUN-01", "OCCURRENCE_FROM": "01/01/2021 02:54",
           "OCCURRENCE_TO": "01/01/2021 03:00", "REGISTRATION_DATE": "01/01/2021 21:52",
           "FIR_CONTENTS": ""}
    assert normalise(row, "MH", cfg, rev_fields, rev_ct) is None


# --- time_band derivation ------------------------------------------------------

def test_time_band_missing_on_date_only():
    feed = {"date_format": "%d/%m/%Y %H:%M", "date_only_format": "%d/%m/%Y"}
    corpus_tb = {"hours": {"night": [22, 4], "early_morning": [4, 7], "day": [7, 22]},
                 "unknowable_above_hours": 12}
    _, _, band = parse_window("01/01/2021", "02/01/2021", feed, corpus_tb)
    assert band == MISSING


def test_time_band_unknowable_on_wide_window():
    feed = {"date_format": "%d/%m/%Y %H:%M", "date_only_format": "%d/%m/%Y"}
    corpus_tb = {"hours": {"night": [22, 4], "early_morning": [4, 7], "day": [7, 22]},
                 "unknowable_above_hours": 12}
    _, _, band = parse_window("01/01/2021 08:00", "02/01/2021 20:00", feed, corpus_tb)
    assert band == UNKNOWABLE


@pytest.mark.parametrize("frm,to,expected", [
    ("01/01/2021 23:00", "01/01/2021 23:30", "night"),   # wraps midnight, both sides late
    ("01/01/2021 05:00", "01/01/2021 05:30", "early_morning"),
    ("01/01/2021 12:00", "01/01/2021 12:30", "day"),
])
def test_time_band_maps_midpoint_hour(frm, to, expected):
    feed = {"date_format": "%d/%m/%Y %H:%M", "date_only_format": "%d/%m/%Y"}
    corpus_tb = {"hours": {"night": [22, 4], "early_morning": [4, 7], "day": [7, 22]},
                 "unknowable_above_hours": 12}
    _, _, band = parse_window(frm, to, feed, corpus_tb)
    assert band == expected


# --- structured field reads ---------------------------------------------------

def test_read_field_blank_is_missing(cfg, rev):
    rev_fields, _ = rev
    value, prov = read_field("", "counter_forensic", "MH", rev_fields, ";")
    assert value == MISSING
    assert prov is None


def test_read_field_tag_none_observed_is_empty_list(cfg, rev):
    rev_fields, _ = rev
    value, prov = read_field("NIL", "tools", "MH", rev_fields, ";")
    assert value == []
    assert prov == "source"


def test_read_field_tag_splits_and_maps(cfg, rev):
    rev_fields, _ = rev
    value, prov = read_field("GOLD ORNAMENTS;CASH", "property_taken", "MH", rev_fields, ";")
    assert value == ["gold", "cash"]
    assert prov == "source"


def test_read_field_categorical_maps(cfg, rev):
    rev_fields, _ = rev
    value, prov = read_field("SELECTIVE", "search_pattern", "MH", rev_fields, ";")
    assert value == "selective"
    assert prov == "source"


# --- structurally absent (KA has no `occupancy` column) -----------------------

def test_structurally_absent_field_is_absent_token(cfg, rev):
    rev_fields, rev_ct = rev
    cols = cfg["states"]["states"]["KA"]["feed"]["columns"]
    row = {native: "" for native in cols.values()}
    row[cols["crime_type"]] = "Burglary - Dwelling"
    row[cols["fir_no"]] = "KA-BEN-01/0001/2021"
    row[cols["occurred_from"]] = "01-Jan-2021 0200 hrs"
    row[cols["occurred_to"]] = "01-Jan-2021 0230 hrs"
    row[cols["registered_at"]] = "01-Jan-2021 0900 hrs"
    case = normalise(row, "KA", cfg, rev_fields, rev_ct)
    assert case is not None
    assert case["mo_ext"]["occupancy"] == ABSENT
    assert case["field_provenance"]["occupancy"] is None


# --- free-text layout (TG) uses the extractor ----------------------------------

def test_free_text_layout_uses_extractor(cfg, rev):
    rev_fields, rev_ct = rev
    cols = cfg["states"]["states"]["TG"]["feed"]["columns"]
    row = {native: "" for native in cols.values()}
    row[cols["crime_type"]] = "MV Theft"
    row[cols["fir_no"]] = "TG-HYD-01/0001/2021"
    row[cols["occurred_from"]] = "2021-01-01T20:00:00"
    row[cols["occurred_to"]] = "2021-01-01T20:30:00"
    row[cols["registered_at"]] = "2021-01-01T21:00:00"
    row[cols["mo_description"]] = "Accused used a crowbar; stole a motorcycle; acted alone."
    row[cols["narrative"]] = row[cols["mo_description"]]
    case = normalise(row, "TG", cfg, rev_fields, rev_ct, LocalStubExtractor())
    assert case["mo_core"]["tools"] == ["crowbar"]
    assert case["field_provenance"]["tools"] == "llm_extracted"
    assert case["mo_ext"]["vehicle_class"] == "motorcycle"


def test_free_text_layout_without_extractor_is_missing(cfg, rev):
    rev_fields, rev_ct = rev
    cols = cfg["states"]["states"]["TG"]["feed"]["columns"]
    row = {native: "" for native in cols.values()}
    row[cols["crime_type"]] = "MV Theft"
    row[cols["fir_no"]] = "TG-HYD-02/0001/2021"
    row[cols["occurred_from"]] = "2021-01-01T20:00:00"
    row[cols["occurred_to"]] = "2021-01-01T20:30:00"
    row[cols["registered_at"]] = "2021-01-01T21:00:00"
    case = normalise(row, "TG", cfg, rev_fields, rev_ct, extractor=None)
    assert case["mo_core"]["tools"] == MISSING
    assert case["field_provenance"]["tools"] is None


# --- integration: real corpus agrees with the answer key ----------------------

def test_agreement_against_truth_sample(cfg, rev):
    rev_fields, rev_ct = rev
    truth = pd.read_parquet("data/final/truth.parquet").set_index("case_id")
    feed_df = pd.read_csv("data/final/feeds/MH.csv", dtype=str, keep_default_na=False).head(200)
    cases = list(normalise_feed("MH", feed_df, cfg, rev_fields, rev_ct))
    assert cases
    matches = 0
    for case in cases:
        trow = truth.loc[case["case_id"]]
        matches += case["crime_type"] == trow["rec_crime_type"]
    assert matches == len(cases)
