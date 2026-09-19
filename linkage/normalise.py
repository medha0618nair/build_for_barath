"""Feed rows -> canonical case records (TECHNICAL_SPEC.md §3.1, §6 "Normalise").

Adapters are `config/states.yaml` (column renames, date formats, structurally
absent columns) and `config/vocab.yaml` (value maps) — the same config the
generator reads to render the feeds in the first place, so there is one
source of truth for "how state X records a crime," not a second copy under
config/adapters/.

    python -m linkage.normalise --check      per-state, per-field agreement
                                              against data/final/truth.parquet

pandas is imported lazily, inside the functions that need it (CSV/parquet
loading, --check), not at module level — handlers/ingest.py imports
normalise()/reverse_vocab()/reverse_crime_type() directly and must not pull
pandas into the Lambda package (CLAUDE.md rule 3: numpy only in Lambda).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime
from pathlib import Path

from linkage import config as config_mod
from linkage import schema
from linkage.extract import Extractor, LocalStubExtractor
from linkage.schema import ABSENT, MISSING, UNKNOWABLE

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "final"


# --- reverse vocab (native label -> canonical value) ------------------------

def reverse_vocab(vocab: dict) -> dict:
    """field -> state_code -> native_label -> canonical_value."""
    rev: dict = {}
    for fname, values in vocab["fields"].items():
        rev[fname] = {}
        for canon, labels in values.items():
            for code, label in labels.items():
                if code == "en":
                    continue
                rev[fname].setdefault(code, {})[label] = canon
    return rev


def reverse_crime_type(vocab: dict) -> dict:
    """state_code -> native_label -> canonical crime_type."""
    rev: dict = {}
    for ct, labels in vocab["crime_type"].items():
        for code, label in labels.items():
            if code == "en":
                continue
            rev.setdefault(code, {})[label] = ct
    return rev


# --- time_band, derived from the occurrence window, never a feed column -----

def _band_of_hour(hour: int, bands: dict) -> str:
    for band, (a, b) in bands.items():
        if a < b:
            if a <= hour < b:
                return band
        elif hour >= a or hour < b:
            return band
    raise ValueError(f"hour {hour} not covered by any time_band")


def parse_window(raw_from: str, raw_to: str, feed: dict, corpus_tb: dict) -> tuple[datetime, datetime, str]:
    """Parse the occurrence window and derive time_band from it.

    A date-only string (no time component survived recording) makes the band
    __MISSING__. A window wider than `unknowable_above_hours` makes it
    __UNKNOWABLE__ even though both timestamps are known — the crime could
    have happened at any hour in between.
    """
    try:
        frm = datetime.strptime(raw_from, feed["date_format"])
        to = datetime.strptime(raw_to, feed["date_format"])
    except ValueError:
        frm = datetime.strptime(raw_from, feed["date_only_format"])
        to = datetime.strptime(raw_to, feed["date_only_format"])
        return frm, to, MISSING

    window_h = (to - frm).total_seconds() / 3600
    if window_h > corpus_tb["unknowable_above_hours"]:
        return frm, to, UNKNOWABLE
    mid = frm + (to - frm) / 2
    return frm, to, _band_of_hour(mid.hour, corpus_tb["hours"])


# --- structured-column reads --------------------------------------------------

def _blank(raw) -> bool:
    # `raw != raw` is the dependency-free NaN test (see linkage/generate/render.py's
    # _is_na) — read_field() is on the Lambda-safe path and must not need pandas.
    return raw is None or raw == "" or (isinstance(raw, float) and raw != raw)


def read_field(raw, fname: str, code: str, rev: dict, delimiter: str | None):
    """Native cell -> (canonical value, provenance). MISSING on a blank cell."""
    if _blank(raw):
        return (MISSING if fname not in schema.TAG_FIELDS else MISSING), None
    labels = rev[fname].get(code, {})
    if fname in schema.TAG_FIELDS:
        tokens = raw.split(delimiter) if delimiter else [raw]
        tags = [labels[tok] for tok in tokens]
        tags = [t for t in tags if t != "none_observed"]
        return tags, "source"
    return labels[raw], "source"


# --- one row -> one canonical case ------------------------------------------

def normalise(row: dict, code: str, cfg: dict, rev_fields: dict, rev_ct: dict,
              extractor: Extractor | None = None) -> dict | None:
    """One feed row -> canonical case dict, or None if out of scope (dropped)."""
    st = cfg["states"]["states"][code]
    feed = st["feed"]
    cols = feed["columns"]
    vocab = cfg["vocab"]

    raw_ct = row[cols["crime_type"]]
    crime_type = rev_ct.get(code, {}).get(raw_ct)
    if crime_type is None:
        return None  # out_of_scope_crime_types: never reaches the pipeline

    fir_no = row[cols["fir_no"]]
    registered_at = datetime.strptime(row[cols["registered_at"]], feed["date_format"])
    year = registered_at.year
    case_id = hashlib.sha1(f"{code}{fir_no}{year}".encode()).hexdigest()[:16]

    frm, to, time_band = parse_window(row[cols["occurred_from"]], row[cols["occurred_to"]],
                                       feed, cfg["corpus"]["time_band"])

    narrative_text = row.get(cols.get("narrative"), "") or ""

    mo: dict[str, object] = {}
    provenance: dict[str, str | None] = {}
    for f in schema.mo_fields(crime_type):
        if f in schema.DERIVED_AT_NORMALISATION:
            continue
        if feed["layout"] == "free_text_mo":
            mo[f], provenance[f] = MISSING, None  # filled below, once, from extraction
        elif f in (st["recording"]["structurally_absent"] or []):
            mo[f], provenance[f] = ABSENT, None
        else:
            raw = row.get(cols[f])
            mo[f], provenance[f] = read_field(raw, f, code, rev_fields, feed["multi_value_delimiter"])

    if feed["layout"] == "free_text_mo" and extractor is not None:
        source_text = row.get(cols.get(schema.MO_DESCRIPTION)) or narrative_text
        extracted = extractor(source_text, crime_type, vocab)
        for f, value in extracted.items():
            if f in mo:
                mo[f] = value
                provenance[f] = "llm_extracted"

    mo["time_band"] = time_band
    provenance["time_band"] = "source" if time_band not in (MISSING, UNKNOWABLE) else None

    family = schema.FAMILY[crime_type]
    return {
        "case_id": case_id,
        "state_code": code,
        "district": row.get(cols.get("district")),
        "fir_no": fir_no,
        "crime_type": crime_type,
        "occurred_from": frm,
        "occurred_to": to,
        "registered_at": registered_at,
        "mo_core": {f: mo[f] for f in schema.MO_CORE},
        "mo_ext": {f: mo[f] for f in schema.MO_EXT[family]},
        "narrative_text": narrative_text,
        "narrative_lang": st["narrative_language"],
        "field_provenance": dict(provenance),
    }


def normalise_feed(code: str, feed_df: pd.DataFrame, cfg: dict, rev_fields: dict, rev_ct: dict,
                    extractor: Extractor | None = None):
    for row in feed_df.to_dict("records"):
        case = normalise(row, code, cfg, rev_fields, rev_ct, extractor)
        if case is not None:
            yield case


def normalise_all(cfg: dict, data_dir: Path, extractor: Extractor | None = None) -> list[dict]:
    """Every feed, normalised, in one list — the corpus the scorer trains
    frequencies and priors against. Offline/batch only (reads with pandas);
    handlers/ingest.py processes one feed file per S3 event with csv.DictReader
    instead, so the Lambda package never needs pandas."""
    import pandas as pd

    rev_fields, rev_ct = reverse_vocab(cfg["vocab"]), reverse_crime_type(cfg["vocab"])
    cases = []
    for code in cfg["states"]["states"]:
        feed_df = pd.read_csv(data_dir / "feeds" / f"{code}.csv", dtype=str, keep_default_na=False)
        cases.extend(normalise_feed(code, feed_df, cfg, rev_fields, rev_ct, extractor))
    return cases


# --- gate: agreement against the generator's answer key ---------------------

# Fields the normaliser reads straight from a structured column for a given
# state — the ones the 95% agreement gate actually applies to. Extracted
# (free-text) and structurally-absent fields are reported but not gated: they
# were never "mapped" in the first place.
def mapped_fields(code: str, cfg: dict) -> list[str]:
    st = cfg["states"]["states"][code]
    if st["feed"]["layout"] == "free_text_mo":
        return ["crime_type"]
    absent = set(st["recording"]["structurally_absent"] or [])
    return ["crime_type", *(f for f in schema.ALL_MO_FIELDS if f not in absent)]


def _truth_value(truth_row: pd.Series, field: str):
    if field == "crime_type":
        return truth_row["rec_crime_type"]
    v = truth_row[f"rec_{field}"]
    if field in schema.TAG_FIELDS and isinstance(v, str) and v not in (MISSING, ABSENT, UNKNOWABLE):
        return [] if v == "" else v.split(";")
    return v


def _case_value(case: dict, field: str):
    if field == "crime_type":
        return case["crime_type"]
    if field == "time_band":
        return case["mo_core"]["time_band"]
    return case["mo_core"].get(field, case["mo_ext"].get(field))


def check(data_dir: Path, cfg: dict, extractor: Extractor | None = None) -> dict:
    import pandas as pd

    truth = pd.read_parquet(data_dir / "truth.parquet").set_index("case_id")
    vocab = cfg["vocab"]
    rev_fields, rev_ct = reverse_vocab(vocab), reverse_crime_type(vocab)

    results: dict[str, dict[str, tuple[int, int]]] = {}
    for code in cfg["states"]["states"]:
        feed_df = pd.read_csv(data_dir / "feeds" / f"{code}.csv", dtype=str, keep_default_na=False)
        cases = list(normalise_feed(code, feed_df, cfg, rev_fields, rev_ct, extractor))
        fields = mapped_fields(code, cfg)
        agree = {f: [0, 0] for f in fields}
        for case in cases:
            if case["case_id"] not in truth.index:
                continue
            trow = truth.loc[case["case_id"]]
            for f in fields:
                truth_v = _truth_value(trow, f)
                if truth_v is None or (not isinstance(truth_v, list) and pd.isna(truth_v)):
                    continue  # field not applicable to this case's crime type
                got = _case_value(case, f)
                agree[f][1] += 1
                if got == truth_v:
                    agree[f][0] += 1
        results[code] = {f: tuple(v) for f, v in agree.items()}
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m linkage.normalise", description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="agreement against truth.parquet rec_* columns")
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--config-dir", type=Path, default=config_mod.CONFIG_DIR)
    ap.add_argument("--threshold", type=float, default=0.95)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    if not args.check:
        ap.print_help()
        return 1

    cfg = config_mod.load(args.config_dir)
    readiness = config_mod.check(cfg)
    if readiness.errors:
        print("config has errors — run `python -m linkage.config -v`")
        return 1

    results = check(args.data_dir, cfg, LocalStubExtractor())

    ok = True
    for code, fields in results.items():
        print(f"\n{code}")
        for f, (hit, total) in fields.items():
            rate = hit / total if total else float("nan")
            flag = ""
            if total and rate < args.threshold:
                flag = "  <-- below threshold"
                ok = False
            print(f"  {f:<20} {hit:>6}/{total:<6} {rate:6.1%}{flag}")

    print(f"\n{'PASS' if ok else 'FAIL'} (threshold {args.threshold:.0%} on mapped fields)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
