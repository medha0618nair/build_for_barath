"""Pluggable MO-field extractor for free-text feeds (TECHNICAL_SPEC.md §3.1, §5).

The interface is a callable: `extractor(text, crime_type, vocab) -> {field: value}`.
It returns only the fields it found evidence for; callers treat a missing key
as unextracted (`__MISSING__`). No boto3 here — the Bedrock Claude Haiku
implementation lives in `handlers/`, which calls through this same interface.
"""
from __future__ import annotations

from typing import Protocol

from linkage import schema


class Extractor(Protocol):
    def __call__(self, text: str, crime_type: str, vocab: dict) -> dict[str, str | list[str]]: ...


class LocalStubExtractor:
    """Phrase-matches the vocab's `en` clauses against narrative/MO text.

    This works on the synthetic corpus because narratives and TG's
    `mo_description` are templated from the same `en` phrases stored in
    `config/vocab.yaml` (see `linkage/generate/render.py`). It is not a real
    extractor for genuine FIR prose — a placeholder that lets the pipeline
    run end to end, locally, with no AWS dependency. `handlers/bedrock_extract.py`
    replaces this for real narratives.
    """

    def __call__(self, text: str, crime_type: str, vocab: dict) -> dict[str, str | list[str]]:
        if not text:
            return {}
        t = text.lower()
        out: dict[str, str | list[str]] = {}
        for f in schema.mo_fields(crime_type):
            if f in schema.DERIVED_AT_NORMALISATION:
                continue
            labels = vocab["fields"].get(f, {})
            if f in schema.TAG_FIELDS:
                found = [v for v, entry in labels.items()
                         if v != "none_observed" and entry.get("en", "").lower() in t]
                if found:
                    out[f] = found
                elif labels.get("none_observed", {}).get("en", "").lower() in t:
                    out[f] = []
            else:
                for v, entry in labels.items():
                    if entry.get("en", "").lower() in t:
                        out[f] = v
                        break
        return out
