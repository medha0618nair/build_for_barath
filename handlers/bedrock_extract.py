"""Bedrock Claude Haiku MO extractor (TECHNICAL_SPEC.md §5, §6).

Implements the `linkage.extract.Extractor` interface. boto3 lives here, never
in `linkage/` (CLAUDE.md rule 2) — `linkage/normalise.py` takes any callable
matching that interface and doesn't know this one talks to AWS.

MODEL_ID is the UNVERIFIED short form from CLAUDE.md. Before this runs against
real Bedrock, replace it with exactly what
`aws bedrock list-foundation-models --region ap-south-1` (and, if it's an
inference-profile-only model, `aws bedrock list-inference-profiles`) returns.
Never guess the versioned form.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

import boto3

from linkage import schema

REGION = "ap-south-1"
MODEL_ID = "anthropic.claude-haiku-4-5"  # UNVERIFIED — see module docstring
MAX_CONCURRENCY = 5  # Bedrock TPM limits, matches the Enrich Map state's MaxConcurrency


def _prompt(text: str, crime_type: str, vocab: dict) -> str:
    fields = [f for f in schema.mo_fields(crime_type) if f not in schema.DERIVED_AT_NORMALISATION]
    enum = {f: sorted(v for v in vocab["fields"][f] if v != "none_observed") for f in fields}
    tag_fields = sorted(schema.TAG_FIELDS & set(fields))
    return (
        "You extract structured modus-operandi fields from an Indian police FIR narrative.\n"
        "Return ONLY a JSON object mapping each field below to one of its allowed values, "
        "or null if the narrative gives no evidence for that field. "
        f"These fields take a JSON list of values instead of one: {tag_fields}. "
        "Never invent a value outside the enum. Never include any text besides the JSON object.\n\n"
        f"Allowed values per field: {json.dumps(enum)}\n\n"
        f"Narrative:\n{text}"
    )


def _invoke(client, text: str, crime_type: str, vocab: dict) -> dict:
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 512,
        "messages": [{"role": "user", "content": _prompt(text, crime_type, vocab)}],
    }
    resp = client.invoke_model(modelId=MODEL_ID, body=json.dumps(body))
    payload = json.loads(resp["body"].read())
    return json.loads(payload["content"][0]["text"])


def _validate(parsed: dict, fields: set[str], vocab: dict) -> dict[str, str | list[str]]:
    out: dict[str, str | list[str]] = {}
    for f, value in parsed.items():
        if f not in fields or value is None:
            continue
        allowed = set(vocab["fields"][f]) - {"none_observed"}
        if f in schema.TAG_FIELDS:
            if isinstance(value, list) and all(v in allowed for v in value):
                out[f] = value
        elif value in allowed:
            out[f] = value
    return out


class BedrockHaikuExtractor:
    """`linkage.extract.Extractor`: strict JSON, one retry on a bad response."""

    def __init__(self, client=None, region: str = REGION):
        self._client = client or boto3.client("bedrock-runtime", region_name=region)

    def __call__(self, text: str, crime_type: str, vocab: dict) -> dict[str, str | list[str]]:
        fields = set(schema.mo_fields(crime_type)) - set(schema.DERIVED_AT_NORMALISATION)
        for attempt in range(2):
            try:
                parsed = _invoke(self._client, text, crime_type, vocab)
                return _validate(parsed, fields, vocab)
            except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                if attempt == 1:
                    return {}
        return {}


def extract_many(items: Iterable[tuple[str, str, dict]], region: str = REGION,
                  max_concurrency: int = MAX_CONCURRENCY) -> list[dict]:
    """(text, crime_type, vocab) triples -> extracted dicts, capped at
    MaxConcurrency 5 to respect Bedrock TPM limits (TECHNICAL_SPEC.md §6)."""
    extractor = BedrockHaikuExtractor(region=region)
    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        return list(pool.map(lambda item: extractor(*item), items))
