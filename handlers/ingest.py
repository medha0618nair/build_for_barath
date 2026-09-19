"""Normalise Lambda: one feed file -> canonical cases + pii pointers
(TECHNICAL_SPEC.md §6 "Normalise", §3.2 `cases`/`pii` tables).

IngestRole reaches S3 (raw/curated) and `cases` only — no `pii` read/decrypt,
no Bedrock (TECHNICAL_SPEC.md §7 IAM table). Writing a pointer plus a tiny
recording-metadata stub to `pii` below needs *write* access, which
IngestRole has; *reading* pii back is PiiRole's job, gated by Verified
Permissions (handlers/pii.py) — a different capability, not granted here.

This corpus carries no real PII (CLAUDE.md: no protected attributes, and
the generator never modelled complainant/victim identity at all) — the
`pii` table write is a placeholder demonstrating the pointer + separate-
table + separate-KMS-key pattern TECHNICAL_SPEC.md §3.1/§7 requires
structurally, not a claim that real personal data lives there yet.

Extraction from free-text narratives (TG-style feeds) is deliberately NOT
done here — that's the Enrich step's job (handlers/enrich.py), run behind
Step Functions' MaxConcurrency-5 Map so Bedrock's TPM limits are respected
independently of ingest throughput. Cases from free-text feeds are written
with their MO fields __MISSING__ and get updated in place once Enrich runs.
"""
from __future__ import annotations

import csv
import io
import os
import uuid

import boto3

from handlers import config_loader
from linkage.normalise import normalise, reverse_crime_type, reverse_vocab

_s3 = boto3.client("s3")
_dynamodb = boto3.resource("dynamodb")

CASES_TABLE = os.environ.get("CASES_TABLE", "crime-linkage-cases")
PII_TABLE = os.environ.get("PII_TABLE", "crime-linkage-pii")

_rev_cache: dict | None = None


def _reverse_maps(cfg: dict) -> tuple[dict, dict]:
    global _rev_cache
    if _rev_cache is None:
        _rev_cache = (reverse_vocab(cfg["vocab"]), reverse_crime_type(cfg["vocab"]))
    return _rev_cache


def _year_quarter(case: dict) -> str:
    dt = case["registered_at"]
    q = (dt.month - 1) // 3 + 1
    return f"{case['crime_type']}#{dt.year}Q{q}"


def _to_dynamo_safe(item: dict) -> dict:
    out = dict(item)
    for f in ("occurred_from", "occurred_to", "registered_at"):
        if f in out and hasattr(out[f], "isoformat"):
            out[f] = out[f].isoformat()
    return out


def handler(event, context):
    """event: {"bucket": ..., "key": ..., "state_code": "MH"} — one Map
    iteration per raw feed file, from Step Functions' Normalise Map state
    (infra/statemachine/pipeline.asl.json)."""
    bucket, key, code = event["bucket"], event["key"], event["state_code"]
    cfg = config_loader.load()
    rev_fields, rev_ct = _reverse_maps(cfg)

    needs_enrichment = cfg["states"]["states"][code]["feed"]["layout"] == "free_text_mo"

    body = _s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(body))

    cases_table = _dynamodb.Table(CASES_TABLE)
    pii_table = _dynamodb.Table(PII_TABLE)
    written, dropped = 0, 0

    with cases_table.batch_writer() as cases_batch, pii_table.batch_writer() as pii_batch:
        for row in reader:
            # No extractor here — free-text fields stay __MISSING__ until Enrich runs.
            case = normalise(row, code, cfg, rev_fields, rev_ct, extractor=None)
            if case is None:
                dropped += 1  # out_of_scope_crime_types, e.g. ROBBERY
                continue

            pii_ref = str(uuid.uuid4())
            pii_batch.put_item(Item={
                "case_id": case["case_id"],
                "pii_ref": pii_ref,
                "note": "placeholder — this corpus carries no real complainant/victim PII",
            })

            item = _to_dynamo_safe({**case, "sk": "META", "pii_ref": pii_ref,
                                    "crime_type_year_quarter": _year_quarter(case),
                                    "needs_enrichment": needs_enrichment})
            cases_batch.put_item(Item=item)
            written += 1

    return {"state_code": code, "key": key, "written": written, "dropped": dropped}
