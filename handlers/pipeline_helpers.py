"""Step Functions Task helpers that enumerate what a Map state needs
(TECHNICAL_SPEC.md §6) — DynamoDB has no "list all keys matching X" a Map
state's ItemsPath can call directly, so these small Task steps do it
between Enrich/Link and their Map states in
infra/statemachine/pipeline.asl.json.

Run under LinkRole (read-only `cases` access is all either needs; neither
touches `pii` or Bedrock).
"""
from __future__ import annotations

import os

import boto3
from boto3.dynamodb.conditions import Attr

_dynamodb = boto3.resource("dynamodb")
CASES_TABLE = os.environ.get("CASES_TABLE", "crime-linkage-cases")
LINK_CHUNK_SIZE = 1000


def _scan_case_ids(filter_expr) -> list[str]:
    table = _dynamodb.Table(CASES_TABLE)
    kwargs = {"ProjectionExpression": "case_id", "FilterExpression": filter_expr}
    ids: list[str] = []
    while True:
        resp = table.scan(**kwargs)
        ids.extend(item["case_id"] for item in resp["Items"])
        if "LastEvaluatedKey" not in resp:
            break
        kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
    return ids


def list_enrich_targets(event, context):
    """case_ids from free_text_mo feeds — set by handlers/ingest.py."""
    ids = _scan_case_ids(Attr("sk").eq("META") & Attr("needs_enrichment").eq(True))
    return {"case_ids": [{"case_id": cid} for cid in ids]}


def list_link_chunks(event, context):
    """Every case_id, split into 1,000-case chunks for the Link Map state
    (TECHNICAL_SPEC.md §6: "Process 1,000 query rows at a time")."""
    ids = _scan_case_ids(Attr("sk").eq("META"))
    chunks = [{"case_ids": ids[i:i + LINK_CHUNK_SIZE]} for i in range(0, len(ids), LINK_CHUNK_SIZE)]
    return {"chunks": chunks}
