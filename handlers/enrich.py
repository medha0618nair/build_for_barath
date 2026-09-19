"""Enrich Lambda: LLM MO-field extraction for free-text feeds
(TECHNICAL_SPEC.md §6 "Enrich", §5). Runs behind Step Functions' Enrich Map
state at MaxConcurrency 5 (infra/statemachine/pipeline.asl.json) to respect
Bedrock TPM limits independently of Ingest's own throughput.

EnrichRole reaches Bedrock, `cases`, S3 vectors — no `pii`
(TECHNICAL_SPEC.md §7 IAM table).
"""
from __future__ import annotations

import os

import boto3

from handlers import config_loader
from handlers.bedrock_extract import BedrockHaikuExtractor
from linkage import schema

_dynamodb = boto3.resource("dynamodb")
CASES_TABLE = os.environ.get("CASES_TABLE", "crime-linkage-cases")

_extractor: BedrockHaikuExtractor | None = None


def _get_extractor() -> BedrockHaikuExtractor:
    global _extractor
    if _extractor is None:
        _extractor = BedrockHaikuExtractor()
    return _extractor


def handler(event, context):
    """event: {"case_id": ...} — one Map iteration per case whose feed
    layout is free_text_mo (Ingest leaves its MO fields __MISSING__)."""
    case_id = event["case_id"]
    table = _dynamodb.Table(CASES_TABLE)
    item = table.get_item(Key={"case_id": case_id, "sk": "META"}).get("Item")
    if item is None:
        return {"case_id": case_id, "extracted": 0}

    cfg = config_loader.load()
    extractor = _get_extractor()
    text = item.get("narrative_text", "")
    extracted = extractor(text, item["crime_type"], cfg["vocab"])
    if not extracted:
        return {"case_id": case_id, "extracted": 0}

    mo_core, mo_ext = dict(item.get("mo_core") or {}), dict(item.get("mo_ext") or {})
    provenance = dict(item.get("field_provenance") or {})
    for field, value in extracted.items():
        if field in schema.MO_CORE:
            mo_core[field] = value
        elif field in mo_ext:
            mo_ext[field] = value
        else:
            continue
        provenance[field] = "llm_extracted"

    table.update_item(
        Key={"case_id": case_id, "sk": "META"},
        UpdateExpression="SET mo_core = :c, mo_ext = :e, field_provenance = :p",
        ExpressionAttributeValues={":c": mo_core, ":e": mo_ext, ":p": provenance},
    )
    return {"case_id": case_id, "extracted": len(extracted)}
