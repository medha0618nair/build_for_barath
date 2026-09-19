"""API Lambda: GET /cases/{id}, GET /cases/{id}/links, POST /feedback
(api/openapi.yaml). Every call writes an audit row with the Cognito
principal (CLAUDE.md; TECHNICAL_SPEC.md §7/§8).

ApiRole reaches `cases`, `links`, `audit`, `feedback` — no `pii` directly
(TECHNICAL_SPEC.md §7 IAM table). A PII request goes through PiiRole via
handlers/pii.py, gated by Verified Permissions, never through this
function — this handler doesn't even have the IAM permissions to try.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import boto3
from boto3.dynamodb.conditions import Key

from handlers import audit

_dynamodb = boto3.resource("dynamodb")
CASES_TABLE = os.environ.get("CASES_TABLE", "crime-linkage-cases")
LINKS_TABLE = os.environ.get("LINKS_TABLE", "crime-linkage-links")
FEEDBACK_TABLE = os.environ.get("FEEDBACK_TABLE", "crime-linkage-feedback")


def _principal(event: dict) -> str:
    claims = event.get("requestContext", {}).get("authorizer", {}).get("jwt", {}).get("claims", {})
    return claims.get("sub", "anonymous")


def _response(status: int, body: dict) -> dict:
    return {"statusCode": status, "headers": {"Content-Type": "application/json"},
            "body": json.dumps(body, default=str)}


def get_case(event, context):
    actor = _principal(event)
    case_id = event["pathParameters"]["id"]
    item = _dynamodb.Table(CASES_TABLE).get_item(Key={"case_id": case_id, "sk": "META"}).get("Item")
    audit.write(actor, "get_case", {"case_id": case_id, "found": item is not None})
    if item is None:
        return _response(404, {"message": f"no such case {case_id}"})
    item = dict(item)
    item.pop("pii_ref", None)  # a pointer only — never returned by the case-record route
    for internal_field in ("sk", "crime_type_year_quarter", "needs_enrichment"):
        item.pop(internal_field, None)
    return _response(200, item)


def get_case_links(event, context):
    actor = _principal(event)
    case_id = event["pathParameters"]["id"]
    params = event.get("queryStringParameters") or {}
    scope = params.get("scope", "same")
    if scope not in ("same", "all"):
        return _response(400, {"message": f"scope must be 'same' or 'all', got {scope!r}"})
    try:
        limit = int(params.get("limit", 10))
    except ValueError:
        return _response(400, {"message": "limit must be an integer"})

    resp = _dynamodb.Table(LINKS_TABLE).query(
        KeyConditionExpression=Key("case_id_a").eq(case_id), Limit=200)
    items = resp.get("Items", [])
    if scope == "same":
        items = [i for i in items if i["pair_class"] == "same_type"]
    items = sorted(items, key=lambda i: -float(i["total_bits"]))[:limit]
    for item in items:
        top = sorted(item.get("contributions", []), key=lambda c: -abs(float(c["bits"])))[:3]
        item["driven_by"] = [f"{c['field']}={c['value_a']}" if c["value_a"] == c["value_b"]
                             else f"{c['field']} disagrees" for c in top]

    audit.write(actor, "get_case_links", {"case_id": case_id, "scope": scope, "n_returned": len(items)})
    return _response(200, {"case_id": case_id, "scope": scope, "links": items})


def post_feedback(event, context):
    actor = _principal(event)
    try:
        body = json.loads(event.get("body") or "{}")
    except json.JSONDecodeError:
        return _response(400, {"message": "invalid JSON body"})

    required = ("case_id_a", "case_id_b", "status", "reason")
    missing = [f for f in required if not body.get(f)]
    if missing:
        return _response(400, {"message": f"missing required field(s): {missing}"})
    if body["status"] not in ("confirmed", "rejected", "unsure"):
        return _response(400, {"message": "status must be confirmed, rejected, or unsure"})

    submitted_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    pair_id = f"{body['case_id_a']}#{body['case_id_b']}"
    record = {**body, "pair_id": pair_id, "actor_id": actor, "submitted_at": submitted_at}
    _dynamodb.Table(FEEDBACK_TABLE).put_item(Item={**record, "sk": f"{actor}#{submitted_at}"})

    audit.write(actor, "post_feedback", {"pair_id": pair_id, "status": body["status"]})
    return _response(201, record)
