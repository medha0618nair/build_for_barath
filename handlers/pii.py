"""PII reader (TECHNICAL_SPEC.md §7 `PiiRole`).

The ONLY function in this system that runs under PiiRole, which reaches
`pii` and its KMS key and nothing else (TECHNICAL_SPEC.md §7 IAM table).
Not exposed on the HTTP API directly and not reachable by API Gateway —
its resource policy permits invocation only from
handlers/pii_request.py's function (ApiRole), which is the one place that
has already checked Verified Permissions for a stated reason before
calling this. No AVP client here: this role isn't granted
verifiedpermissions:IsAuthorized at all (see infra/template.yaml's
PiiRole policy), so even a bug here couldn't skip the gate.
"""
from __future__ import annotations

import os

import boto3

_dynamodb = boto3.resource("dynamodb")
PII_TABLE = os.environ.get("PII_TABLE", "crime-linkage-pii")


def handler(event, context):
    """event: {"case_id": ...} — a direct Lambda invoke, not API Gateway."""
    case_id = event["case_id"]
    item = _dynamodb.Table(PII_TABLE).get_item(Key={"case_id": case_id}).get("Item")
    if item is None:
        return {"statusCode": 404, "body": {"message": f"no pii record for {case_id}"}}
    return {"statusCode": 200, "body": item}
