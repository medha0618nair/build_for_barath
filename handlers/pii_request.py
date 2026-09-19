"""PII request gate (TECHNICAL_SPEC.md §7, §8 `POST /cases/{id}/pii-request`).

Runs under ApiRole, which the IAM table (TECHNICAL_SPEC.md §7) lists as
reaching Verified Permissions — NOT `pii` directly. This function's whole
job is: call AVP's IsAuthorized with the stated reason, and only on ALLOW
invoke handlers/pii_reader.py's function, which is the one thing actually
running under PiiRole. On DENY it never touches `pii` at all; there's no
"try anyway" code path, because ApiRole's IAM policy doesn't grant it read
access to `pii` in the first place — the boundary is enforced twice
(here, and structurally by IAM), matching CLAUDE.md's "it holds even if
the Lambda code has a bug."
"""
from __future__ import annotations

import json
import os

import boto3

from handlers import audit

_avp = boto3.client("verifiedpermissions")
_lambda = boto3.client("lambda")

POLICY_STORE_ID = os.environ.get("POLICY_STORE_ID")
PII_READER_FUNCTION = os.environ.get("PII_READER_FUNCTION")


def _principal(event: dict) -> str:
    claims = event.get("requestContext", {}).get("authorizer", {}).get("jwt", {}).get("claims", {})
    return claims.get("sub", "anonymous")


def _response(status: int, body: dict) -> dict:
    return {"statusCode": status, "headers": {"Content-Type": "application/json"},
            "body": json.dumps(body, default=str)}


def handler(event, context):
    actor = _principal(event)
    case_id = event["pathParameters"]["id"]
    try:
        body = json.loads(event.get("body") or "{}")
    except json.JSONDecodeError:
        return _response(400, {"message": "invalid JSON body"})
    reason = (body.get("reason") or "").strip()

    decision = _avp.is_authorized(
        policyStoreId=POLICY_STORE_ID,
        principal={"entityType": "CrimeLinkage::User", "entityId": actor},
        action={"actionType": "CrimeLinkage::Action", "actionId": "AssumePiiRole"},
        resource={"entityType": "CrimeLinkage::Case", "entityId": case_id},
        context={"contextMap": {"reason": {"string": reason}}},
    )
    allowed = decision["decision"] == "ALLOW"
    audit.write(actor, "pii_request", {"case_id": case_id, "decision": decision["decision"], "reason": reason})

    if not allowed:
        return _response(403, {"message": "PII access denied — a non-empty reason is required",
                               "determiningPolicies": decision.get("determiningPolicies", [])})

    invoke_resp = _lambda.invoke(FunctionName=PII_READER_FUNCTION, Payload=json.dumps({"case_id": case_id}))
    payload = json.loads(invoke_resp["Payload"].read())
    return _response(payload.get("statusCode", 200), payload.get("body", {}))
