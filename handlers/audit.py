"""Shared audit-row writer (TECHNICAL_SPEC.md §3.2 `audit` table).

Every API call writes one row here with the Cognito principal — CLAUDE.md's
"the linkage engine cannot see personal data" claim is an IAM boundary plus
a separate KMS key, not a promise; the audit trail is what makes every
access to `cases`/`links`/`pii` attributable to an actor after the fact.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import boto3

_dynamodb = boto3.resource("dynamodb")
AUDIT_TABLE = os.environ.get("AUDIT_TABLE", "crime-linkage-audit")


def write(actor_id: str, action: str, detail: dict | None = None) -> None:
    _dynamodb.Table(AUDIT_TABLE).put_item(Item={
        "actor_id": actor_id,
        "iso_timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "action": action,
        "detail": detail or {},
    })
