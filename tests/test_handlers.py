from __future__ import annotations

import json
import os

# handlers/*.py construct boto3 clients/resources at *import* time (module
# scope) — a region must exist before any `import handlers.X` happens, in
# every test here, not just the ones that also mock DynamoDB.
os.environ.setdefault("AWS_DEFAULT_REGION", "ap-south-1")

import boto3
import pytest
from moto import mock_aws


def test_bits_desc_sk_orders_descending_bits():
    from handlers.link import _bits_desc_sk

    high = _bits_desc_sk(10.0, "partner-a")
    mid = _bits_desc_sk(2.0, "partner-b")
    low = _bits_desc_sk(-5.0, "partner-c")
    # lexicographic string order must equal descending numeric bits order
    assert sorted([low, mid, high]) == [high, mid, low]


def test_bits_desc_sk_clamps_extreme_values():
    from handlers.link import _bits_desc_sk

    # shouldn't raise or produce a negative/overflowing encoded prefix
    sk_low = _bits_desc_sk(-1000.0, "x")
    sk_high = _bits_desc_sk(1000.0, "x")
    assert sk_low > sk_high  # very negative bits sort last


def test_api_response_shape():
    from handlers.api import _response

    resp = _response(404, {"message": "no such case"})
    assert resp["statusCode"] == 404
    assert json.loads(resp["body"]) == {"message": "no such case"}


def test_api_principal_falls_back_to_anonymous():
    from handlers.api import _principal

    assert _principal({}) == "anonymous"
    event = {"requestContext": {"authorizer": {"jwt": {"claims": {"sub": "user-123"}}}}}
    assert _principal(event) == "user-123"


@pytest.fixture
def dynamodb_env(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "ap-south-1")
    with mock_aws():
        client = boto3.client("dynamodb", region_name="ap-south-1")
        client.create_table(
            TableName="test-cases",
            AttributeDefinitions=[{"AttributeName": "case_id", "AttributeType": "S"},
                                  {"AttributeName": "sk", "AttributeType": "S"}],
            KeySchema=[{"AttributeName": "case_id", "KeyType": "HASH"},
                      {"AttributeName": "sk", "KeyType": "RANGE"}],
            BillingMode="PAY_PER_REQUEST",
        )
        client.create_table(
            TableName="test-audit",
            AttributeDefinitions=[{"AttributeName": "actor_id", "AttributeType": "S"},
                                  {"AttributeName": "iso_timestamp", "AttributeType": "S"}],
            KeySchema=[{"AttributeName": "actor_id", "KeyType": "HASH"},
                      {"AttributeName": "iso_timestamp", "KeyType": "RANGE"}],
            BillingMode="PAY_PER_REQUEST",
        )
        monkeypatch.setenv("CASES_TABLE", "test-cases")
        monkeypatch.setenv("AUDIT_TABLE", "test-audit")
        yield


def test_api_get_case_round_trip(dynamodb_env):
    import importlib

    import handlers.api as api
    import handlers.audit as audit
    importlib.reload(audit)
    importlib.reload(api)

    table = boto3.resource("dynamodb", region_name="ap-south-1").Table("test-cases")
    table.put_item(Item={"case_id": "c1", "sk": "META", "crime_type": "VEHICLE_THEFT",
                         "mo_core": {"time_band": "night"}, "mo_ext": {}, "pii_ref": "should-not-leak",
                         "needs_enrichment": False})

    resp = api.get_case({"pathParameters": {"id": "c1"}}, None)
    assert resp["statusCode"] == 200
    body = json.loads(resp["body"])
    assert body["case_id"] == "c1"
    assert "pii_ref" not in body
    assert "needs_enrichment" not in body

    audit_table = boto3.resource("dynamodb", region_name="ap-south-1").Table("test-audit")
    rows = audit_table.scan()["Items"]
    assert len(rows) == 1
    assert rows[0]["action"] == "get_case"


def test_api_get_case_not_found(dynamodb_env):
    import importlib

    import handlers.api as api
    importlib.reload(api)

    resp = api.get_case({"pathParameters": {"id": "nope"}}, None)
    assert resp["statusCode"] == 404
