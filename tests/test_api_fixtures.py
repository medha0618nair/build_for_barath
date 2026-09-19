from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer

import pytest

from api.serve_fixtures import make_handler


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    fixtures_dir = tmp_path_factory.mktemp("fixtures")
    (fixtures_dir / "cases").mkdir()
    (fixtures_dir / "links").mkdir()

    case = {"case_id": "c1", "state_code": "MH", "crime_type": "BURGLARY_RESIDENTIAL",
            "mo_core": {"time_band": "night"}, "mo_ext": {}, "narrative_text": "...",
            "narrative_lang": "en", "field_provenance": {}}
    (fixtures_dir / "cases" / "c1.json").write_text(json.dumps(case), encoding="utf-8")

    links_same = {"case_id": "c1", "scope": "same", "links": [
        {"partner_id": "c2", "rank": 1, "total_bits": 3.5, "prior_bits": -15.0,
         "pair_class": "same_type", "driven_by": ["entry_point=roof"], "contributions": []},
    ]}
    (fixtures_dir / "links" / "c1_same.json").write_text(json.dumps(links_same), encoding="utf-8")
    links_all = {**links_same, "scope": "all", "links": []}
    (fixtures_dir / "links" / "c1_all.json").write_text(json.dumps(links_all), encoding="utf-8")

    httpd = HTTPServer(("127.0.0.1", 0), make_handler(fixtures_dir))
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    httpd.shutdown()
    thread.join(timeout=5)


def _get(url: str) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(url) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _post(url: str, body: dict) -> tuple[int, dict]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_get_case(server):
    status, body = _get(f"{server}/v1/cases/c1")
    assert status == 200
    assert body["case_id"] == "c1"


def test_get_case_not_found(server):
    status, body = _get(f"{server}/v1/cases/nope")
    assert status == 404
    assert "message" in body


def test_get_links_default_scope_same(server):
    status, body = _get(f"{server}/v1/cases/c1/links")
    assert status == 200
    assert body["scope"] == "same"
    assert body["links"][0]["partner_id"] == "c2"


def test_get_links_scope_all(server):
    status, body = _get(f"{server}/v1/cases/c1/links?scope=all")
    assert status == 200
    assert body["scope"] == "all"


def test_get_links_invalid_scope(server):
    status, body = _get(f"{server}/v1/cases/c1/links?scope=bogus")
    assert status == 400


def test_post_feedback_success(server):
    status, body = _post(f"{server}/v1/feedback", {
        "case_id_a": "c1", "case_id_b": "c2", "status": "confirmed", "reason": "matches MO",
    })
    assert status == 201
    assert body["pair_id"] == "c1#c2"


def test_post_feedback_missing_reason(server):
    status, body = _post(f"{server}/v1/feedback", {
        "case_id_a": "c1", "case_id_b": "c2", "status": "confirmed", "reason": "",
    })
    assert status == 400


def test_post_feedback_invalid_status(server):
    status, body = _post(f"{server}/v1/feedback", {
        "case_id_a": "c1", "case_id_b": "c2", "status": "maybe", "reason": "x",
    })
    assert status == 400
