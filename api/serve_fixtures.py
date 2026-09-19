"""Serves frontend/fixtures/ over the three routes in api/openapi.yaml, so
the API contract can be exercised (and the frontend built against it) with
no database, no AWS, nothing but the fixture files build_fixtures.py wrote.

    python -m api.serve_fixtures --port 8000

    GET  /v1/cases/{id}
    GET  /v1/cases/{id}/links?scope=same|all
    POST /v1/feedback
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "frontend" / "fixtures"

CASE_RE = re.compile(r"^/v1/cases/([^/]+)$")
LINKS_RE = re.compile(r"^/v1/cases/([^/]+)/links$")
FEEDBACK_RE = re.compile(r"^/v1/feedback$")


def make_handler(fixtures_dir: Path):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 (http.server's required name)
            parsed = urlparse(self.path)

            m = LINKS_RE.match(parsed.path)
            if m:
                case_id = m.group(1)
                scope = parse_qs(parsed.query).get("scope", ["same"])[0]
                if scope not in ("same", "all"):
                    self._json(400, {"message": f"scope must be 'same' or 'all', got {scope!r}"})
                    return
                path = fixtures_dir / "links" / f"{case_id}_{scope}.json"
                if not path.exists():
                    self._json(404, {"message": f"no fixture for case {case_id}"})
                    return
                self._json(200, json.loads(path.read_text(encoding="utf-8")))
                return

            m = CASE_RE.match(parsed.path)
            if m:
                case_id = m.group(1)
                path = fixtures_dir / "cases" / f"{case_id}.json"
                if not path.exists():
                    self._json(404, {"message": f"no such case {case_id}"})
                    return
                self._json(200, json.loads(path.read_text(encoding="utf-8")))
                return

            self._json(404, {"message": f"no route for GET {parsed.path}"})

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if not FEEDBACK_RE.match(parsed.path):
                self._json(404, {"message": f"no route for POST {parsed.path}"})
                return
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._json(400, {"message": "invalid JSON body"})
                return
            required = ("case_id_a", "case_id_b", "status", "reason")
            missing = [f for f in required if not body.get(f)]
            if missing:
                self._json(400, {"message": f"missing required field(s): {missing}"})
                return
            if body["status"] not in ("confirmed", "rejected", "unsure"):
                self._json(400, {"message": "status must be confirmed, rejected, or unsure"})
                return
            record = {**body, "pair_id": f"{body['case_id_a']}#{body['case_id_b']}",
                      "submitted_at": "1970-01-01T00:00:00Z"}
            self._json(201, record)

        def log_message(self, fmt: str, *args) -> None:  # quiet by default
            pass

    return Handler


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m api.serve_fixtures", description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--fixtures-dir", type=Path, default=FIXTURES_DIR)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    server = HTTPServer(("127.0.0.1", args.port), make_handler(args.fixtures_dir))
    print(f"serving {args.fixtures_dir} on http://127.0.0.1:{args.port}/v1")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
