"""Loads config/*.yaml from S3 — as JSON, not YAML.

Lambda deps are numpy only (CLAUDE.md rule 3, restated for this phase: no
sklearn, no pandas, and by the same logic no PyYAML either — every extra
dependency is unzipped-package-size risk for no benefit here). Config is
pure JSON-shaped data, so `scripts/upload_config.py` converts
config/*.yaml -> JSON once at deploy time (where PyYAML runs locally, not
in Lambda) and uploads the JSON; handlers here just call json.loads, a
stdlib module.
"""
from __future__ import annotations

import json
import os

import boto3

_s3 = boto3.client("s3")
CONFIG_BUCKET = os.environ.get("CONFIG_BUCKET")
CONFIG_PREFIX = os.environ.get("CONFIG_PREFIX", "config")

_FILES = ("marginals", "loadings", "states", "corpus", "vocab")
_cache: dict | None = None


def load() -> dict:
    global _cache
    if _cache is not None:
        return _cache
    cfg = {}
    for name in _FILES:
        obj = _s3.get_object(Bucket=CONFIG_BUCKET, Key=f"{CONFIG_PREFIX}/{name}.json")
        cfg[name] = json.loads(obj["Body"].read())
    _cache = cfg
    return cfg
