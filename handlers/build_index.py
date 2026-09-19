"""BuildIndex Lambda: embed every case's narrative, write one vectors.npy
to S3 (TECHNICAL_SPEC.md §6 "BuildIndex", §5 embedding model).

Retrieval uses narrative embedding cosine ONLY (CLAUDE.md hard rule 1) —
this step reads narrative_text and nothing else from `cases`.

Runs after Enrich in the Step Functions pipeline
(infra/statemachine/pipeline.asl.json), once per pipeline execution, not
per case — a single Lambda invocation, not a Map state.
"""
from __future__ import annotations

import io
import json
import os

import boto3
import numpy as np
from boto3.dynamodb.conditions import Attr

from handlers.cohere_embed import CohereEmbedder

_dynamodb = boto3.resource("dynamodb")
_s3 = boto3.client("s3")

CASES_TABLE = os.environ.get("CASES_TABLE", "crime-linkage-cases")
VECTORS_BUCKET = os.environ.get("VECTORS_BUCKET")
VECTORS_PREFIX = os.environ.get("VECTORS_PREFIX", "vectors")


def _scan_narratives() -> tuple[list[str], list[str]]:
    table = _dynamodb.Table(CASES_TABLE)
    ids: list[str] = []
    texts: list[str] = []
    kwargs = {"ProjectionExpression": "case_id, narrative_text", "FilterExpression": Attr("sk").eq("META")}
    while True:
        resp = table.scan(**kwargs)
        for item in resp["Items"]:
            ids.append(item["case_id"])
            texts.append(item.get("narrative_text", ""))
        if "LastEvaluatedKey" not in resp:
            break
        kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
    return ids, texts


def handler(event, context):
    ids, texts = _scan_narratives()
    if not ids:
        return {"n_cases": 0}

    embedder = CohereEmbedder()
    embedder.fit(texts)
    vectors = embedder.transform(texts)

    buf = io.BytesIO()
    np.save(buf, vectors)
    _s3.put_object(Bucket=VECTORS_BUCKET, Key=f"{VECTORS_PREFIX}/index.npy", Body=buf.getvalue())
    _s3.put_object(Bucket=VECTORS_BUCKET, Key=f"{VECTORS_PREFIX}/case_ids.json",
                   Body=json.dumps(ids).encode("utf-8"), ContentType="application/json")

    return {"n_cases": len(ids), "dim": int(vectors.shape[1])}
