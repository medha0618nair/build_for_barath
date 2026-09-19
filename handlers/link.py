"""Link Lambda: chunked retrieval + FS scoring for one 1,000-case chunk
(TECHNICAL_SPEC.md §6 "LinkBatch", §4). One Step Functions Map iteration =
one chunk of up to 1,000 query case_ids; this function retrieves each
query's top-50 by embedding cosine against the full corpus's vector index,
scores every candidate, and writes the ranked result to `links`.

LinkRole reaches `cases`, vectors, `links` — no `pii`, no Bedrock
(TECHNICAL_SPEC.md §7 IAM table). Inference is pure numpy from
frequencies.json/priors.json/weights.json (CLAUDE.md rule 3) — this
package ships numpy only, no scikit-learn, no pandas.

The full corpus's vector index and case_ids are cached at module scope, so
a warm Lambda container reuses them across Map iterations within the same
execution instead of re-fetching from S3 every time.
"""
from __future__ import annotations

import io
import json
import os
from decimal import Decimal

import boto3
import numpy as np

from linkage.score import score_pair

_s3 = boto3.client("s3")
_dynamodb = boto3.resource("dynamodb")

CASES_TABLE = os.environ.get("CASES_TABLE", "crime-linkage-cases")
LINKS_TABLE = os.environ.get("LINKS_TABLE", "crime-linkage-links")
VECTORS_BUCKET = os.environ.get("VECTORS_BUCKET")
VECTORS_PREFIX = os.environ.get("VECTORS_PREFIX", "vectors")
MODEL_BUCKET = os.environ.get("MODEL_BUCKET")
MODEL_PREFIX = os.environ.get("MODEL_PREFIX", "model")

TOP_K = 50
_cache: dict = {}


def _load_json_s3(bucket: str, key: str):
    return json.loads(_s3.get_object(Bucket=bucket, Key=key)["Body"].read())


def _context() -> dict:
    if not _cache:
        buf = io.BytesIO(_s3.get_object(Bucket=VECTORS_BUCKET,
                                        Key=f"{VECTORS_PREFIX}/index.npy")["Body"].read())
        _cache["vectors"] = np.load(buf)
        _cache["ids"] = _load_json_s3(VECTORS_BUCKET, f"{VECTORS_PREFIX}/case_ids.json")
        _cache["id_index"] = {cid: i for i, cid in enumerate(_cache["ids"])}
        _cache["freqs"] = _load_json_s3(MODEL_BUCKET, f"{MODEL_PREFIX}/frequencies.json")
        _cache["priors"] = _load_json_s3(MODEL_BUCKET, f"{MODEL_PREFIX}/priors.json")
        _cache["weights"] = _load_json_s3(MODEL_BUCKET, f"{MODEL_PREFIX}/weights.json")
    return _cache


def _get_case(case_id: str) -> dict | None:
    return _dynamodb.Table(CASES_TABLE).get_item(Key={"case_id": case_id, "sk": "META"}).get("Item")


def _bits_desc_sk(total_bits: float, partner_id: str) -> str:
    # Lexicographic SK order must equal descending total_bits (TECHNICAL_SPEC.md
    # §3.2: `links` PK case_id_a SK bits_desc#case_id_b).
    encoded = max(0, min(999999, 999999 - int(round((total_bits + 100) * 1000))))
    return f"{encoded:06d}#{partner_id}"


def _num(x) -> Decimal | None:
    return None if x is None else Decimal(str(round(float(x), 6)))


def _contribution_item(c) -> dict:
    return {"field": c.field, "value_a": c.value_a, "value_b": c.value_b,
            "u": _num(c.u), "bits": _num(c.bits),
            "provenance": {"a": c.provenance_a, "b": c.provenance_b}}


def handler(event, context):
    """event: {"case_ids": [...up to 1000...]}"""
    ctx = _context()
    vectors, ids, id_index = ctx["vectors"], ctx["ids"], ctx["id_index"]
    freqs, priors, weights = ctx["freqs"], ctx["priors"], ctx["weights"]
    alpha_same, alpha_cross = weights["alpha_same"], weights["alpha_cross"]

    chunk_ids = [c for c in event["case_ids"] if c in id_index]
    chunk_idx = [id_index[c] for c in chunk_ids]
    if not chunk_idx:
        return {"n_queries": 0, "n_links_written": 0}

    block = vectors[chunk_idx]
    sims = block @ vectors.T  # (chunk, n) — one chunk resident at a time, never the full n x n matrix

    links_table = _dynamodb.Table(LINKS_TABLE)
    written = 0
    case_cache: dict[str, dict | None] = {}

    for row_offset, global_row in enumerate(chunk_idx):
        row = sims[row_offset].copy()
        row[global_row] = -np.inf
        k_eff = min(TOP_K, len(ids) - 1)
        top_idx = np.argpartition(row, -k_eff)[-k_eff:]
        top_idx = top_idx[np.argsort(-row[top_idx])]

        query_id = ids[global_row]
        case_a = case_cache.setdefault(query_id, _get_case(query_id))
        if case_a is None:
            continue

        scored = []
        for j in top_idx:
            partner_id = ids[j]
            case_b = case_cache.get(partner_id)
            if partner_id not in case_cache:
                case_b = case_cache[partner_id] = _get_case(partner_id)
            if case_b is None:
                continue
            result = score_pair(case_a, case_b, freqs, priors, alpha_same, alpha_cross)
            scored.append((partner_id, result))
        scored.sort(key=lambda t: -t[1]["total_bits"])

        with links_table.batch_writer() as batch:
            for rank, (partner_id, result) in enumerate(scored, start=1):
                batch.put_item(Item={
                    "case_id_a": query_id,
                    "sk": _bits_desc_sk(result["total_bits"], partner_id),
                    "partner_id": partner_id,
                    "rank": rank,
                    "total_bits": _num(result["total_bits"]),
                    "prior_bits": _num(result["prior_bits"]),
                    "pair_class": result["pair_class"],
                    "contributions": [_contribution_item(c) for c in result["contributions"]],
                })
                written += 1

    return {"n_queries": len(chunk_idx), "n_links_written": written}
