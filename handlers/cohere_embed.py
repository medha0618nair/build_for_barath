"""Cohere Embed Multilingual via Bedrock (TECHNICAL_SPEC.md §5).

Implements `linkage.retrieve.Embedder` — fit() is a no-op (Cohere's model
is pretrained and hosted, nothing to fit locally); transform() batches
narratives through Bedrock's invoke_model. Swapping this in for
linkage.retrieve.HashingTfidfEmbedder or SentenceTransformerEmbedder is a
config change (handlers/build_index.py picks the embedder), not a code
change, because all three share the same interface.

MODEL_ID is the UNVERIFIED short form from CLAUDE.md. Verify with
`aws bedrock list-foundation-models --region ap-south-1` (and
`list-inference-profiles` if it's profile-only) before this runs against
real Bedrock — never guess the versioned form.
"""
from __future__ import annotations

import json

import boto3
import numpy as np

REGION = "ap-south-1"
MODEL_ID = "cohere.embed-multilingual-v3"  # UNVERIFIED — see module docstring
BATCH_SIZE = 96  # Cohere Bedrock's per-call text limit


class CohereEmbedder:
    def __init__(self, client=None, region: str = REGION, input_type: str = "search_document"):
        self._client = client or boto3.client("bedrock-runtime", region_name=region)
        self.input_type = input_type

    def fit(self, texts: list[str]) -> "CohereEmbedder":
        return self  # hosted, pretrained — nothing to fit

    def transform(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start:start + BATCH_SIZE]
            body = {"texts": batch, "input_type": self.input_type}
            resp = self._client.invoke_model(modelId=MODEL_ID, body=json.dumps(body))
            payload = json.loads(resp["body"].read())
            vectors.extend(payload["embeddings"])
        return np.array(vectors, dtype=np.float32)
