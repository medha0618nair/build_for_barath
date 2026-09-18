"""Narrative retrieval: embed, then chunked cosine top-50 (TECHNICAL_SPEC.md §4.6, §6).

Retrieval uses narrative embedding cosine ONLY — never structured MO fields,
never location (CLAUDE.md hard rule 1: filtering by a feature you also score
inflates the prior by exactly that feature's agreement weight).

Default embedder is a local hashing TF-IDF stub: no AWS, no vocabulary file
to ship, pure numpy. `handlers/cohere_embed.py` swaps in Cohere Embed
Multilingual for real narratives (TECHNICAL_SPEC.md §5) behind the same
`Embedder` interface.

Similarity is chunked: 1,000 query rows at a time against the full corpus,
top 50 kept, the rest of the block discarded — a full 45k x 45k matrix would
be ~8 GB of float32, never materialised (TECHNICAL_SPEC.md §6).
"""
from __future__ import annotations

import re
from typing import Protocol

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class Embedder(Protocol):
    def fit(self, texts: list[str]) -> "Embedder": ...
    def transform(self, texts: list[str]) -> np.ndarray: ...


class HashingTfidfEmbedder:
    """Local stub: hash tokens into `dim` buckets, weight by corpus IDF,
    L2-normalise so cosine similarity is a plain dot product. Not a real
    semantic embedder (no meaning-preserving geometry across paraphrases) —
    a placeholder so the pipeline runs with no AWS dependency.
    """

    def __init__(self, dim: int = 512, seed: int = 0):
        self.dim = dim
        self.seed = seed
        self.idf_: np.ndarray | None = None

    def _hash(self, token: str) -> int:
        # stable across runs (unlike Python's salted hash()), no external state
        h = 2166136261
        for ch in token:
            h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
        return (h ^ self.seed) % self.dim

    def _counts(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            v[self._hash(tok)] += 1.0
        return v

    def fit(self, texts: list[str]) -> "HashingTfidfEmbedder":
        df = np.zeros(self.dim, dtype=np.float64)
        for text in texts:
            seen = np.zeros(self.dim, dtype=bool)
            for tok in tokenize(text):
                seen[self._hash(tok)] = True
            df += seen
        n = max(len(texts), 1)
        self.idf_ = np.log((n + 1) / (df + 1)) + 1.0
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        if self.idf_ is None:
            raise RuntimeError("call fit() before transform()")
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            tf = self._counts(text)
            vec = tf * self.idf_
            norm = np.linalg.norm(vec)
            out[i] = vec / norm if norm > 0 else vec
        return out


def embed_corpus(case_ids: list[str], narratives: list[str], embedder: Embedder | None = None,
                  dim: int = 512) -> tuple[np.ndarray, Embedder]:
    embedder = embedder or HashingTfidfEmbedder(dim=dim)
    embedder.fit(narratives)
    vectors = embedder.transform(narratives)
    return vectors, embedder


def top_k(vectors: np.ndarray, case_ids: list[str], k: int = 50,
          chunk_size: int = 1000) -> dict[str, list[tuple[str, float]]]:
    """Chunked cosine top-k, excluding self. vectors must be L2-normalised
    rows so `chunk @ vectors.T` is cosine similarity directly."""
    n = len(case_ids)
    out: dict[str, list[tuple[str, float]]] = {}
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        block = vectors[start:end]                    # (chunk, dim)
        sims = block @ vectors.T                       # (chunk, n) — one chunk resident at a time
        for row_offset, global_row in enumerate(range(start, end)):
            row = sims[row_offset]
            row = row.copy()
            row[global_row] = -np.inf                  # exclude self
            k_eff = min(k, n - 1)
            idx = np.argpartition(row, -k_eff)[-k_eff:]
            idx = idx[np.argsort(-row[idx])]
            out[case_ids[global_row]] = [(case_ids[i], float(row[i])) for i in idx]
        del sims
    return out
