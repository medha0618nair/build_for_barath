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


def ngrams(tokens: list[str], ngram_range: tuple[int, int] = (1, 2)) -> list[str]:
    lo, hi = ngram_range
    out = []
    for n in range(lo, hi + 1):
        if n == 1:
            out.extend(tokens)
        else:
            out.extend("_".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
    return out


class HashingTfidfEmbedder:
    """Local stub: hash word n-grams into `dim` buckets, weight by corpus
    IDF, L2-normalise so cosine similarity is a plain dot product. Not a
    real semantic embedder (no meaning-preserving geometry across
    paraphrases) — a placeholder so the pipeline runs with no AWS
    dependency.

    `max_df` drops boilerplate: a bucket that shows up in more than that
    fraction of documents gets its IDF zeroed out — cheap approximation of
    scikit-learn's `max_df`, coarser here because multiple n-grams can hash
    into the same bucket. `sublinear_tf` uses 1+log(count) instead of raw
    counts, so a term repeated many times in one narrative (the generator's
    filler sentences do this) doesn't dominate the vector.
    """

    def __init__(self, dim: int = 512, seed: int = 0, ngram_range: tuple[int, int] = (1, 2),
                 sublinear_tf: bool = False, max_df: float = 1.0):
        self.dim = dim
        self.seed = seed
        self.ngram_range = ngram_range
        self.sublinear_tf = sublinear_tf
        self.max_df = max_df
        self.idf_: np.ndarray | None = None

    def _hash(self, token: str) -> int:
        # stable across runs (unlike Python's salted hash()), no external state
        h = 2166136261
        for ch in token:
            h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
        return (h ^ self.seed) % self.dim

    def _terms(self, text: str) -> list[str]:
        return ngrams(tokenize(text), self.ngram_range)

    def _counts(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for term in self._terms(text):
            v[self._hash(term)] += 1.0
        if self.sublinear_tf:
            nonzero = v > 0
            v[nonzero] = 1.0 + np.log(v[nonzero])
        return v

    def fit(self, texts: list[str]) -> "HashingTfidfEmbedder":
        df = np.zeros(self.dim, dtype=np.float64)
        for text in texts:
            seen = np.zeros(self.dim, dtype=bool)
            for term in self._terms(text):
                seen[self._hash(term)] = True
            df += seen
        n = max(len(texts), 1)
        idf = np.log((n + 1) / (df + 1)) + 1.0
        idf[df / n > self.max_df] = 0.0  # boilerplate: drop buckets that are everywhere
        self.idf_ = idf
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


class SentenceTransformerEmbedder:
    """Local semantic embedder: sentence-transformers, cached under
    ~/.cache/torch/sentence_transformers after the first run, no AWS. Same
    Embedder interface as the TF-IDF stub and handlers/cohere_embed.py —
    swapping to Cohere later is a config change, not a code change.
    `fit()` is a no-op: the model is pretrained, not fit per-corpus.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 64):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def fit(self, texts: list[str]) -> "SentenceTransformerEmbedder":
        self._load()
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        vectors = model.encode(texts, batch_size=self.batch_size, show_progress_bar=False,
                                normalize_embeddings=True, convert_to_numpy=True)
        return vectors.astype(np.float32)


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
