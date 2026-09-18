from __future__ import annotations

import numpy as np
import pytest

from linkage.retrieve import HashingTfidfEmbedder, embed_corpus, tokenize, top_k


def test_tokenize_lowercases_and_splits():
    assert tokenize("Accused used a CROWBAR!") == ["accused", "used", "a", "crowbar"]


def test_embedder_rows_are_unit_normalised():
    ids = ["a", "b", "c"]
    texts = ["roof entry cctv disabled", "roof entry cctv disabled", "vehicle theft market area"]
    vecs, _ = embed_corpus(ids, texts, HashingTfidfEmbedder(dim=64))
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms[norms > 0], 1.0, atol=1e-5)


def test_identical_narratives_are_most_similar():
    ids = ["a", "b", "c"]
    texts = [
        "the accused got in through the roof and disabled the cctv camera",
        "the accused got in through the roof and disabled the cctv camera",
        "the accused stole a motorcycle from a market area",
    ]
    vecs, _ = embed_corpus(ids, texts, HashingTfidfEmbedder(dim=128))
    neighbours = top_k(vecs, ids, k=2)
    assert neighbours["a"][0][0] == "b"
    assert neighbours["a"][0][1] > neighbours["a"][1][1]


def test_top_k_excludes_self_and_respects_k():
    ids = [f"c{i}" for i in range(10)]
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(10, 16)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    neighbours = top_k(vecs, ids, k=3, chunk_size=4)
    for cid, nbrs in neighbours.items():
        assert len(nbrs) == 3
        assert cid not in {n for n, _ in nbrs}


def test_top_k_chunking_matches_unchunked():
    ids = [f"c{i}" for i in range(20)]
    rng = np.random.default_rng(1)
    vecs = rng.normal(size=(20, 8)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    chunked = top_k(vecs, ids, k=5, chunk_size=3)
    unchunked = top_k(vecs, ids, k=5, chunk_size=1000)
    assert chunked.keys() == unchunked.keys()
    for cid in chunked:
        c_ids = [n for n, _ in chunked[cid]]
        u_ids = [n for n, _ in unchunked[cid]]
        assert c_ids == u_ids
        for (_, c_sim), (_, u_sim) in zip(chunked[cid], unchunked[cid]):
            assert c_sim == pytest.approx(u_sim, abs=1e-4)
