from __future__ import annotations

import numpy as np
import pytest

from linkage.retrieve import HashingTfidfEmbedder, SentenceTransformerEmbedder, embed_corpus, ngrams, tokenize, top_k


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


def test_ngrams_default_range_is_unigrams_and_bigrams():
    assert ngrams(["a", "b", "c"]) == ["a", "b", "c", "a_b", "b_c"]


def test_ngrams_unigrams_only():
    assert ngrams(["a", "b", "c"], ngram_range=(1, 1)) == ["a", "b", "c"]


def test_max_df_zeroes_out_boilerplate_terms():
    # "the" and "accused" appear in every doc; "roof" only in one.
    texts = [
        "the accused entered through the roof",
        "the accused disabled the cctv camera",
        "the accused fled on a motorcycle",
    ]
    embedder = HashingTfidfEmbedder(dim=256, ngram_range=(1, 1), max_df=0.5)
    embedder.fit(texts)
    boilerplate_bucket = embedder._hash("the")
    rare_bucket = embedder._hash("roof")
    assert embedder.idf_[boilerplate_bucket] == 0.0
    assert embedder.idf_[rare_bucket] > 0.0


def test_sublinear_tf_dampens_repeated_terms_no_warning():
    embedder = HashingTfidfEmbedder(dim=64, sublinear_tf=True)
    embedder.fit(["roof roof roof roof roof", "door"])
    # a term repeated 5x should count for less than 5x a term seen once,
    # once both are log-dampened
    counts = embedder._counts("roof roof roof roof roof")
    assert counts.max() == pytest.approx(1 + np.log(5))


def test_sentence_transformer_embedder_same_interface():
    pytest.importorskip("sentence_transformers")
    ids = ["a", "b", "c"]
    texts = [
        "the accused entered through the roof and disabled the cctv camera",
        "the burglar climbed in via the roof and cut the cctv wires",
        "the accused stole a motorcycle from a market area",
    ]
    vecs, _ = embed_corpus(ids, texts, SentenceTransformerEmbedder())
    assert vecs.shape[0] == 3
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)
    neighbours = top_k(vecs, ids, k=2)
    # semantically closer paraphrase (a, b) should beat the unrelated one (c)
    assert neighbours["a"][0][0] == "b"


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
