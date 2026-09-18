"""DIAGNOSTIC ONLY — never imported by linkage/retrieve.py, linkage/train.py,
or linkage/sweep.py, and never wired into Context's default retrieval path.

OracleMoFieldEmbedder vectorises the structured MO fields (one-hot per
field=value) directly. That VIOLATES CLAUDE.md hard rule 1 — retrieval must
use narrative embedding cosine only, never a field that's also scored,
because filtering/retrieving by a scored field inflates the prior by
exactly that field's own agreement weight. This embedder exists solely to
measure retrieval's *ceiling*: how much recall@k is achievable if blocking
could see the MO fields directly, as an upper bound to compare narrative
embedders against — not a candidate for the pipeline.
"""
from __future__ import annotations

import numpy as np

from linkage.features import SENTINELS


def _is_sentinel(value) -> bool:
    return isinstance(value, str) and value in SENTINELS


def _keys(case: dict) -> list[str]:
    keys = []
    for section in ("mo_core", "mo_ext"):
        for field, value in case.get(section, {}).items():
            if value is None or _is_sentinel(value):
                continue
            if isinstance(value, list):
                keys.extend(f"{field}={v}" for v in value)
            else:
                keys.append(f"{field}={value}")
    return keys


class OracleMoFieldEmbedder:
    def __init__(self):
        self.index_: dict[str, int] = {}

    def fit_cases(self, cases: list[dict]) -> "OracleMoFieldEmbedder":
        keys = sorted({k for case in cases for k in _keys(case)})
        self.index_ = {k: i for i, k in enumerate(keys)}
        return self

    def transform_cases(self, cases: list[dict]) -> np.ndarray:
        out = np.zeros((len(cases), len(self.index_)), dtype=np.float32)
        for i, case in enumerate(cases):
            for k in _keys(case):
                j = self.index_.get(k)
                if j is not None:
                    out[i, j] = 1.0
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        return out / norm


def oracle_vectors(cases: list[dict]) -> np.ndarray:
    embedder = OracleMoFieldEmbedder().fit_cases(cases)
    return embedder.transform_cases(cases)
