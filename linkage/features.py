"""Fellegi-Sunter weights (TECHNICAL_SPEC.md §4.1-4.2).

    agreement    weight = log2(m / u)
    disagreement weight = log2((1 - m) / (1 - u))

`u` is a value's recorded frequency in the relevant pool; `m` is that
value's Dirichlet-implied same-offender frequency:

    m_v = (alpha * u_v + 1) / (alpha + 1)

(`m_v` for categorical fields and `m_tag` for multi-label tags are the same
formula algebraically — (1-p)/(a+1) + p = (1 + a*p)/(a+1) — kept as two
names below only because TECHNICAL_SPEC.md §4.2 presents them separately.)

When two cases *disagree* on a categorical field there is no single matched
value to look up an m_v/u_v pair for, so the field falls back to its own
aggregate agreement probabilities, summed over every value in the pool:

    u_field = sum(u_v^2)        # P(two unrelated cases agree on *something*)
    m_field = sum(u_v * m_v)    # P(the same offender repeats their own value)
                                 #   = E[theta_v^2] for theta ~ Dirichlet(alpha*p)

and disagreement bits are log2((1-m_field)/(1-u_field)) — the same formula,
at field level instead of value level. This reproduces TECHNICAL_SPEC.md
§4.3's `occupancy *disagrees*` row (see tests/test_features.py).

Tag fields (schema.TAG_FIELDS) carry one independent rate per tag. Each tag
present in *either* case is scored on its own: mutual presence is an
agreement (log2(m_tag/u_tag)); presence in only one case falls back to the
same disagreement formula, using that tag's own u_tag/m_tag as the "field".
A tag absent from both cases carries no information and is skipped — CLAUDE.md's
__MISSING__/__ABSENT__/__UNKNOWABLE__ "zero evidence" rule, extended to
"never observed at all."

Handles __MISSING__/__ABSENT__/__UNKNOWABLE__ as zero evidence: any field
carrying one of those tokens on either side contributes nothing, never a
disagreement (CLAUDE.md hard rule).
"""
from __future__ import annotations

from dataclasses import dataclass
from math import log2

from linkage.generate.corrupt import ABSENT, MISSING, UNKNOWABLE

SENTINELS = frozenset({ABSENT, MISSING, UNKNOWABLE})


@dataclass
class Contribution:
    field: str
    value_a: object
    value_b: object
    u: float
    m: float
    bits: float
    provenance_a: str | None = None
    provenance_b: str | None = None

    def to_dict(self) -> dict:
        return {
            "field": self.field, "value_a": self.value_a, "value_b": self.value_b,
            "u": round(self.u, 6), "bits": round(self.bits, 6),
            "provenance": {"a": self.provenance_a, "b": self.provenance_b},
        }


def m_of(u: float, alpha: float) -> float:
    """m_v = (alpha*u + 1) / (alpha + 1), theta ~ Dirichlet(alpha*p)."""
    return (alpha * u + 1) / (alpha + 1)


# Same formula, named to match TECHNICAL_SPEC.md §4.2's separate presentation.
m_categorical = m_of
m_tag = m_of


def agreement_bits(m: float, u: float) -> float:
    return log2(m / u)


def disagreement_bits(m: float, u: float) -> float:
    return log2((1 - m) / (1 - u))


def field_aggregate(u_dist: dict[str, float], alpha: float) -> tuple[float, float]:
    """(u_field, m_field): aggregate agreement probabilities used when a
    categorical field's values disagree (see module docstring)."""
    u_field = sum(u * u for u in u_dist.values())
    m_field = sum(u * m_of(u, alpha) for u in u_dist.values())
    return u_field, m_field


def _is_sentinel(value) -> bool:
    return isinstance(value, str) and value in SENTINELS


def score_categorical(field: str, value_a, value_b, u_dist: dict[str, float], alpha: float,
                       prov_a: str | None = None, prov_b: str | None = None) -> Contribution | None:
    if value_a is None or value_b is None or _is_sentinel(value_a) or _is_sentinel(value_b):
        return None
    if value_a == value_b:
        u_v = u_dist.get(value_a)
        if not u_v:
            return None
        m_v = m_of(u_v, alpha)
        return Contribution(field, value_a, value_b, u_v, m_v, agreement_bits(m_v, u_v), prov_a, prov_b)
    u_field, m_field = field_aggregate(u_dist, alpha)
    if not (0 < u_field < 1) or not (0 < m_field < 1):
        return None
    return Contribution(field, value_a, value_b, u_field, m_field,
                         disagreement_bits(m_field, u_field), prov_a, prov_b)


def score_tag(field: str, tags_a, tags_b, u_dist: dict[str, float], alpha: float,
              prov_a: str | None = None, prov_b: str | None = None) -> list[Contribution]:
    if tags_a is None or tags_b is None or _is_sentinel(tags_a) or _is_sentinel(tags_b):
        return []
    set_a, set_b = set(tags_a), set(tags_b)
    out: list[Contribution] = []
    for tag in sorted(set_a | set_b):
        u_t = u_dist.get(tag)
        if not u_t:
            continue
        m_t = m_of(u_t, alpha)
        agree = tag in set_a and tag in set_b
        bits = agreement_bits(m_t, u_t) if agree else disagreement_bits(m_t, u_t)
        out.append(Contribution(f"{field}:{tag}", tag in set_a, tag in set_b, u_t, m_t, bits,
                                 prov_a, prov_b))
    return out
