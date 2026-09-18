"""Connected components over scored edges (TECHNICAL_SPEC.md §4.7).

The edge threshold is deliberately high: 10 bits. That's roughly two-thirds
of the way from "no evidence" to the burglary same-type prior's magnitude
(~15 bits) closed by evidence alone — i.e. it takes several rare-value
agreements (an entry_point + a counter_forensic hit is already +3.9 bits in
the §4.3 worked example) before an edge survives at all. A low threshold
lets transitive chaining (A-B linked, B-C linked, A-C unrelated) merge
unrelated offenders into one bogus mega-cluster; requiring strong pairwise
evidence on *every* edge in the component keeps that from happening.
Mean intra-cluster score is reported alongside membership so an analyst can
see how confident the whole cluster is, not just that it exists.
"""
from __future__ import annotations

from collections import defaultdict

EDGE_THRESHOLD_BITS = 10.0


def connected_components(edges: list[tuple[str, str, float]],
                          threshold: float = EDGE_THRESHOLD_BITS) -> list[dict]:
    """edges: (case_a, case_b, total_bits). Returns clusters with >=2 members,
    each {'members': [...], 'edges': n, 'mean_bits': ...}, largest first."""
    strong = [(a, b, bits) for a, b, bits in edges if bits >= threshold]

    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b, _ in strong:
        union(a, b)

    members: dict[str, set[str]] = defaultdict(set)
    for a, b, _ in strong:
        members[find(a)].add(a)
        members[find(a)].add(b)

    bits_sum: dict[str, float] = defaultdict(float)
    bits_n: dict[str, int] = defaultdict(int)
    for a, b, bits in strong:
        root = find(a)
        bits_sum[root] += bits
        bits_n[root] += 1

    clusters = [
        {"members": sorted(ids), "edges": bits_n[root], "mean_bits": bits_sum[root] / bits_n[root]}
        for root, ids in members.items() if len(ids) >= 2
    ]
    clusters.sort(key=lambda c: (-len(c["members"]), -c["mean_bits"]))
    return clusters
