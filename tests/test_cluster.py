from __future__ import annotations

from linkage.cluster import connected_components


def test_strong_edges_form_a_cluster():
    edges = [("a", "b", 15.0), ("b", "c", 12.0)]
    clusters = connected_components(edges, threshold=10.0)
    assert len(clusters) == 1
    assert clusters[0]["members"] == ["a", "b", "c"]
    assert clusters[0]["edges"] == 2


def test_weak_edges_are_dropped_not_chained():
    # a-b strong, b-c weak: b-c must not merge a and c transitively.
    edges = [("a", "b", 15.0), ("b", "c", 3.0)]
    clusters = connected_components(edges, threshold=10.0)
    assert len(clusters) == 1
    assert clusters[0]["members"] == ["a", "b"]


def test_singletons_are_not_clusters():
    edges = [("a", "b", 2.0)]
    assert connected_components(edges, threshold=10.0) == []


def test_mean_bits_computed_correctly():
    edges = [("a", "b", 10.0), ("a", "c", 20.0)]
    clusters = connected_components(edges, threshold=10.0)
    assert clusters[0]["mean_bits"] == 15.0


def test_larger_clusters_sort_first():
    edges = [("a", "b", 12.0), ("c", "d", 12.0), ("c", "e", 12.0)]
    clusters = connected_components(edges, threshold=10.0)
    assert clusters[0]["members"] == ["c", "d", "e"]
