import pytest

from nacredb import FlatIndex

def make_index(metric="l2"):
    idx = FlatIndex(dim=3, metric=metric)
    idx.add("a", [0, 0, 0])
    idx.add("b", [1, 1, 1])
    idx.add("c", [5, 5, 5])
    return idx


def test_finds_nearest_first():
    hits = make_index().search([0.9, 0.9, 0.9], k=2)
    assert [h[0] for h in hits] == ["b", "a"]


def test_k_larger_than_size_returns_all():
    assert len(make_index().search([0, 0, 0], k=10)) == 3


def test_empty_index_returns_nothing():
    assert FlatIndex(dim=3).search([0, 0, 0]) == []


def test_rejects_wrong_dimension():
    with pytest.raises(ValueError):
        make_index().add("d", [1, 2])


def test_rejects_duplicate_id():
    with pytest.raises(ValueError):
        make_index().add("a", [9, 9, 9])


def test_cosine_ignores_magnitude():
    idx = FlatIndex(dim=2, metric="cosine")
    idx.add("a", [1, 0])
    idx.add("b", [10, 0])
    idx.add("c", [0, 1])
    hits = idx.search([2, 0], k=3)
    assert {hits[0][0], hits[1][0]} == {"a", "b"}
    assert hits[2][0] == "c"

def test_add_batch_matches_single_adds():
    idx = FlatIndex(dim=3)
    idx.add_batch(["a", "b", "c"], [[0, 0, 0], [1, 1, 1], [5, 5, 5]])
    hits = idx.search([0.9, 0.9, 0.9], k=2)
    assert [h[0] for h in hits] == ["b", "a"]


def test_add_batch_rejects_duplicates():
    idx = make_index()
    with pytest.raises(ValueError):
        idx.add_batch(["x", "a"], [[1, 1, 1], [2, 2, 2]])
    with pytest.raises(ValueError):
        idx.add_batch(["y", "y"], [[1, 1, 1], [2, 2, 2]])

def test_matches_naive_computation_on_random_data():
    import numpy as np

    rng = np.random.default_rng(0)
    data = rng.random((500, 16), dtype=np.float32)
    q = rng.random(16, dtype=np.float32)
    idx = FlatIndex(dim=16, metric="l2")
    idx.add_batch(range(500), data)

    naive = ((data - q) ** 2).sum(axis=1)
    expected = list(np.argsort(naive)[:10])
    assert [h[0] for h in idx.search(q, k=10)] == expected