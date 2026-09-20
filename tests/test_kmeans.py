import numpy as np
import pytest

from nacredb.kmeans import assign, kmeans


def blobs(seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[0, 0], [10, 10], [-10, 10]], dtype=np.float32)
    data = np.vstack([c + rng.normal(0, 0.3, (100, 2)) for c in centers])
    return data.astype(np.float32), centers


def test_recovers_well_separated_clusters():
    data, centers = blobs()
    found = kmeans(data, k=3, seed=1)
    for c in centers:                       # every true center has a centroid nearby
        assert np.min(np.linalg.norm(found - c, axis=1)) < 0.5


def test_is_deterministic_for_a_seed():
    data, _ = blobs()
    assert np.array_equal(kmeans(data, 3, seed=7), kmeans(data, 3, seed=7))


def test_assign_matches_naive_nearest():
    rng = np.random.default_rng(0)
    data = rng.random((200, 8), dtype=np.float32)
    cents = rng.random((5, 8), dtype=np.float32)
    naive = [int(np.argmin(((cents - x) ** 2).sum(axis=1))) for x in data]
    assert list(assign(data, cents)) == naive


def test_rejects_bad_k():
    data, _ = blobs()
    with pytest.raises(ValueError):
        kmeans(data, k=0)
    with pytest.raises(ValueError):
        kmeans(data, k=len(data) + 1)


def test_no_nans_when_k_equals_n():
    data = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float32)
    assert not np.isnan(kmeans(data, k=3)).any()