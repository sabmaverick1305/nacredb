import numpy as np
import pytest

from nacredb import FlatIndex, IVFIndex


def clustered(n_per=200, dim=8, n_clusters=10, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 10, (n_clusters, dim))
    data = np.vstack([c + rng.normal(0, 1.0, (n_per, dim)) for c in centers])
    return data.astype(np.float32)


def build_pair(nlist=10, nprobe=1):
    data = clustered()
    ids = list(range(len(data)))
    flat = FlatIndex(dim=8)
    flat.add_batch(ids, data)
    ivf = IVFIndex(dim=8, nlist=nlist, nprobe=nprobe)
    ivf.train(data)
    ivf.add_batch(ids, data)
    return data, flat, ivf


def test_probing_every_partition_matches_flat_exactly():
    data, flat, ivf = build_pair(nlist=10)
    for q in data[::250]:
        assert [h[0] for h in ivf.search(q, k=10, nprobe=10)] == [h[0] for h in flat.search(q, k=10)]


def test_few_probes_still_find_neighbors_on_clustered_data():
    data, flat, ivf = build_pair(nlist=10, nprobe=2)
    hit = total = 0
    for q in data[::20]:
        truth = {h[0] for h in flat.search(q, k=10)}
        got = {h[0] for h in ivf.search(q, k=10)}
        hit += len(truth & got)
        total += len(truth)
    assert hit / total > 0.9


def test_len_counts_all_added_vectors():
    _, _, ivf = build_pair()
    assert len(ivf) == 2000


def test_requires_training_first():
    ivf = IVFIndex(dim=8, nlist=4)
    with pytest.raises(RuntimeError):
        ivf.add_batch([1], np.zeros((1, 8)))
    with pytest.raises(RuntimeError):
        ivf.search(np.zeros(8))


def test_rejects_duplicates_and_bad_shapes():
    data, _, ivf = build_pair()
    with pytest.raises(ValueError):
        ivf.add_batch([0], data[:1])                  # id 0 already exists
    with pytest.raises(ValueError):
        ivf.add_batch([9999], np.zeros((1, 3)))       # wrong dim
    with pytest.raises(ValueError):
        ivf.search(np.zeros(3))


def test_empty_index_returns_nothing():
    ivf = IVFIndex(dim=8, nlist=2)
    ivf.train(clustered(n_per=5, n_clusters=2))
    assert ivf.search(np.zeros(8)) == []