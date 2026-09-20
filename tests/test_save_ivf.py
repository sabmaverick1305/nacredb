import json
import struct

import numpy as np
import pytest

from nacredb import FlatIndex, IVFIndex
from nacredb.fileformat import FormatError


def clustered(n_per=100, dim=8, n_clusters=6, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 10, (n_clusters, dim))
    return np.vstack([c + rng.normal(0, 1.0, (n_per, dim)) for c in centers]).astype(np.float32)


def build_ivf(nlist=6, nprobe=2):
    data = clustered()
    ivf = IVFIndex(dim=8, nlist=nlist, nprobe=nprobe)
    ivf.train(data)
    ivf.add_batch([f"v{i}" for i in range(len(data))], data)
    return data, ivf


def test_roundtrip_gives_identical_search_results(tmp_path):
    data, ivf = build_ivf()
    path = tmp_path / "ivf.ndb"
    ivf.save(path)

    loaded = IVFIndex.load(path)
    assert (loaded.dim, loaded.nlist, loaded.nprobe, len(loaded)) == (8, 6, 2, len(ivf))
    for q in data[::97]:
        for nprobe in (1, 2, 6):
            assert loaded.search(q, k=5, nprobe=nprobe) == ivf.search(q, k=5, nprobe=nprobe)


def test_loaded_index_keeps_working(tmp_path):
    data, ivf = build_ivf()
    path = tmp_path / "ivf.ndb"
    ivf.save(path)
    loaded = IVFIndex.load(path)

    loaded.add_batch(["new"], data[:1])                     # can still add
    assert loaded.search(data[0], k=1)[0][1] < 1e-3
    with pytest.raises(ValueError):
        loaded.add_batch(["v0"], data[:1])                  # old ids still rejected


def test_empty_partitions_survive_roundtrip(tmp_path):
    data = clustered()
    ivf = IVFIndex(dim=8, nlist=6)
    ivf.train(data)
    ivf.add_batch(["a", "b"], data[:2])                     # most partitions stay empty
    path = tmp_path / "sparse.ndb"
    ivf.save(path)
    loaded = IVFIndex.load(path)
    assert len(loaded) == 2
    assert loaded.search(data[0], k=1, nprobe=6)[0][0] == "a"


def test_untrained_index_cannot_be_saved(tmp_path):
    with pytest.raises(RuntimeError):
        IVFIndex(dim=8, nlist=4).save(tmp_path / "x.ndb")


def test_wrong_loader_gives_clear_error(tmp_path):
    _, ivf = build_ivf()
    ivf.save(tmp_path / "ivf.ndb")
    flat = FlatIndex(dim=3)
    flat.add("a", [1, 2, 3])
    flat.save(tmp_path / "flat.ndb")

    with pytest.raises(FormatError, match="IVFIndex.load"):
        FlatIndex.load(tmp_path / "ivf.ndb")
    with pytest.raises(FormatError, match="FlatIndex.load"):
        IVFIndex.load(tmp_path / "flat.ndb")


def test_truncated_ivf_file_is_rejected(tmp_path):
    _, ivf = build_ivf()
    path = tmp_path / "ivf.ndb"
    ivf.save(path)
    path.write_bytes(path.read_bytes()[:-16])
    with pytest.raises(FormatError, match="size mismatch"):
        IVFIndex.load(path)


def test_version_1_flat_files_still_load(tmp_path):
    """A file exactly as step 4 wrote it (format version 1)."""
    ids = ["a", "b"]
    vectors = np.array([[1, 2, 3], [4, 5, 6]], dtype="<f4")
    ids_blob = json.dumps(ids, separators=(",", ":")).encode()
    header = struct.pack("<8sIIIBQQ", b"NACREDB\0", 1, 4096, 3, 0, 2, len(ids_blob))
    body = header.ljust(4096, b"\0") + ids_blob
    body += b"\0" * (4096 - len(ids_blob))          # pad to the next page boundary
    body += vectors.tobytes()
    path = tmp_path / "old.ndb"
    path.write_bytes(body)

    loaded = FlatIndex.load(path)
    assert len(loaded) == 2
    assert loaded.search([4, 5, 6], k=1)[0][0] == "b"