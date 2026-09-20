import os
import struct

import pytest

from nacredb import FlatIndex
from nacredb.fileformat import FormatError


def build(ids=("a", "b", "c"), metric="l2"):
    idx = FlatIndex(dim=3, metric=metric)
    idx.add_batch(list(ids), [[0, 0, 0], [1, 1, 1], [5, 5, 5]])
    return idx


def test_roundtrip_preserves_everything(tmp_path):
    path = tmp_path / "db.ndb"
    original = build(metric="cosine")
    original.save(path)

    loaded = FlatIndex.load(path)
    assert (loaded.dim, loaded.metric, len(loaded)) == (3, "cosine", 3)
    q = [0.9, 0.9, 0.9]
    assert loaded.search(q, k=3) == original.search(q, k=3)


def test_int_ids_stay_ints(tmp_path):
    path = tmp_path / "db.ndb"
    build(ids=(10, 20, 30)).save(path)
    hits = FlatIndex.load(path).search([1, 1, 1], k=1)
    assert hits[0][0] == 20 and isinstance(hits[0][0], int)


def test_empty_index_roundtrip(tmp_path):
    path = tmp_path / "empty.ndb"
    FlatIndex(dim=4).save(path)
    loaded = FlatIndex.load(path)
    assert len(loaded) == 0 and loaded.dim == 4


def test_file_starts_with_magic_and_is_page_aligned(tmp_path):
    path = tmp_path / "db.ndb"
    build().save(path)
    data = path.read_bytes()
    assert data[:8] == b"NACREDB\0"
    assert len(data) == 4096 + 4096 + 3 * 3 * 4   # header page + ids page + vectors


def test_rejects_foreign_file(tmp_path):
    path = tmp_path / "junk.ndb"
    path.write_bytes(b"x" * 5000)
    with pytest.raises(FormatError, match="not a nacredb file"):
        FlatIndex.load(path)


def test_rejects_future_version(tmp_path):
    path = tmp_path / "db.ndb"
    build().save(path)
    data = bytearray(path.read_bytes())
    data[8:12] = struct.pack("<I", 999)
    path.write_bytes(bytes(data))
    with pytest.raises(FormatError, match="unsupported format version"):
        FlatIndex.load(path)


def test_rejects_truncated_file(tmp_path):
    path = tmp_path / "db.ndb"
    build().save(path)
    path.write_bytes(path.read_bytes()[:-8])
    with pytest.raises(FormatError, match="size mismatch"):
        FlatIndex.load(path)


def test_failed_save_leaves_old_file_intact_and_no_temp_files(tmp_path):
    path = tmp_path / "db.ndb"
    build().save(path)

    bad = FlatIndex(dim=3)
    bad.add((1, 2), [1, 1, 1])              # tuple ids can't be saved
    with pytest.raises(TypeError):
        bad.save(path)

    assert len(FlatIndex.load(path)) == 3    # old file still good
    assert os.listdir(tmp_path) == ["db.ndb"]  # no leftover temp file