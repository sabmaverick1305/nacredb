"""Single-file format.

Version 2 is current. Version 1 files (always flat) can still be read.

Layout (all integers little-endian):

    page 0            header, zero-padded to 4096 bytes
                        0..8    magic "NACREDB\\0"
                        8..12   format_version (u32)
                        12..16  page_size      (u32)
                        16..20  dim            (u32)
                        20      metric         (u8: 0=l2, 2=cosine)
                        21..29  count          (u64)
                        29..37  ids_len        (u64)
                        -- version 2 only --
                        37      kind           (u8: 0=flat, 1=ivf)
                        38..42  nlist          (u32, 0 for flat)
                        42..46  nprobe         (u32, 0 for flat)
    ids section       JSON array of ids (int or str), ids_len bytes,
                      starting right after the header page. For IVF the
                      ids are in partition order.
    ivf section       (IVF only) nlist * dim float32 centroids, then
                      nlist u64 partition sizes
    vectors section   count * dim float32, starting at the next page
                      boundary (page-aligned so it can be mmap'd later).
                      For IVF, vectors are grouped by partition.
"""
import json
import os
import struct
import tempfile
from dataclasses import dataclass

import numpy as np

MAGIC = b"NACREDB\0"
FORMAT_VERSION = 2
PAGE_SIZE = 4096

_METRIC_CODES = {"l2": 0, "cosine": 2}
_METRIC_NAMES = {code: name for name, code in _METRIC_CODES.items()}
_KIND_FLAT, _KIND_IVF = 0, 1
_HEADER_V1 = struct.Struct("<8sIIIBQQ")
_HEADER_V2 = struct.Struct("<8sIIIBQQBII")


class FormatError(ValueError):
    """The file is not a valid (or supported) nacredb file."""


@dataclass
class Contents:
    kind: str                      # "flat" or "ivf"
    dim: int
    metric: str
    ids: list
    vectors: np.ndarray
    nlist: int = 0
    nprobe: int = 0
    centroids: np.ndarray = None   # (nlist, dim) for ivf
    list_sizes: np.ndarray = None  # (nlist,) for ivf


def _round_up_to_page(n):
    return -(-n // PAGE_SIZE) * PAGE_SIZE


def save(path, dim, metric, ids, vectors, ivf=None):
    """Write atomically: temp file in the same directory, fsync, then rename.

    ivf: None for a flat index, or a dict with keys "nprobe", "centroids",
    "list_sizes" for an IVF index (vectors must be grouped by partition).
    """
    ids = list(ids)
    for i in ids:
        if isinstance(i, bool) or not isinstance(i, (int, str)):
            raise TypeError(f"ids must be int or str to be saved, got {type(i).__name__}")

    ids_blob = json.dumps(ids, separators=(",", ":")).encode("utf-8")
    vec = np.ascontiguousarray(vectors, dtype="<f4")

    kind, nlist, nprobe, aux = _KIND_FLAT, 0, 0, b""
    if ivf is not None:
        centroids = np.ascontiguousarray(ivf["centroids"], dtype="<f4")
        sizes = np.ascontiguousarray(ivf["list_sizes"], dtype="<u8")
        if centroids.ndim != 2 or centroids.shape[1] != dim:
            raise ValueError("centroids must have shape (nlist, dim)")
        if len(sizes) != len(centroids) or int(sizes.sum()) != len(ids):
            raise ValueError("list_sizes must have one entry per centroid and sum to len(ids)")
        kind, nlist, nprobe = _KIND_IVF, len(centroids), int(ivf["nprobe"])
        aux = centroids.tobytes() + sizes.tobytes()

    header = _HEADER_V2.pack(
        MAGIC, FORMAT_VERSION, PAGE_SIZE, dim, _METRIC_CODES[metric],
        len(ids), len(ids_blob), kind, nlist, nprobe,
    )
    aux_end = PAGE_SIZE + len(ids_blob) + len(aux)
    vec_offset = _round_up_to_page(aux_end)

    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".nacredb-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(header.ljust(PAGE_SIZE, b"\0"))
            f.write(ids_blob)
            f.write(aux)
            f.write(b"\0" * (vec_offset - aux_end))
            f.write(vec.tobytes())
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def load(path):
    """Return a `Contents`. Raises FormatError on any problem."""
    with open(path, "rb") as f:
        head = f.read(PAGE_SIZE)
        if len(head) < PAGE_SIZE:
            raise FormatError("file too small to be a nacredb file")
        if head[:8] != MAGIC:
            raise FormatError("not a nacredb file")
        (version,) = struct.unpack_from("<I", head, 8)
        if not 1 <= version <= FORMAT_VERSION:
            raise FormatError(f"unsupported format version {version}")

        if version == 1:
            _, _, page_size, dim, code, count, ids_len = _HEADER_V1.unpack_from(head)
            kind_code, nlist, nprobe = _KIND_FLAT, 0, 0
        else:
            (_, _, page_size, dim, code, count, ids_len,
             kind_code, nlist, nprobe) = _HEADER_V2.unpack_from(head)

        if page_size != PAGE_SIZE:
            raise FormatError(f"unsupported page size {page_size}")
        if code not in _METRIC_NAMES:
            raise FormatError(f"unknown metric code {code}")
        if kind_code not in (_KIND_FLAT, _KIND_IVF):
            raise FormatError(f"unknown index kind {kind_code}")
        if (kind_code == _KIND_IVF) != (nlist > 0):
            raise FormatError("nlist does not match index kind")

        aux_len = nlist * (dim * 4 + 8)
        vec_offset = _round_up_to_page(PAGE_SIZE + ids_len + aux_len)
        expected_size = vec_offset + count * dim * 4
        if os.fstat(f.fileno()).st_size != expected_size:
            raise FormatError("file size mismatch (truncated or corrupt)")

        try:
            ids = json.loads(f.read(ids_len).decode("utf-8"))
        except ValueError as e:
            raise FormatError(f"corrupt ids section: {e}") from None
        if not isinstance(ids, list) or len(ids) != count:
            raise FormatError("ids section does not match vector count")

        centroids = sizes = None
        if kind_code == _KIND_IVF:
            centroids = (
                np.frombuffer(f.read(nlist * dim * 4), dtype="<f4")
                .astype(np.float32).reshape(nlist, dim)
            )
            sizes = np.frombuffer(f.read(nlist * 8), dtype="<u8").astype(np.int64)
            if int(sizes.sum()) != count:
                raise FormatError("partition sizes do not match vector count")

        f.seek(vec_offset)
        raw = f.read(count * dim * 4)
    vectors = np.frombuffer(raw, dtype="<f4").astype(np.float32).reshape(count, dim)
    return Contents(
        kind="ivf" if kind_code == _KIND_IVF else "flat",
        dim=dim, metric=_METRIC_NAMES[code], ids=ids, vectors=vectors,
        nlist=nlist, nprobe=nprobe, centroids=centroids, list_sizes=sizes,
    )