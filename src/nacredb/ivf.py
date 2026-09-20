import numpy as np

from . import fileformat
from .kmeans import assign, kmeans


class IVFIndex:
    """Inverted-file index: k-means partitions + scan only the nearest few.

    Approximate: a true neighbor sitting in a partition that isn't probed is
    missed. `nprobe` trades recall for speed (nprobe == nlist is exact).
    Scores are squared L2, lower is better, same as FlatIndex.
    """

    def __init__(self, dim, nlist=100, nprobe=8):
        if nlist < 1 or nprobe < 1:
            raise ValueError("nlist and nprobe must be >= 1")
        self.dim = dim
        self.nlist = nlist
        self.nprobe = nprobe
        self._centroids = None            # (nlist, dim) once trained
        self._c_sq = None
        self._lists = None                # per partition: dict(vectors, sq_norms, ids)
        self._seen = set()
        self._count = 0

    def __len__(self):
        return self._count

    @property
    def is_trained(self):
        return self._centroids is not None

    def train(self, data):
        data = np.asarray(data, dtype=np.float32)
        if data.ndim != 2 or data.shape[1] != self.dim:
            raise ValueError(f"expected shape (n, {self.dim}), got {data.shape}")
        self._centroids = kmeans(data, self.nlist)
        self._c_sq = np.einsum("ij,ij->i", self._centroids, self._centroids)
        self._lists = [
            {
                "vectors": np.empty((0, self.dim), dtype=np.float32),
                "sq_norms": np.empty(0, dtype=np.float32),
                "ids": np.empty(0, dtype=object),
            }
            for _ in range(self.nlist)
        ]

    def add_batch(self, ids, vectors):
        if not self.is_trained:
            raise RuntimeError("train() the index before adding vectors")
        m = np.asarray(vectors, dtype=np.float32)
        if m.ndim != 2 or m.shape[1] != self.dim:
            raise ValueError(f"expected shape (n, {self.dim}), got {m.shape}")
        ids = list(ids)
        if len(ids) != len(m):
            raise ValueError("ids and vectors must have the same length")
        if len(set(ids)) != len(ids) or self._seen.intersection(ids):
            raise ValueError("duplicate id in batch")

        labels = assign(m, self._centroids)
        ids_arr = np.empty(len(ids), dtype=object)
        ids_arr[:] = ids
        sq = np.einsum("ij,ij->i", m, m)
        for lst in np.unique(labels):
            sel = labels == lst
            part = self._lists[lst]
            part["vectors"] = np.vstack([part["vectors"], m[sel]])
            part["sq_norms"] = np.concatenate([part["sq_norms"], sq[sel]])
            part["ids"] = np.concatenate([part["ids"], ids_arr[sel]])
        self._seen.update(ids)
        self._count += len(ids)

    def search(self, query, k=5, nprobe=None):
        if not self.is_trained:
            raise RuntimeError("train() the index before searching")
        q = np.asarray(query, dtype=np.float32)
        if q.shape != (self.dim,):
            raise ValueError(f"expected shape ({self.dim},), got {q.shape}")
        if self._count == 0 or k <= 0:
            return []
        nprobe = min(nprobe or self.nprobe, self.nlist)

        # 1) which partitions are closest to the query?
        c_scores = self._c_sq - 2.0 * (self._centroids @ q)
        if nprobe < self.nlist:
            probe = np.argpartition(c_scores, nprobe - 1)[:nprobe]
        else:
            probe = np.arange(self.nlist)

        # 2) scan only those partitions
        q_sq = float(np.dot(q, q))
        score_parts, id_parts = [], []
        for lst in probe:
            part = self._lists[lst]
            if len(part["ids"]) == 0:
                continue
            s = part["sq_norms"] - 2.0 * (part["vectors"] @ q) + q_sq
            score_parts.append(np.maximum(s, 0.0))
            id_parts.append(part["ids"])
        if not score_parts:
            return []
        scores = np.concatenate(score_parts)
        ids = np.concatenate(id_parts)

        # 3) best k among the candidates
        if k < len(scores):
            top = np.argpartition(scores, k - 1)[:k]
            top = top[np.argsort(scores[top])]
        else:
            top = np.argsort(scores)
        return [(ids[i], float(scores[i])) for i in top]

    def save(self, path):
        """Write the trained index to a single file (atomic replace)."""
        if not self.is_trained:
            raise RuntimeError("train() the index before saving")
        ids, chunks, sizes = [], [], []
        for part in self._lists:                     # vectors grouped by partition
            ids.extend(part["ids"].tolist())
            chunks.append(part["vectors"])
            sizes.append(len(part["ids"]))
        fileformat.save(
            path, self.dim, "l2", ids, np.vstack(chunks),
            ivf={"nprobe": self.nprobe, "centroids": self._centroids, "list_sizes": sizes},
        )

    @classmethod
    def load(cls, path):
        """Read an index written by `save`. No retraining needed."""
        c = fileformat.load(path)
        if c.kind != "ivf":
            raise fileformat.FormatError("this file holds a flat index; use FlatIndex.load()")
        index = cls(c.dim, nlist=c.nlist, nprobe=c.nprobe)
        index._centroids = c.centroids
        index._c_sq = np.einsum("ij,ij->i", c.centroids, c.centroids)
        index._lists = []
        start = 0
        for size in c.list_sizes:
            end = start + int(size)
            vectors = c.vectors[start:end].copy()
            ids = np.empty(end - start, dtype=object)
            ids[:] = c.ids[start:end]
            index._lists.append({
                "vectors": vectors,
                "sq_norms": np.einsum("ij,ij->i", vectors, vectors),
                "ids": ids,
            })
            start = end
        index._seen = set(c.ids)
        index._count = len(c.ids)
        return index