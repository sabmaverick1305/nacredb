import numpy as np

from . import fileformat

class FlatIndex:
    """Exact brute-force index. Scores are always 'lower is better'."""

    METRICS = ("l2", "cosine")

    def __init__(self, dim, metric="l2"):
        if metric not in self.METRICS:
            raise ValueError(f"metric must be one of {self.METRICS}")
        self.dim = dim
        self.metric = metric
        self._vectors = np.empty((0, dim), dtype=np.float32)
        self._sq_norms = np.empty(0, dtype=np.float32)  # |x|^2, kept in sync with _vectors
        self._ids = []
        self._seen = set()

    def __len__(self):
        return len(self._ids)

    def add(self, id, vector):
        v = np.asarray(vector, dtype=np.float32)
        if v.shape != (self.dim,):
            raise ValueError(f"expected shape ({self.dim},), got {v.shape}")
        if id in self._seen:
            raise ValueError(f"duplicate id: {id!r}")
        self._vectors = np.vstack([self._vectors, v])  # O(n) copy; fixed in a later step
        self._sq_norms = np.append(self._sq_norms, np.dot(v, v))
        self._ids.append(id)
        self._seen.add(id)

    def add_batch(self, ids, vectors):
        m = np.asarray(vectors, dtype=np.float32)
        if m.ndim != 2 or m.shape[1] != self.dim:
            raise ValueError(f"expected shape (n, {self.dim}), got {m.shape}")
        ids = list(ids)
        if len(ids) != len(m):
            raise ValueError("ids and vectors must have the same length")
        if len(set(ids)) != len(ids) or self._seen.intersection(ids):
            raise ValueError("duplicate id in batch")
        self._vectors = np.vstack([self._vectors, m])
        self._sq_norms = np.concatenate([self._sq_norms, np.einsum("ij,ij->i", m, m)])
        self._ids.extend(ids)
        self._seen.update(ids)

    def search(self, query, k=5):
        q = np.asarray(query, dtype=np.float32)
        if q.shape != (self.dim,):
            raise ValueError(f"expected shape ({self.dim},), got {q.shape}")
        if len(self) == 0 or k <= 0:
            return []
        dots = self._vectors @ q
        if self.metric == "l2":
            # |x - q|^2 = |x|^2 - 2 x.q + |q|^2  (no temporary n x dim array)
            scores = np.maximum(self._sq_norms - 2.0 * dots + np.dot(q, q), 0.0)
        else:  # cosine
            norms = np.sqrt(self._sq_norms) * np.linalg.norm(q)
            scores = 1.0 - dots / np.maximum(norms, 1e-12)
        n = len(self)
        if k < n:
            part = np.argpartition(scores, k - 1)[:k]   # O(n) selection of the k best
            top = part[np.argsort(scores[part])]         # sort only those k
        else:
            top = np.argsort(scores)
        return [(self._ids[i], float(scores[i])) for i in top]

    def save(self, path):
        """Write the index to a single file (atomic replace)."""
        fileformat.save(path, self.dim, self.metric, self._ids, self._vectors)

    @classmethod
    def load(cls, path):
        """Read an index written by `save`."""
        c = fileformat.load(path)
        if c.kind != "flat":
            raise fileformat.FormatError("this file holds an IVF index; use IVFIndex.load()")
        index = cls(c.dim, c.metric)
        if len(c.ids):
            index.add_batch(c.ids, c.vectors)
        return index