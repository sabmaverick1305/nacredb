"""Baseline benchmark: exact search on random data.

Recall is 1.0 by construction (flat vs. itself). This becomes meaningful
once an approximate index exists and is passed in as the candidate.
"""

import time
import numpy as np

from nacredb import FlatIndex, recall_at_k

DIM, N, NQ, K = 128, 50_000, 200, 10

rng = np.random.default_rng(42)
data = rng.random((N, DIM), dtype=np.float32)
queries = rng.random((NQ, DIM), dtype=np.float32)

index = FlatIndex(dim=DIM, metric="l2")
index.add_batch(range(N), data)

truth = [[i for i, _ in index.search(q, K)] for q in queries]

candidate = index
t = time.perf_counter()

got = [[i for i, _ in candidate.search(q, K)] for q in queries]
secs = time.perf_counter() - t

print(f"n={N} dim={DIM} k={K}  recall@{K}={recall_at_k(truth, got):.3f}  QPS={NQ / secs:.0f}")