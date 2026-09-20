"""IVF vs. exact search: recall@k and QPS as nprobe grows.

Two datasets, same code path:
  clustered - Gaussian mixture; a rough stand-in for real embeddings
  uniform   - random noise; no structure for IVF to exploit (worst case)
"""
import time

import numpy as np

from nacredb import FlatIndex, IVFIndex, recall_at_k

DIM, N, NQ, K = 128, 50_000, 200, 10
NLIST = 100
NPROBES = [1, 2, 4, 8, 16, 32, 100]


def make_clustered(rng, n, nq, dim, n_clusters=100, spread=0.5):
    centers = rng.random((n_clusters, dim), dtype=np.float32)
    def sample(m):
        which = rng.integers(0, n_clusters, m)
        return (centers[which] + rng.normal(0, spread, (m, dim))).astype(np.float32)
    return sample(n), sample(nq)


def make_uniform(rng, n, nq, dim):
    return rng.random((n, dim), dtype=np.float32), rng.random((nq, dim), dtype=np.float32)


def qps(fn, queries):
    t = time.perf_counter()
    out = [fn(q) for q in queries]
    return out, len(queries) / (time.perf_counter() - t)


def run(name, data, queries):
    ids = list(range(len(data)))

    flat = FlatIndex(dim=DIM)
    flat.add_batch(ids, data)
    truth_hits, flat_qps = qps(lambda q: flat.search(q, K), queries)
    truth = [[i for i, _ in hits] for hits in truth_hits]

    ivf = IVFIndex(dim=DIM, nlist=NLIST)
    t = time.perf_counter()
    ivf.train(data[:20_000])          # train on a sample; add everything
    ivf.add_batch(ids, data)
    build = time.perf_counter() - t

    print(f"\n== {name}: n={len(data)} dim={DIM} k={K} nlist={NLIST}")
    print(f"flat (exact):  recall@{K}=1.000  QPS={flat_qps:.0f}      (IVF build {build:.1f}s)")
    print(f"{'nprobe':>7} {'recall@10':>10} {'QPS':>7} {'speedup':>8}")
    for nprobe in NPROBES:
        hits, rate = qps(lambda q: ivf.search(q, K, nprobe=nprobe), queries)
        got = [[i for i, _ in h] for h in hits]
        print(f"{nprobe:>7} {recall_at_k(truth, got):>10.3f} {rate:>7.0f} {rate / flat_qps:>7.1f}x")


rng = np.random.default_rng(42)
run("clustered", *make_clustered(rng, N, NQ, DIM))
run("uniform", *make_uniform(rng, N, NQ, DIM))