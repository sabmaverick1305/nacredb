"""How does the partition count (nlist) change the speed/recall tradeoff?

For each nlist, sweep nprobe and report the fastest setting that still
reaches recall@10 >= TARGET. Same clustered dataset as run_ivf.py.
"""
import time

import numpy as np

from nacredb import FlatIndex, IVFIndex, recall_at_k

DIM, N, NQ, K = 128, 50_000, 200, 10
NLISTS = [25, 50, 100, 224, 400]
TARGET = 0.95


def make_clustered(rng, n, nq, dim, n_clusters=100, spread=0.5):
    centers = rng.random((n_clusters, dim), dtype=np.float32)
    def sample(m):
        which = rng.integers(0, n_clusters, m)
        return (centers[which] + rng.normal(0, spread, (m, dim))).astype(np.float32)
    return sample(n), sample(nq)


def qps(fn, queries):
    t = time.perf_counter()
    out = [fn(q) for q in queries]
    return out, len(queries) / (time.perf_counter() - t)


rng = np.random.default_rng(42)
data, queries = make_clustered(rng, N, NQ, DIM)
ids = list(range(N))

flat = FlatIndex(dim=DIM)
flat.add_batch(ids, data)
hits, flat_qps = qps(lambda q: flat.search(q, K), queries)
truth = [[i for i, _ in h] for h in hits]
print(f"flat (exact): QPS={flat_qps:.0f}   target recall@{K} >= {TARGET}\n")
print(f"{'nlist':>6} {'build s':>8} {'nprobe':>7} {'recall':>7} {'QPS':>7} {'speedup':>8} {'% scanned':>10}")

for nlist in NLISTS:
    ivf = IVFIndex(dim=DIM, nlist=nlist)
    t = time.perf_counter()
    ivf.train(data[:20_000])
    ivf.add_batch(ids, data)
    build = time.perf_counter() - t

    best = None
    for nprobe in sorted({1, 2, 4, 8, 16, 32, 64, 128, nlist}):
        if nprobe > nlist:
            continue
        got_hits, rate = qps(lambda q: ivf.search(q, K, nprobe=nprobe), queries)
        got = [[i for i, _ in h] for h in got_hits]
        r = recall_at_k(truth, got)
        if r >= TARGET:
            best = (nprobe, r, rate)
            break
    if best is None:
        print(f"{nlist:>6} {build:>8.1f}   never reached the target")
    else:
        nprobe, r, rate = best
        print(f"{nlist:>6} {build:>8.1f} {nprobe:>7} {r:>7.3f} {rate:>7.0f} {rate / flat_qps:>7.1f}x {100 * nprobe / nlist:>9.0f}%")