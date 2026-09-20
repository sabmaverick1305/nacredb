# nacredb

**A small vector database you can read in one sitting. Built layer by layer.**

Pronounced "NAY-ker". *Nacre* is the layered material a pearl is made of: a
tiny core, built up one layer at a time.

> **Status: pre-alpha and educational.** nacredb exists to show how a vector
> database works, in under 500 lines of readable Python. It is not a
> production database and does not try to compete with FAISS, Qdrant, or
> pgvector. The API will change.

## What you get

- **`FlatIndex`**: exact nearest-neighbor search (L2 and cosine)
- **`IVFIndex`**: approximate search with k-means partitions and an `nprobe` speed/recall dial
- **A single-file, versioned format**: atomic saves, strict validation, old files keep loading
- **A benchmark harness**: recall@k and QPS measured against exact search at every step
- **38 tests**, including independent correctness checks that a benchmark alone cannot give you

## Install

```bash
pip install nacredb
```

## From source (tests and benchmarks)

```bash
git clone https://github.com/sabmaverick1305/nacredb.git && cd nacredb
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Quick start

```bash
git clone https://github.com/sabmaverick1305/nacredb.git && cd nacredb
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

```python
import numpy as np
from nacredb import FlatIndex, IVFIndex

# Demo data with cluster structure, like real embeddings. (Uniform random
# noise has no structure, so approximate search misses more neighbors.)
rng = np.random.default_rng(0)
centers = rng.random((100, 64))
data = (centers[rng.integers(0, 100, 10_000)] + rng.normal(0, 0.05, (10_000, 64))).astype("float32")

# Exact search
flat = FlatIndex(dim=64, metric="l2")          # or "cosine"
flat.add_batch(range(len(data)), data)
print(flat.search(data[0], k=3))               # [(id, score), ...], lower score = closer

# Approximate search: train once, add, search
ivf = IVFIndex(dim=64, nlist=100, nprobe=8)
ivf.train(data)
ivf.add_batch(range(len(data)), data)
print(ivf.search(data[0], k=3))                # same neighbors here; approximate in general

# Persist to a single file and load it back
ivf.save("vectors.ndb")
ivf = IVFIndex.load("vectors.ndb")
```

## How it is built

```
        FlatIndex          IVFIndex          <- search, add_batch, save, load
              \               /
               \         kmeans.py           <- trains the partitions
                \           /
              fileformat.py                  <- versioned single-file format
```

| File | What it teaches |
|---|---|
| `flat.py` | Exact search, the norms trick, top-k with `argpartition` |
| `kmeans.py` | Lloyd's algorithm in NumPy, empty-cluster handling |
| `ivf.py` | Partitioning vectors and probing only the nearest few |
| `fileformat.py` | Header, versioning, page alignment, atomic writes |
| `metrics.py` | `recall_at_k`, the number that keeps approximate search honest |

## Measured results

Tables 2 and 3 come from the scripts in `bench/` (50,000 vectors, 128
dimensions, k=10, 200 queries). **Recall is reproducible; speed depends on
your machine**, so treat speedups as approximate and run the scripts yourself.

### 1. Making exact search faster (`bench/run_flat.py` measures the current version)

| Version | Queries/sec |
|---|---|
| First version: straightforward NumPy | ~127 |
| Current: precomputed norms + `argpartition` for top-k | ~1,200 |

Measured on the author's machine. The first row is an earlier version of
`flat.py` that is no longer in the repo.

### 2. The approximate-search tradeoff (`bench/run_ivf.py`, clustered data)

| nprobe | recall@10 | speedup vs. exact |
|---|---|---|
| 1 | 0.882 | ~23x |
| 4 | 0.952 | ~9x |
| 8 | 0.967 | ~5x |
| 16 | 0.982 | ~3x |
| 100 (all partitions) | 1.000 | 0.6x (slower than exact) |

`nprobe` is the dial: probing fewer partitions is faster but can miss a true
neighbor. Probing everything is exact, and then IVF is pure overhead.

On uniform random noise, IVF recall at `nprobe=8` is only ~0.30: **IVF depends
on structure in the data.** Always test on data that resembles your own.

### 3. Choosing the partition count (`bench/run_nlist.py`)

| nlist | nprobe needed for recall >= 0.95 | % of data scanned |
|---|---|---|
| 25 | 16 | 64% |
| 50 | 8 | 16% |
| 100 | 4 | 4% |
| 224 | 8 | 4% |
| 400 | 16 | 4% |

Too few partitions gives no speedup; too many adds per-query overhead. A
common rule of thumb is `nlist` near the square root of the vector count. The
synthetic data here has exactly 100 true clusters, so `nlist=100` has an
unfair advantage; real embeddings will not.

## Lessons this project tries to show

1. **Build the benchmark before you optimize.** Every change above was
   measured against exact search.
2. **A benchmark cannot prove correctness.** Recall is computed against the
   same code, so a bug can hide behind a perfect score. Independent tests
   (`nprobe=nlist` must match exact search) catch what benchmarks cannot.
3. **Version the file format on day one**, and test that old files still load.
4. **Write atomically**: temp file, fsync, rename, so a crash never leaves a
   half-written database.
5. **Approximate search is a dial, not a switch.** Report recall and speed
   together, always.

## Limitations (on purpose)

- Pure Python and NumPy, single-threaded
- L2 only for `IVFIndex`; ids must be `int` or `str`
- No delete or update, no metadata filtering, no write-ahead log
- Each `save` rewrites the whole file
- Benchmarks use synthetic data, not real embeddings

## Roadmap

- int8 scalar quantization (4x less memory)
- A Rust port of the hot loops, called from Python
- HNSW as a third index

## License

Apache-2.0
