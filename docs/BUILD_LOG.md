# Build log: how nacredb was built, step by step

nacredb was built in small steps, and every step was measured. This log
records what we did, what the numbers said, and what surprised us.

One honest note first: the project was committed to git once, at `v0.0.1`, so
the earlier stages are not separate commits or tags. This log is the
step-by-step record. Numbers come from the author's machine unless stated
otherwise; **recall is reproducible, speed varies by machine and by run**
(the same exact-search benchmark ranged from about 660 to 1,200 queries per
second across runs), so treat speedups as approximate.

## At a glance

| Step | What was added | Tests after | Headline result |
|---|---|---|---|
| 0 | Project setup | 1 | `pip install -e ".[dev]"` and a smoke test |
| 1 | `FlatIndex`: exact search | 7 | Correct, simple, slow to build |
| 2 | `add_batch` + `recall_at_k` + benchmark | 11 | Baseline: about 127 queries/sec |
| 3 | Faster exact search | 12 | About 127 to about 1,200 queries/sec |
| 4 | Save and load to one file | 20 | Atomic writes, versioned format |
| 5a | k-means | 25 | 50k x 128 vectors, k=100 in under a second |
| 5b | `IVFIndex` | 31 | Exact when `nprobe` equals `nlist` |
| 5c | The `nprobe` sweep | 31 | Recall 0.967 at about 5x faster |
| 6 | Save and load for IVF (format v2) | 38 | Old files still load |
| 7 | Tuning `nlist` | 38 | A plateau around 100 to 224 |
| 8 | Publishing to PyPI | 38 | Caught a misleading quick start |

The whole library stays under 500 lines.

## Step 0: the idea, and the scope

The project started as "can I build a vector database from scratch?" After
comparing FAISS, pgvector, Qdrant, and Milvus, it was clear the field is
crowded: each of them is mature and optimized in ways a from-scratch project
cannot match quickly.

So the goal changed from "highly optimized" to "small and understandable",
inspired by SQLite's story: small, embedded, one file, reliable. Everything
after that follows from one rule: **prefer the version a reader can follow.**

The name is *nacre*, the layered material a pearl is made of: a tiny core,
built one layer at a time. Earlier candidates collided with existing
projects, so a quick name search saved trouble later.

## Step 1: exact search (`FlatIndex`)

**What we did.** Store vectors in one NumPy matrix, compute the distance from
the query to every vector, and return the k smallest.

**Design choice.** Scores are always "lower is better" (squared L2, or
1 minus cosine similarity). One rule means every future index can share one
comparison.

**Known shortcuts, written down on purpose.** Each `add` copied the whole
matrix (`np.vstack`), and searching sorted all n scores. Both were fixed
later, and only after they were measured.

## Step 2: the benchmark harness

**What we did.** Added `add_batch` (so 50,000 vectors load without 50,000
copies), `recall_at_k`, and `bench/run_flat.py`.

**What we measured.** 50,000 vectors, 128 dimensions, k=10: about 127
queries/sec, with recall 1.000.

**What surprised us.** Recall was 1.000 *by construction*: the ground truth
came from the same code being measured. The harness only becomes informative
once something approximate is compared against it. Keep that in mind for
step 3.

## Step 3: making exact search faster

**Two changes, measured separately:**

1. **Precompute squared norms.** Since `|x - q|^2 = |x|^2 - 2 x.q + |q|^2`,
   store each vector's `|x|^2` once and turn every query into a single
   matrix-vector multiply. The old version built a temporary 50,000 x 128
   array (about 25 MB) on every query.
2. **`argpartition` for top-k.** Select the k best scores in linear time and
   sort only those k, instead of sorting all 50,000.

**What we measured.** About 127 to about 1,200 queries/sec, roughly 9x. In an
earlier measurement on a different machine the norms trick gave about 4x and
`argpartition` about another 1.7x.

**The lesson that mattered.** Because recall is measured against the same
code path, a bug in the new distance formula would still print `recall=1.000`.
So we added an **independent test** that compares results against the plain
formula on random data. A benchmark tells you about speed; it cannot prove
correctness. Floating-point note: the expanded formula can produce tiny
negative values from cancellation in float32, so scores are clamped at 0.

## Step 4: save and load

**What we did.** A single-file format:

- **Page 0:** a header with a magic number, `format_version`, page size,
  `dim`, metric, vector count, and the size of the ids section.
- **Ids:** a JSON array (ids must be `int` or `str`).
- **Vectors:** raw float32, starting on a page boundary so they could be
  memory-mapped later.

**Atomic saves.** Write to a temporary file in the same directory, `fsync`,
then rename over the target. A crash leaves either the old file or the new
one, never a half-written mix. A test forces a failed save and checks that
the old file is intact and no temp file is left behind.

**Strict loading.** Wrong magic, a future version, or a size mismatch raise a
clear `FormatError` before any large read.

**Deliberate limits.** Each save rewrites the whole file; ids are JSON. Both
are fine for a first format and noted in the README.

## Step 5a: k-means

**What we did.** Lloyd's algorithm in about 40 lines of NumPy. It reuses the
norms trick to find the nearest centroid, computes cluster means with a
one-hot matrix product, and re-seeds any cluster that ends up empty (an empty
cluster would otherwise stay stuck forever).

**What we measured.** Training on 50,000 x 128 vectors with 100 clusters took
under a second in our environment.

## Step 5b: the IVF index

**What we did.** `IVFIndex` trains k-means on some data, files every vector
into its nearest centroid's bucket, and at query time scans only the
`nprobe` nearest buckets.

**The key test.** With `nprobe` equal to `nlist`, every bucket is scanned, so
the result must match `FlatIndex` exactly. It is the check that the bucket
bookkeeping is right, and it does not depend on the benchmark.

## Step 5c: the `nprobe` dial

**What we measured (clustered data, 50,000 x 128, k=10):**

| nprobe | recall@10 | speedup vs. exact |
|---|---|---|
| 1 | 0.882 | ~23x |
| 4 | 0.952 | ~9x |
| 8 | 0.967 | ~5x |
| 16 | 0.982 | ~3x |
| 100 (all) | 1.000 | 0.6x (slower than exact) |

**What surprised us.**

- **Scanning everything is slower than exact search.** At `nprobe=100` IVF
  pays partition bookkeeping for nothing. Approximate indexes only pay off at
  the low end of the dial.
- **The speedup was smaller than the ideal.** Probing 8 of 100 buckets
  touches about 8% of the data, yet the speedup was about 5x, not about 12x.
  Per-partition Python overhead eats the difference, which is a clear argument
  for moving that loop to a compiled language later.
- **The dataset decides everything.** On uniform random noise, recall at
  `nprobe=8` was only about 0.30. Random noise has no clusters to exploit.
  Our first "clustered" dataset was also too easy (recall hit 1.000 by
  `nprobe=4`), so we made the clusters overlap more to get a realistic curve.

## Step 6: saving an IVF index (format version 2)

**What we did.** Extended the format: the header gained the index kind,
`nlist`, and `nprobe`; a new section holds the centroids and partition sizes;
and vectors are stored **grouped by partition**. Loading is then a set of
slices: no retraining, no re-assigning, identical results.

**Backward compatibility.** Version 1 files (flat indexes written at step 4)
must keep loading. A test builds a version 1 file byte by byte, exactly as
step 4 wrote it, and checks the new code still reads it. Loading an IVF file
with `FlatIndex.load` (and the reverse) gives an error that names the right
loader.

## Step 7: how many partitions?

**What we measured.** For each `nlist`, the smallest `nprobe` reaching
recall 0.95:

| nlist | nprobe needed | % of data scanned |
|---|---|---|
| 25 | 16 | 64% |
| 50 | 8 | 16% |
| 100 | 4 | 4% |
| 224 | 8 | 4% |
| 400 | 16 | 4% |

**Reading it.** Too few partitions gives no speedup. Beyond a point, more
partitions keep the scanned fraction the same but add per-query overhead. The
best region is a plateau (100 to 224 here), and the common rule of thumb,
`nlist` near the square root of the vector count, lands inside it.

**A caveat we kept in the README.** The synthetic data has exactly 100 true
clusters, so `nlist=100` has an unfair advantage. Real embeddings do not come
with a known cluster count.

## Step 8: publishing (the mistakes were the lesson)

We published to TestPyPI first, then to PyPI. Things that were caught, and
what they teach:

- **The quick start was misleading.** It used uniform random data, and on
  that data IVF at `nprobe=8` returned different neighbors than exact search
  (recall about 0.64 for the top 3). Nothing was broken: uniform noise is the
  worst case for IVF, the same limit the benchmarks showed. But a first
  example should not look like a bug. An earlier check had matched only by
  luck. We switched the demo to clustered data and verified that flat, IVF,
  and the reloaded index agree (recall@3 = 1.000 over 200 queries).
- **A README is frozen into each release.** PyPI never lets you replace a
  released version, so this was fixed before the real upload.
- **A one-character typo in the project URLs** (`sabmaverick-1305` instead of
  `sabmaverick1305`) would have shipped broken links. It was found by reading
  the actual file, not by trusting a summary of it.
- **Commit emails are public.** The first commit exposed a personal address;
  because the repository had one commit and no clones, it was rewritten to
  GitHub's noreply address. Author, committer, and tag all record an email, so
  all three had to be checked (`git log`, `git cat-file`).
- **Tokens:** account-wide API tokens were deleted after the first uploads and
  replaced with project-scoped ones.
- **Wrong environment:** a build failed because the shell was still inside a
  test virtual environment. Keep build and install-test environments separate.

## What we would do differently

- **Commit and tag every step as it happens.** The earlier stages cannot be
  checked out, which is why this log exists.
- **Keep the old versions of hot code** (for example the first `flat.py`) in
  a `history/` folder, so the first benchmark row can be reproduced.
- **Test on data that resembles real embeddings** from the start. Everything
  here is synthetic.

## Where it goes next

- int8 scalar quantization (about 4x less memory)
- A Rust port of the hot loops, called from Python, benchmarked against this
  version
- HNSW as a third index

Each will follow the same loop that worked here: build the benchmark,
change one thing, measure, and add an independent correctness test.
