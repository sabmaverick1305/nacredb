import numpy as np


def assign(data, centroids):
    """Index of the nearest centroid (squared L2) for each row of `data`."""
    data = np.asarray(data, dtype=np.float32)
    c_sq = np.einsum("ij,ij->i", centroids, centroids)
    # |x - c|^2 = |x|^2 - 2 x.c + |c|^2. The |x|^2 term is the same for every
    # centroid, so it can be dropped when we only want the argmin.
    scores = c_sq[None, :] - 2.0 * (data @ centroids.T)
    return scores.argmin(axis=1)


def kmeans(data, k, iters=20, seed=0):
    """Lloyd's algorithm. Returns centroids of shape (k, dim), float32."""
    data = np.asarray(data, dtype=np.float32)
    n = len(data)
    if not 1 <= k <= n:
        raise ValueError(f"k must be between 1 and {n}, got {k}")

    rng = np.random.default_rng(seed)
    centroids = data[rng.choice(n, size=k, replace=False)].copy()
    labels = None

    for _ in range(iters):
        new_labels = assign(data, centroids)
        if labels is not None and np.array_equal(new_labels, labels):
            break                                   # converged
        labels = new_labels

        # Mean of each cluster via a one-hot matrix product (fast BLAS call).
        onehot = np.zeros((n, k), dtype=np.float32)
        onehot[np.arange(n), labels] = 1.0
        counts = onehot.sum(axis=0)
        sums = onehot.T @ data
        nonempty = counts > 0
        centroids[nonempty] = sums[nonempty] / counts[nonempty, None]

        # An empty cluster would stay stuck; re-seed it from a random point.
        empty = np.flatnonzero(~nonempty)
        if len(empty):
            centroids[empty] = data[rng.choice(n, size=len(empty), replace=False)]
    return centroids