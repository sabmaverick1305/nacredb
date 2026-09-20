def recall_at_k(truth, got):
    """Mean fraction of the true top-k ids that appear in the returned ids.

    truth, got: lists with one list of ids per query.
    """
    if len(truth) != len(got):
        raise ValueError("truth and got must have the same number of queries")
    if not truth:
        return 0.0
    total = 0.0
    for t, g in zip(truth, got):
        total += len(set(t) & set(g)) / max(len(t), 1)
    return total / len(truth)