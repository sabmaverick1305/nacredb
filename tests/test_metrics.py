from nacredb import recall_at_k


def test_perfect_recall():
    assert recall_at_k([[1, 2, 3]], [[3, 2, 1]]) == 1.0


def test_partial_recall_is_averaged_over_queries():
    truth = [[1, 2], [3, 4]]
    got = [[1, 9], [3, 4]]          # 0.5 and 1.0
    assert recall_at_k(truth, got) == 0.75


def test_no_overlap_is_zero():
    assert recall_at_k([[1, 2]], [[8, 9]]) == 0.0