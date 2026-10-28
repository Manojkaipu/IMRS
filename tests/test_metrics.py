import numpy as np
import pytest

from imrs.metrics import (
    average_precision_at_k,
    evaluate_rankings,
    ndcg_at_k,
    neighbor_recall,
    precision_at_k,
    recall_at_k,
)


def test_perfect_ranking_scores_one():
    rel = np.ones(10)
    assert average_precision_at_k(rel, 10, n_relevant=50) == 1.0
    assert ndcg_at_k(rel, 10, n_relevant=50) == pytest.approx(1.0)
    assert precision_at_k(rel, 10) == 1.0


def test_recall_uses_gallery_count_not_retrieved_hits():
    # The original code divided by the number of hits in the retrieved list, so this was 1.0.
    rel = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    assert recall_at_k(rel, 10, n_relevant=20) == pytest.approx(0.1)


def test_average_precision_known_value():
    rel = np.array([1, 0, 1, 0])  # precisions at hits: 1/1, 2/3
    assert average_precision_at_k(rel, 4, n_relevant=2) == pytest.approx((1 + 2 / 3) / 2)
    # normalised by min(k, n_relevant): missing relevant items are penalised
    assert average_precision_at_k(rel, 4, n_relevant=4) == pytest.approx((1 + 2 / 3) / 4)


def test_ndcg_ideal_comes_from_gallery_not_retrieved_list():
    # Only one hit, at rank 3. If the ideal were computed by re-sorting the retrieved list
    # (as in the original code), a single hit anywhere would give a misleadingly high score.
    rel = np.array([0, 0, 1, 0, 0])
    expected = (1 / np.log2(4)) / sum(1 / np.log2(i + 2) for i in range(5))
    assert ndcg_at_k(rel, 5, n_relevant=10) == pytest.approx(expected)


def test_no_relevant_items_is_zero_not_nan():
    rel = np.zeros(5)
    assert average_precision_at_k(rel, 5, 0) == 0.0
    assert ndcg_at_k(rel, 5, 0) == 0.0
    assert recall_at_k(rel, 5, 0) == 0.0


def test_evaluate_rankings_handles_padding():
    ranked = np.array([[0, 0, -1], [1, 0, 1]])
    out = evaluate_rankings(ranked, np.array([0, 1]), np.array([2, 2]), k=3)
    assert out["Hit@1"] == 1.0
    assert out["P@3"] == pytest.approx((2 / 3 + 2 / 3) / 2)
    assert out["R@3"] == pytest.approx(1.0)


def test_neighbor_recall():
    approx = np.array([[1, 2, 3], [4, 5, 6]])
    exact = np.array([[1, 2, 9], [4, 5, 6]])
    assert neighbor_recall(approx, exact, 3) == pytest.approx((2 + 3) / 6)


def test_neighbor_recall_counts_ties_at_kth_as_hits():
    # 3 and 9 are equally similar to the query; exact search kept 9, the index kept 3
    approx, exact = np.array([[1, 2, 3]]), np.array([[1, 2, 9]])
    assert neighbor_recall(approx, exact, 3, np.array([[0.9, 0.8, 0.7]]),
                           np.array([[0.9, 0.8, 0.7]])) == pytest.approx(1.0)
    assert neighbor_recall(approx, exact, 3, np.array([[0.9, 0.8, 0.6]]),
                           np.array([[0.9, 0.8, 0.7]])) == pytest.approx(2 / 3)
