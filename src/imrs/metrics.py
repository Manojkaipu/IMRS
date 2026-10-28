"""Retrieval metrics for single-label (class-match) relevance.

All functions take `rel`, a 0/1 array over the ranked results of one query (rank 1 first),
and `n_relevant`, the number of relevant items in the *whole gallery* (query excluded).
Using the gallery count — not the number of hits that happened to be retrieved — is what
makes recall and nDCG meaningful.
"""
from __future__ import annotations

import numpy as np


def precision_at_k(rel: np.ndarray, k: int) -> float:
    return float(np.sum(rel[:k])) / k


def recall_at_k(rel: np.ndarray, k: int, n_relevant: int) -> float:
    if n_relevant == 0:
        return 0.0
    return float(np.sum(rel[:k])) / n_relevant


def hit_at_k(rel: np.ndarray, k: int) -> float:
    """1 if any relevant item is in the top-k (a.k.a. Recall@k in metric-learning papers)."""
    return float(np.any(rel[:k]))


def average_precision_at_k(rel: np.ndarray, k: int, n_relevant: int) -> float:
    """AP@k normalised by min(k, n_relevant), so a perfect ranking scores 1.0."""
    denom = min(k, n_relevant)
    if denom == 0:
        return 0.0
    r = np.asarray(rel[:k], dtype=float)
    hits = np.cumsum(r)
    ranks = np.arange(1, len(r) + 1)
    return float(np.sum(r * hits / ranks)) / denom


def ndcg_at_k(rel: np.ndarray, k: int, n_relevant: int) -> float:
    r = np.asarray(rel[:k], dtype=float)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(r * discounts[: len(r)]))
    idcg = float(np.sum(discounts[: min(k, n_relevant)]))
    return 0.0 if idcg == 0 else dcg / idcg


def evaluate_rankings(ranked_labels: np.ndarray, query_labels: np.ndarray,
                      n_relevant: np.ndarray, k: int) -> dict[str, float]:
    """Mean metrics over queries.

    ranked_labels: (Q, >=k) gallery labels in rank order (pad with -1 if a search returned fewer)
    query_labels:  (Q,)
    n_relevant:    (Q,) relevant gallery items per query
    """
    rel = ranked_labels[:, :k] == query_labels[:, None]
    out = {
        f"mAP@{k}": np.mean([average_precision_at_k(r, k, n) for r, n in zip(rel, n_relevant)]),
        f"P@{k}": np.mean([precision_at_k(r, k) for r in rel]),
        f"R@{k}": np.mean([recall_at_k(r, k, n) for r, n in zip(rel, n_relevant)]),
        f"nDCG@{k}": np.mean([ndcg_at_k(r, k, n) for r, n in zip(rel, n_relevant)]),
        "Hit@1": np.mean([hit_at_k(r, 1) for r in rel]),
    }
    return {name: float(v) for name, v in out.items()}


def neighbor_recall(approx: np.ndarray, exact: np.ndarray, k: int,
                    approx_sims: np.ndarray | None = None, exact_sims: np.ndarray | None = None,
                    eps: float = 1e-5) -> float:
    """Fraction of the exact top-k neighbours an ANN index recovers (the standard ANN recall).

    With similarities given, a returned neighbour counts if it is at least as close as the
    exact k-th neighbour (as in ann-benchmarks), so ties at the boundary that two exact
    searches break differently are not scored as misses.
    """
    if approx_sims is None or exact_sims is None:
        hits = [len(np.intersect1d(a[:k], e[:k])) for a, e in zip(approx, exact)]
    else:
        hits = (approx_sims[:, :k] >= exact_sims[:, k - 1: k] - eps).sum(axis=1)
    return float(np.mean(hits)) / k
