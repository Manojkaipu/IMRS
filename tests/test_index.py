import numpy as np
import pytest

from imrs.index import INDEXERS, VectorStore, auto_nlist, auto_pq_m, build_index, exact_search
from imrs.metrics import neighbor_recall

from .conftest import unit_vectors


def test_exact_search_matches_bruteforce():
    X, Q = unit_vectors(500, 16), unit_vectors(20, 16, seed=1)
    sims, idx = exact_search(X, Q, 5, chunk=7)
    ref = np.argsort(-(Q @ X.T), axis=1)[:, :5]
    np.testing.assert_array_equal(idx, ref)
    assert np.all(np.diff(sims, axis=1) <= 1e-6)


@pytest.mark.parametrize("name", INDEXERS)
def test_every_index_returns_cosine_scores_and_good_recall(name):
    X = unit_vectors(2000, 32)
    Q = X[:50]
    index = build_index(name, X)
    sims, pos = index.search(Q, 10)
    assert sims.shape == pos.shape == (50, 10)
    _, exact = exact_search(X, Q, 10)
    floor = {"flat": 1.0, "hnsw": 0.9, "ivf_flat": 0.5, "ivf_pq": 0.2, "annoy": 0.5}[name]
    assert neighbor_recall(pos, exact, 10) >= floor
    if name != "ivf_pq":  # PQ scores are approximations by design
        np.testing.assert_allclose(sims[:, 0], 1.0, atol=1e-3)  # each query finds itself


def test_search_params_trade_speed_for_recall():
    X = unit_vectors(3000, 32)
    _, exact = exact_search(X, X[:100], 10)
    index = build_index("ivf_flat", X)
    index.set_search_params(nprobe=1)
    low = neighbor_recall(index.search(X[:100], 10)[1], exact, 10)
    index.set_search_params(nprobe=64)
    high = neighbor_recall(index.search(X[:100], 10)[1], exact, 10)
    assert high > low and high > 0.95


def test_ivf_sizing_rules():
    assert auto_nlist(30_000) <= 30_000 // 39
    assert auto_nlist(100) >= 1
    assert 512 % auto_pq_m(512) == 0 and 768 % auto_pq_m(768) == 0


def test_vector_store_add_dedupe_remove_roundtrip(tmp_path):
    X = unit_vectors(100, 8)
    paths = [f"c/{i}.jpg" for i in range(100)]
    store = VectorStore(8, "hnsw")
    assert store.add(X, paths, ["c"] * 100) == 100
    assert store.add(X[:10], paths[:10], ["c"] * 10) == 0  # already present
    hits = store.search(X[42], 3)[0]
    assert hits[0][0] == "c/42.jpg" and hits[0][2] == pytest.approx(1.0, abs=1e-4)

    # incremental add goes into the live index
    Y = unit_vectors(5, 8, seed=9)
    store.add(Y, [f"new/{i}.jpg" for i in range(5)], ["new"] * 5)
    assert store.search(Y[3], 1)[0][0][0] == "new/3.jpg"

    assert store.remove(["c/42.jpg", "missing.jpg"]) == 1
    assert "c/42.jpg" not in store
    assert all(p != "c/42.jpg" for p, _, _ in store.search(X[42], 5)[0])

    store.save(tmp_path / "s")
    loaded = VectorStore.load(tmp_path / "s")
    assert len(loaded) == len(store) == 104
    assert loaded.search(Y[0], 1)[0][0][0] == "new/0.jpg"


def test_vector_store_switch_index():
    X = unit_vectors(300, 8)
    store = VectorStore(8)
    store.add(X, [str(i) for i in range(300)], ["x"] * 300)
    for name in INDEXERS:
        store.set_indexer(name)
        assert store.search(X[7], 1)[0][0][0] == "7"
