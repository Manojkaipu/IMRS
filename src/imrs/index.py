"""Nearest-neighbour indexes behind one small interface, plus an ID-mapped vector store.

All indexes take L2-normalised vectors and return (similarities, positions) where
similarity is cosine / inner product, so results are comparable across backends.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

INDEXERS = ("flat", "hnsw", "ivf_flat", "ivf_pq", "annoy")


def _f32(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.float32)


def auto_nlist(n: int) -> int:
    """~4*sqrt(N) inverted lists, the usual FAISS rule of thumb, capped so each list
    gets >= 39 training points (FAISS's own minimum for k-means)."""
    return int(max(1, min(4 * math.sqrt(n), n // 39)))


def auto_pq_m(d: int, target_bytes: int = 64) -> int:
    """Number of PQ sub-quantisers: must divide d; aim for ~64 bytes per vector."""
    for m in range(min(target_bytes, d), 0, -1):
        if d % m == 0:
            return m
    return 1


class AnnIndex:
    """Common interface. `search` returns (sims, positions); missing results have position -1."""

    name = "base"

    def add(self, X: np.ndarray) -> None:
        raise NotImplementedError

    def search(self, Q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def set_search_params(self, **params) -> None:
        """Tune the speed/accuracy knob without rebuilding (efSearch, nprobe, search_k)."""

    @property
    def ntotal(self) -> int:
        raise NotImplementedError


class FaissIndex(AnnIndex):
    def __init__(self, name: str, d: int, n_train_hint: int, *, hnsw_m: int = 32, ef_construction: int = 200,
                 ef_search: int = 64, nlist: int | None = None, nprobe: int = 8, pq_m: int | None = None):
        import faiss

        self.faiss, self.name, self.d = faiss, name, d
        metric = faiss.METRIC_INNER_PRODUCT
        if name == "flat":
            self.index = faiss.IndexFlatIP(d)
        elif name == "hnsw":
            self.index = faiss.IndexHNSWFlat(d, hnsw_m, metric)
            self.index.hnsw.efConstruction = ef_construction
            self.index.hnsw.efSearch = ef_search
        elif name in ("ivf_flat", "ivf_pq"):
            nlist = nlist or auto_nlist(n_train_hint)
            self._quantizer = faiss.IndexFlatIP(d)
            if name == "ivf_flat":
                self.index = faiss.IndexIVFFlat(self._quantizer, d, nlist, metric)
            else:
                # each PQ codebook has 2**nbits centroids and needs at least that many training points
                nbits = int(min(8, max(1, math.floor(math.log2(max(2, n_train_hint))))))
                self.index = faiss.IndexIVFPQ(self._quantizer, d, nlist, pq_m or auto_pq_m(d), nbits, metric)
            self.index.nprobe = min(nprobe, nlist)
        else:
            raise ValueError(name)

    def add(self, X):
        X = _f32(X)
        if not self.index.is_trained:
            self.index.train(X)
        self.index.add(X)

    def search(self, Q, k):
        return self.index.search(_f32(Q), int(k))

    def set_search_params(self, ef_search=None, nprobe=None, **_):
        if ef_search is not None and self.name == "hnsw":
            self.index.hnsw.efSearch = int(ef_search)
        if nprobe is not None and self.name.startswith("ivf"):
            self.index.nprobe = int(nprobe)

    @property
    def ntotal(self):
        return self.index.ntotal


class AnnoyIndex(AnnIndex):
    """Annoy can't add after build, so vectors are buffered and the forest is (re)built lazily."""

    name = "annoy"

    def __init__(self, d: int, n_trees: int = 50, search_k: int = -1):
        self.d, self.n_trees, self.search_k = d, n_trees, search_k
        self._vecs: list[np.ndarray] = []
        self._ann = None

    def add(self, X):
        self._vecs.append(_f32(X))
        self._ann = None

    def _build(self):
        from annoy import AnnoyIndex as _Annoy

        ann = _Annoy(self.d, "angular")
        for i, v in enumerate(np.concatenate(self._vecs) if self._vecs else []):
            ann.add_item(i, v)
        ann.build(self.n_trees)
        self._ann = ann

    def search(self, Q, k):
        if self._ann is None:
            self._build()
        k = int(k)
        sims = np.full((len(Q), k), -np.inf, dtype=np.float32)
        pos = np.full((len(Q), k), -1, dtype=np.int64)
        for qi, q in enumerate(_f32(Q)):
            ii, dd = self._ann.get_nns_by_vector(q, k, search_k=self.search_k, include_distances=True)
            # Annoy "angular" distance is sqrt(2 - 2cos) for unit vectors.
            sims[qi, : len(ii)] = 1.0 - np.square(dd) / 2.0
            pos[qi, : len(ii)] = ii
        return sims, pos

    def set_search_params(self, search_k=None, **_):
        if search_k is not None:
            self.search_k = int(search_k)

    @property
    def ntotal(self):
        return sum(len(v) for v in self._vecs)


def build_index(name: str, X: np.ndarray, **params) -> AnnIndex:
    """Create an index of type `name` and add `X` to it."""
    name = {"flat_ip": "flat"}.get(name, name)
    if name == "annoy":
        index: AnnIndex = AnnoyIndex(X.shape[1], **{k: v for k, v in params.items() if k in ("n_trees", "search_k")})
    elif name in INDEXERS:
        index = FaissIndex(name, X.shape[1], len(X), **params)
    else:
        raise ValueError(f"Unknown indexer '{name}'. Choose from: {', '.join(INDEXERS)}")
    index.add(X)
    if isinstance(index, AnnoyIndex):
        index._build()  # build eagerly so build time is measured here, not on the first query
    return index


def exact_search(X: np.ndarray, Q: np.ndarray, k: int, chunk: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Brute-force top-k by inner product in NumPy (ground truth for ANN recall)."""
    sims_out, idx_out = [], []
    for s in range(0, len(Q), chunk):
        S = Q[s:s + chunk] @ X.T
        top = np.argpartition(-S, kth=min(k, S.shape[1] - 1), axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(S, top, 1), axis=1)
        top = np.take_along_axis(top, order, 1)
        idx_out.append(top)
        sims_out.append(np.take_along_axis(S, top, 1))
    return np.concatenate(sims_out), np.concatenate(idx_out)


# ----------------------------------------------------------------------------------------
# ID-mapped store used by the app
# ----------------------------------------------------------------------------------------

class VectorStore:
    """Vectors + metadata (relative path, label) keyed by a stable 64-bit id.

    The NumPy arrays are the source of truth; the ANN index is derived from them. Additions
    go straight into the index (rebuilding only for backends that require it); removals
    rebuild the index from the arrays, which costs seconds — never a re-embedding pass.
    """

    def __init__(self, dim: int, indexer: str = "flat", **index_params):
        self.dim, self.indexer, self.index_params = dim, indexer, index_params
        self.vectors = np.zeros((0, dim), dtype=np.float32)
        self.paths: list[str] = []
        self.labels: list[str] = []
        self._pos: dict[str, int] = {}
        self._index: AnnIndex | None = None

    def __len__(self) -> int:
        return len(self.paths)

    def __contains__(self, path: str) -> bool:
        return path in self._pos

    def add(self, vectors: np.ndarray, paths: list[str], labels: list[str]) -> int:
        """Add new items; paths already present are skipped. Returns how many were added."""
        keep, seen = [], set(self._pos)
        for i, p in enumerate(paths):
            if p not in seen:
                keep.append(i)
                seen.add(p)
        if not keep:
            return 0
        V = _f32(vectors[keep])
        start = len(self.paths)
        self.vectors = np.concatenate([self.vectors, V])
        for off, i in enumerate(keep):
            self.paths.append(paths[i])
            self.labels.append(labels[i])
            self._pos[paths[i]] = start + off
        if self._index is not None and self.indexer in ("flat", "hnsw"):
            self._index.add(V)
        else:
            self._index = None  # IVF quantiser / Annoy forest should be refit on the new distribution
        return len(keep)

    def remove(self, paths: list[str]) -> int:
        drop = {self._pos[p] for p in paths if p in self._pos}
        if not drop:
            return 0
        keep = np.array([i not in drop for i in range(len(self.paths))])
        self.vectors = self.vectors[keep]
        self.paths = [p for p, k in zip(self.paths, keep) if k]
        self.labels = [l for l, k in zip(self.labels, keep) if k]
        self._pos = {p: i for i, p in enumerate(self.paths)}
        self._index = None
        return len(drop)

    def set_indexer(self, indexer: str, **index_params) -> None:
        if indexer != self.indexer or index_params != self.index_params:
            self.indexer, self.index_params, self._index = indexer, index_params, None

    @property
    def index(self) -> AnnIndex:
        if self._index is None:
            self._index = build_index(self.indexer, self.vectors, **self.index_params)
        return self._index

    def search(self, Q: np.ndarray, k: int) -> list[list[tuple[str, str, float]]]:
        """Top-k (path, label, score) per query."""
        if len(self) == 0:
            return [[] for _ in range(len(Q))]
        sims, pos = self.index.search(np.atleast_2d(Q), min(k, len(self)))
        return [[(self.paths[p], self.labels[p], float(s)) for s, p in zip(srow, prow) if p >= 0]
                for srow, prow in zip(sims, pos)]

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / "store.npz", vectors=self.vectors, paths=np.asarray(self.paths, dtype=str),
                 labels=np.asarray(self.labels, dtype=str))
        (d / "store.json").write_text(json.dumps({"dim": self.dim, "indexer": self.indexer,
                                                  "index_params": self.index_params}, indent=2))

    @classmethod
    def load(cls, directory: str | Path) -> VectorStore:
        d = Path(directory)
        meta = json.loads((d / "store.json").read_text(encoding="utf-8"))
        store = cls(meta["dim"], meta["indexer"], **meta["index_params"])
        arr = np.load(d / "store.npz", allow_pickle=False)
        store.add(arr["vectors"], arr["paths"].tolist(), arr["labels"].tolist())
        return store
