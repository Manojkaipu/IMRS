"""Benchmarking: retrieval quality, speed and ANN accuracy for backbone x index combinations.

Protocol (leave-one-out over a stratified query sample):
  * gallery = every image of the evaluated classes; each query is removed from its own results
  * relevant = same class; metrics normalised by the class size in the gallery
  * ANN indexes are also scored against exact search on the same vectors (neighbour recall),
    which isolates the error the index introduces from the error the embedding introduces
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from functools import partial

import numpy as np

from .config import Settings
from .data import ImageSet, class_disjoint_split, load_image_set, sample_queries
from .embeddings import EmbeddingCache
from .index import AnnIndex, build_index, exact_search
from .metrics import evaluate_rankings, neighbor_recall
from .models import FINETUNED_SUFFIX, apply_projection, load_embedder, load_projection

log = logging.getLogger(__name__)


@dataclass
class BenchConfig:
    k: int = 10
    n_queries: int = 2000
    eval_classes: str = "all"  # "all" | "heldout"
    split_seed: int = 0
    query_seed: int = 123
    query_expansion: int = 0  # average query expansion over the top-n neighbours (0 = off)
    n_latency_queries: int = 200
    index_params: dict = field(default_factory=dict)


def embeddings_for(settings: Settings, backbone: str, images: ImageSet, batch_size: int = 64,
                   num_workers: int = 0, base_embedder=None, progress: bool = True) -> tuple[np.ndarray, ImageSet]:
    """Embeddings (from cache where possible) for `images`; drops images that failed to load.

    Fine-tuned backbones (`<base>_ft`) reuse the base model's cache and only apply the head.
    `base_embedder` lets callers that already hold the base model (the app) avoid reloading it.
    """
    base = backbone[: -len(FINETUNED_SUFFIX)] if backbone.endswith(FINETUNED_SUFFIX) else backbone
    cache = EmbeddingCache(settings.cache_dir, base)
    Z, ok = cache.get(base_embedder or partial(load_embedder, base, settings.device), images.root, images.paths,
                      batch_size=batch_size, num_workers=num_workers, progress=progress)
    images = images.subset(ok)
    if base != backbone and len(Z):
        head, _ = load_projection(settings.checkpoint_dir, base, Z.shape[1])
        Z = apply_projection(head, Z, settings.device)
    return Z, images



def eval_subset(images: ImageSet, cfg: BenchConfig) -> np.ndarray:
    """Boolean mask of the images that belong to the evaluated classes."""
    if cfg.eval_classes == "all":
        return np.ones(len(images), dtype=bool)
    if cfg.eval_classes == "heldout":
        _, test_classes = class_disjoint_split(len(images.classes), seed=cfg.split_seed)
        return np.isin(images.labels, test_classes)
    raise ValueError(cfg.eval_classes)


def _drop_self(sims: np.ndarray, pos: np.ndarray, q_idx: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    out_s = np.full((len(q_idx), k), -np.inf, dtype=np.float32)
    out_p = np.full((len(q_idx), k), -1, dtype=np.int64)
    for i, (s, p) in enumerate(zip(sims, pos)):
        keep = p != q_idx[i]
        s, p = s[keep][:k], p[keep][:k]
        out_s[i, : len(s)], out_p[i, : len(p)] = s, p
    return out_s, out_p


def _search_excluding_self(index: AnnIndex, X: np.ndarray, q_idx: np.ndarray, k: int,
                           qe: int = 0) -> tuple[np.ndarray, np.ndarray, float]:
    t0 = time.perf_counter()
    sims, pos = _drop_self(*index.search(X[q_idx], k + 1), q_idx, k)
    if qe > 0:
        # Average query expansion: the query plus its top-n neighbours (Chum et al., 2007).
        nb = np.where(pos[:, :qe] >= 0, pos[:, :qe], q_idx[:, None])
        Q2 = X[q_idx] + X[nb].sum(axis=1)
        Q2 /= np.linalg.norm(Q2, axis=1, keepdims=True)
        s2, p2 = index.search(Q2, k + 1)
        sims, pos = _drop_self(s2, p2, q_idx, k)
    return sims, pos, time.perf_counter() - t0


def index_size_bytes(index: AnnIndex) -> int | None:
    if hasattr(index, "faiss"):
        return int(index.faiss.serialize_index(index.index).nbytes)
    return None


def benchmark_index(X: np.ndarray, labels: np.ndarray, indexer: str, cfg: BenchConfig,
                    q_idx: np.ndarray, exact_pos: np.ndarray | None = None) -> dict:
    k = cfg.k
    t0 = time.perf_counter()
    index = build_index(indexer, X, **cfg.index_params.get(indexer, {}))
    build_s = time.perf_counter() - t0

    _, pos, search_s = _search_excluding_self(index, X, q_idx, k, cfg.query_expansion)

    # single-query latency (what an interactive user actually feels)
    lat = []
    for qi in q_idx[: cfg.n_latency_queries]:
        t = time.perf_counter()
        index.search(X[qi: qi + 1], k + 1)
        lat.append((time.perf_counter() - t) * 1e3)

    counts = np.bincount(labels, minlength=labels.max() + 1)
    ranked = np.where(pos >= 0, labels[np.clip(pos, 0, None)], -1)
    row = {
        **evaluate_rankings(ranked, labels[q_idx], counts[labels[q_idx]] - 1, k),
        "build_s": build_s,
        "qps_batch": len(q_idx) / search_s,
        "latency_ms_p50": float(np.percentile(lat, 50)),
        "latency_ms_p95": float(np.percentile(lat, 95)),
        "index_mb": (lambda b: None if b is None else b / 2**20)(index_size_bytes(index)),
    }
    if exact_pos is not None and cfg.query_expansion == 0:
        row[f"ann_recall@{k}"] = neighbor_recall(pos, exact_pos, k)
    return row


def run_benchmark(settings: Settings, backbones: list[str], indexers: list[str], cfg: BenchConfig,
                  batch_size: int = 64, num_workers: int = 0) -> list[dict]:
    images_all = load_image_set(settings.data_root)
    rows = []
    for backbone in backbones:
        eval_classes = cfg.eval_classes
        if backbone.endswith(FINETUNED_SUFFIX) and eval_classes != "heldout":
            log.warning("%s was trained on half the classes; evaluating on the held-out half only", backbone)
            eval_classes = "heldout"
        cfg_b = BenchConfig(**{**cfg.__dict__, "eval_classes": eval_classes})
        Z, images = embeddings_for(settings, backbone, images_all, batch_size, num_workers)
        mask = eval_subset(images, cfg_b)
        X, labels = Z[mask], images.labels[mask]
        q_idx = sample_queries(labels, cfg.n_queries, seed=cfg.query_seed)
        _, exact_pos = _drop_self(*exact_search(X, X[q_idx], cfg.k + 1), q_idx, cfg.k)
        for indexer in indexers:
            log.info("bench %s / %s on %d images (%d queries)", backbone, indexer, len(X), len(q_idx))
            row = {"backbone": backbone, "indexer": indexer, "eval_classes": eval_classes,
                   "n_gallery": len(X), "n_queries": len(q_idx), "dim": X.shape[1], "k": cfg.k,
                   "query_expansion": cfg.query_expansion}
            row.update(benchmark_index(X, labels, indexer, cfg_b, q_idx, exact_pos))
            log.info("  %s", {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
            rows.append(row)
    return rows


# Speed/accuracy knob per index type, swept from fast-and-rough to slow-and-exact.
SWEEPS = {
    "hnsw": ("ef_search", [8, 16, 32, 64, 128, 256, 512]),
    "ivf_flat": ("nprobe", [1, 2, 4, 8, 16, 32, 64]),
    "ivf_pq": ("nprobe", [1, 2, 4, 8, 16, 32, 64]),
    "annoy": ("search_k", [100, 500, 1000, 5000, 10000, 50000]),
}


def run_sweep(settings: Settings, backbone: str, indexers: list[str], cfg: BenchConfig) -> list[dict]:
    """Recall-vs-throughput curves: build each index once, then vary its search-time knob."""
    Z, images = embeddings_for(settings, backbone, load_image_set(settings.data_root))
    mask = eval_subset(images, cfg)
    X, labels = Z[mask], images.labels[mask]
    q_idx = sample_queries(labels, cfg.n_queries, seed=cfg.query_seed)
    _, exact_pos = _drop_self(*exact_search(X, X[q_idx], cfg.k + 1), q_idx, cfg.k)
    counts = np.bincount(labels)
    rows = []
    for indexer in indexers:
        if indexer not in SWEEPS:
            continue
        knob, values = SWEEPS[indexer]
        index = build_index(indexer, X, **cfg.index_params.get(indexer, {}))
        for v in values:
            index.set_search_params(**{knob: v})
            index.search(X[q_idx[:50]], cfg.k + 1)  # warm-up
            _, pos, secs = _search_excluding_self(index, X, q_idx, cfg.k)
            ranked = np.where(pos >= 0, labels[np.clip(pos, 0, None)], -1)
            m = evaluate_rankings(ranked, labels[q_idx], counts[labels[q_idx]] - 1, cfg.k)
            rows.append({"backbone": backbone, "indexer": indexer, "param": knob, "value": v,
                         f"ann_recall@{cfg.k}": neighbor_recall(pos, exact_pos, cfg.k),
                         f"mAP@{cfg.k}": m[f"mAP@{cfg.k}"], "qps_batch": len(q_idx) / secs})
            log.info("sweep %s %s=%s -> %s", indexer, knob, v, rows[-1])
    return rows


def calibrate_coverage(settings: Settings, backbone: str = "clip_vit_b32", min_good: int = 3, k: int = 10,
                       prompt: str = "a photo of a {}") -> tuple[dict, list[dict]]:
    """Measure the "is this concept missing?" trigger that drives web expansion.

    For every class we issue its text prompt twice: against the full gallery (concept present)
    and against the gallery with that class removed (concept missing, the situation expansion
    is meant to fix). The trigger looks at the `min_good`-th best cosine score; we report how
    well that separates the two cases (ROC-AUC) and the threshold maximising balanced accuracy.
    Also reports text->image retrieval quality (P@k of class prompts) as a by-product.
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    from .data import pretty_class

    images_all = load_image_set(settings.data_root)
    Z, images = embeddings_for(settings, backbone, images_all)
    embedder = load_embedder(backbone, settings.device, settings.checkpoint_dir)
    if not embedder.supports_text:
        raise ValueError(f"{backbone} has no text encoder; calibrate a CLIP backbone")
    names = [pretty_class(c) for c in images.classes]
    T = embedder.embed_text([prompt.format(n) for n in names])
    S = T @ Z.T  # (classes, images)

    rows, p_at_k = [], []
    for c, name in enumerate(names):
        s = S[c]
        top = np.argsort(-s)[:k]
        p_at_k.append(float(np.mean(images.labels[top] == c)))
        present = np.sort(s)[-min_good]
        absent = np.sort(s[images.labels != c])[-min_good]
        rows.append({"class": name, "score_present": float(present), "score_absent": float(absent),
                     f"text_P@{k}": p_at_k[-1]})

    y = np.r_[np.ones(len(rows)), np.zeros(len(rows))]
    scores = np.r_[[r["score_present"] for r in rows], [r["score_absent"] for r in rows]]
    fpr, tpr, thr = roc_curve(y, scores)
    best = int(np.argmax(tpr - fpr))
    summary = {
        "backbone": backbone, "min_good": min_good, "n_classes": len(rows),
        "roc_auc": float(roc_auc_score(y, scores)),
        "best_threshold": float(thr[best]),
        "tpr_at_best": float(tpr[best]),  # concept present -> correctly NOT expanded
        "fpr_at_best": float(fpr[best]),  # concept missing -> wrongly NOT expanded
        f"text_to_image_P@{k}": float(np.mean(p_at_k)),
    }
    return summary, rows
