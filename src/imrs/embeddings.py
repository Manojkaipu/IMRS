"""Batch embedding + an incremental on-disk embedding cache.

The cache stores vectors *together with* their relative image paths, so it can be reused
across runs, merged with newly added images, and never silently misaligns vectors and
labels when an image fails to decode.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .models import Embedder

log = logging.getLogger(__name__)


class _ImageFiles(Dataset):
    def __init__(self, files: Sequence[Path], preprocess):
        self.files, self.preprocess = list(files), preprocess

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        try:
            img = Image.open(self.files[i]).convert("RGB")
            return self.preprocess(img), i
        except Exception:
            return None, i


def _collate(batch):
    ok = [(x, i) for x, i in batch if x is not None]
    failed = [i for x, i in batch if x is None]
    if not ok:
        return None, [], failed
    xs, idx = zip(*ok)
    return torch.stack(xs), list(idx), failed


@torch.no_grad()
def embed_files(embedder: Embedder, files: Sequence[Path], batch_size: int = 64,
                num_workers: int = 0, progress: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Embed image files.

    Returns (embeddings, ok) where `ok` is a boolean mask over `files`; `embeddings` has one
    row per True entry, in order. Callers must filter their metadata with `ok`.
    """
    from tqdm import tqdm

    loader = DataLoader(_ImageFiles(files, embedder.preprocess), batch_size=batch_size,
                        num_workers=num_workers, collate_fn=_collate)
    out = np.zeros((len(files), embedder.dim), dtype=np.float32)
    ok = np.zeros(len(files), dtype=bool)
    for x, idx, failed in tqdm(loader, desc=f"embed[{embedder.name}]", disable=not progress):
        if failed:
            log.warning("could not read %d image(s), e.g. %s", len(failed), files[failed[0]])
        if x is None:
            continue
        out[idx] = embedder.encode_images(x.to(embedder.device)).float().cpu().numpy()
        ok[idx] = True
    return out[ok], ok


class EmbeddingCache:
    """`<dir>/emb_<backbone>.npz` holding `embeddings` (N, d) and `paths` (N,) relative to the data root."""

    def __init__(self, cache_dir: str | Path, backbone: str):
        self.fp = Path(cache_dir) / f"emb_{backbone}.npz"
        self.backbone = backbone

    def load(self) -> dict[str, np.ndarray]:
        if not self.fp.exists():
            return {}
        d = np.load(self.fp, allow_pickle=False)
        if "paths" not in d.files:  # legacy file from the original course version: no paths, unusable
            log.warning("%s has no 'paths' array; ignoring it", self.fp)
            return {}
        return dict(zip(d["paths"].tolist(), d["embeddings"]))

    def save(self, paths: Sequence[str], embeddings: np.ndarray) -> None:
        self.fp.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.fp.with_suffix(".tmp.npz")
        np.savez(tmp, embeddings=embeddings.astype(np.float32), paths=np.asarray(paths, dtype=str))
        tmp.replace(self.fp)

    def get(self, embedder: Embedder | Callable[[], Embedder], root: Path, paths: Sequence[str],
            **kw) -> tuple[np.ndarray, np.ndarray]:
        """Embeddings for `paths`, computing only the ones not cached yet.

        `embedder` may be a zero-argument factory, so a fully cached run never loads a model.
        Returns (embeddings, ok) with the same contract as `embed_files`.
        """
        cached = self.load()
        missing = [p for p in paths if p not in cached]
        if missing:
            if not isinstance(embedder, Embedder):
                embedder = embedder()
            log.info("%s: %d cached, embedding %d new", self.fp.name, len(paths) - len(missing), len(missing))
            Z, ok = embed_files(embedder, [Path(root) / p for p in missing], **kw)
            cached.update(zip([p for p, k in zip(missing, ok) if k], Z))
            if cached:
                self.save(list(cached), np.stack(list(cached.values())))
        ok = np.array([p in cached for p in paths], dtype=bool)
        if not ok.any():
            return np.zeros((0, 0), np.float32), ok
        Z = np.stack([cached[p] for p, k in zip(paths, ok) if k])
        return Z.astype(np.float32), ok
