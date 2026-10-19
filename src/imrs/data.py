"""Dataset preprocessing, listing and evaluation splits."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

from .config import DYNAMIC_PREFIX, IMAGE_EXTS

ImageFile.LOAD_TRUNCATED_IMAGES = True
log = logging.getLogger(__name__)

# Caltech-256's "clutter" category is background noise, not an object class; retrieval
# papers drop it, and keeping it would reward matching random backgrounds.
EXCLUDED_CLASSES = ("257.clutter",)
MANIFEST = "manifest.csv"


@dataclass
class ImageSet:
    """Image paths (relative to `root`, posix style) with integer class labels."""

    root: Path
    paths: list[str]
    labels: np.ndarray
    classes: list[str]

    def __len__(self) -> int:
        return len(self.paths)

    def abspath(self, rel: str) -> Path:
        return self.root / rel

    def subset(self, mask: np.ndarray) -> ImageSet:
        idx = np.flatnonzero(mask)
        return ImageSet(self.root, [self.paths[i] for i in idx], self.labels[idx], self.classes)


def pretty_class(name: str) -> str:
    """'003.backpack' -> 'backpack', 'dynamic_red-sports-car' -> 'red sports car'."""
    if name.startswith(DYNAMIC_PREFIX):
        name = name[len(DYNAMIC_PREFIX):]
    head, _, tail = name.partition(".")
    if tail and head.isdigit():
        name = tail
    return name.replace("-101", "").replace("_", " ").replace("-", " ").strip()


def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def load_image_set(root: str | Path, include_dynamic: bool = False,
                   exclude: tuple[str, ...] = EXCLUDED_CLASSES) -> ImageSet:
    """List a class-per-folder dataset.

    Uses `manifest.csv` when present (written by `preprocess`), otherwise walks the folders.
    Web-expansion folders (`dynamic_*`) are skipped unless `include_dynamic` is set; they are
    then appended as extra classes so they never change the label ids of the base classes.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root} (set IMRS_DATA_ROOT or pass --data_root)")

    man = root / MANIFEST
    if man.exists():
        df = pd.read_csv(man)
        df = df[~df["class_name"].isin(exclude)]
        df = df[[(root / p).exists() for p in df["path"]]]
        classes = sorted(df["class_name"].unique())
        c2i = {c: i for i, c in enumerate(classes)}
        paths = df["path"].tolist()
        labels = [c2i[c] for c in df["class_name"]]
    else:
        classes = sorted(d.name for d in root.iterdir()
                         if d.is_dir() and not d.name.startswith(DYNAMIC_PREFIX) and d.name not in exclude)
        paths, labels = [], []
        for ci, c in enumerate(classes):
            for p in sorted((root / c).iterdir()):
                if _is_image(p):
                    paths.append(p.relative_to(root).as_posix())
                    labels.append(ci)

    if include_dynamic:
        for d in sorted(root.glob(f"{DYNAMIC_PREFIX}*")):
            if not d.is_dir():
                continue
            files = [p for p in sorted(d.iterdir()) if _is_image(p)]
            if files:
                classes.append(d.name)
                paths += [p.relative_to(root).as_posix() for p in files]
                labels += [len(classes) - 1] * len(files)

    return ImageSet(root, paths, np.asarray(labels, dtype=np.int64), list(classes))


# ----------------------------------------------------------------------------------------
# Splits
# ----------------------------------------------------------------------------------------

def class_disjoint_split(n_classes: int, train_frac: float = 0.5, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Split class ids into train / held-out sets (the standard metric-learning protocol:
    a fine-tuned model is only ever evaluated on classes it never saw during training)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_classes)
    n_train = int(round(n_classes * train_frac))
    return np.sort(perm[:n_train]), np.sort(perm[n_train:])


def sample_queries(labels: np.ndarray, n_queries: int, seed: int = 123) -> np.ndarray:
    """Pick query indices stratified over classes so small classes are not under-represented."""
    rng = np.random.default_rng(seed)
    n_queries = min(n_queries, len(labels))
    classes = np.unique(labels)
    per_class = max(1, n_queries // len(classes))
    picked = []
    for c in classes:
        idx = np.flatnonzero(labels == c)
        picked.extend(rng.choice(idx, size=min(per_class, len(idx)), replace=False))
    picked = np.asarray(picked)
    if len(picked) < n_queries:  # top up uniformly at random
        rest = np.setdiff1d(np.arange(len(labels)), picked)
        picked = np.concatenate([picked, rng.choice(rest, n_queries - len(picked), replace=False)])
    return np.sort(rng.permutation(picked)[:n_queries])


# ----------------------------------------------------------------------------------------
# Preprocessing (resize + pad, perceptual-hash de-duplication, optional contrast/colour norm)
# ----------------------------------------------------------------------------------------

def aspect_pad_resize(img: Image.Image, size: int = 224) -> Image.Image:
    w, h = img.size
    scale = size / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    img = img.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def apply_clahe(img: Image.Image, clip: float = 2.0, tile: int = 8) -> Image.Image:
    import cv2

    lab = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2LAB)
    lum, a, b = cv2.split(lab)
    lum = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(lum)
    return Image.fromarray(cv2.cvtColor(cv2.merge([lum, a, b]), cv2.COLOR_LAB2RGB))


def color_normalize(img: Image.Image) -> Image.Image:
    x = np.asarray(img, dtype=np.float32)
    x = (x - x.mean(axis=(0, 1))) / (x.std(axis=(0, 1)) + 1e-6) * 50 + 128
    return Image.fromarray(np.clip(x, 0, 255).astype(np.uint8))


def preprocess_dataset(raw_root: str | Path, out_root: str | Path, img_size: int = 224,
                       phash_threshold: int = 4, use_clahe: bool = False, color_norm: bool = False) -> pd.DataFrame:
    """Resize/pad every image, drop near-duplicates within each class, write `manifest.csv`.

    Near-duplicates matter for retrieval: an exact copy of the query sitting in the gallery
    inflates every metric without saying anything about the model.
    """
    from imagehash import phash
    from tqdm import tqdm

    raw_root, out_root = Path(raw_root), Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    classes = sorted(d.name for d in raw_root.iterdir() if d.is_dir())
    rows, skipped_dup, skipped_bad = [], 0, 0
    for cls in tqdm(classes, desc="preprocess"):
        (out_root / cls).mkdir(exist_ok=True)
        seen = []
        for p in sorted((raw_root / cls).iterdir()):
            if not _is_image(p):
                continue
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                skipped_bad += 1
                continue
            h = phash(img)  # hash the original so enhancement settings don't change what counts as a duplicate
            if any(h - o <= phash_threshold for o in seen):
                skipped_dup += 1
                continue
            seen.append(h)
            if use_clahe:
                img = apply_clahe(img)
            if color_norm:
                img = color_normalize(img)
            rel = f"{cls}/{p.stem}.jpg"
            aspect_pad_resize(img, img_size).save(out_root / rel, quality=95)
            rows.append({"path": rel, "class_name": cls})
    df = pd.DataFrame(rows)
    df.to_csv(out_root / MANIFEST, index=False)
    log.info("preprocess: kept=%d duplicates=%d unreadable=%d classes=%d",
             len(df), skipped_dup, skipped_bad, len(classes))
    return df
