"""Runtime configuration.

Every path can be set with an environment variable (or a `.env`-style shell export),
so nothing in the codebase is tied to a particular machine.

    IMRS_DATA_ROOT   preprocessed dataset (one sub-folder per class)      default: data/caltech256_proc
    IMRS_RAW_ROOT    raw Caltech-256 download                             default: data/256_ObjectCategories
    IMRS_RUNS_DIR    benchmark outputs, embedding caches, checkpoints     default: runs
    IMRS_DEVICE      "cuda" / "cpu" / "mps"                               default: auto
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def default_device() -> str:
    env = os.environ.get("IMRS_DEVICE")
    if env:
        return env
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


@dataclass
class Settings:
    data_root: Path = field(default_factory=lambda: Path(os.environ.get("IMRS_DATA_ROOT", "data/caltech256_proc")))
    raw_root: Path = field(default_factory=lambda: Path(os.environ.get("IMRS_RAW_ROOT", "data/256_ObjectCategories")))
    runs_dir: Path = field(default_factory=lambda: Path(os.environ.get("IMRS_RUNS_DIR", "runs")))
    device: str = field(default_factory=default_device)

    @property
    def cache_dir(self) -> Path:
        """Embedding caches (`emb_<backbone>.npz`), shared by the benchmark and the app."""
        return self.runs_dir / "cache"

    @property
    def app_state_dir(self) -> Path:
        """Vector-store snapshots written by the Streamlit app."""
        return self.runs_dir / "app_state"

    @property
    def checkpoint_dir(self) -> Path:
        return self.runs_dir / "checkpoints"


# Folders created by web expansion live next to the class folders but are never treated as classes.
DYNAMIC_PREFIX = "dynamic_"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")
