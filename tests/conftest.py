"""Shared fixtures: a tiny synthetic class-per-folder dataset and a fast fake backbone,
so the full pipeline can be tested in seconds without model downloads or network access."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from imrs import models
from imrs.config import Settings

N_CLASSES = 8
PER_CLASS = 12


class FakeEmbedder(models.Embedder):
    """Mean colour of a 16x16 thumbnail through a fixed random projection."""

    name = "fake"
    dim = 32
    supports_text = True

    def __init__(self, device="cpu"):
        super().__init__(device)
        g = torch.Generator().manual_seed(0)
        self.W = torch.randn(3 * 4, self.dim, generator=g)

    @property
    def preprocess(self):
        def prep(img: Image.Image):
            x = torch.from_numpy(np.asarray(img.resize((16, 16)), dtype=np.float32) / 255.0)
            # mean colour of the 4 quadrants -> 12 features
            q = [x[:8, :8], x[:8, 8:], x[8:, :8], x[8:, 8:]]
            return torch.cat([t.mean(dim=(0, 1)) for t in q])

        return prep

    def encode_images(self, batch):
        return F.normalize(batch @ self.W, dim=-1)

    def encode_text(self, texts):
        # "colour words" map to the embedding of a solid image of that class colour
        out = []
        for t in texts:
            c = int(t.split()[-1][-1]) if t.split()[-1][-1].isdigit() else 0
            rgb = torch.tensor(class_color(c), dtype=torch.float32) / 255.0
            out.append(rgb.repeat(4))
        return F.normalize(torch.stack(out) @ self.W, dim=-1)


def class_color(c: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(1000 + c)
    return tuple(int(v) for v in rng.integers(0, 256, 3))


def make_dataset(root, n_classes=N_CLASSES, per_class=PER_CLASS, seed=0):
    rng = np.random.default_rng(seed)
    for c in range(n_classes):
        d = root / f"{c + 1:03d}.class{c}"
        d.mkdir(parents=True)
        base = np.array(class_color(c), dtype=np.float32)
        for i in range(per_class):
            arr = np.clip(base + rng.normal(0, 18, (32, 32, 3)), 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(d / f"img_{i:03d}.jpg", quality=95)
    clutter = root / "257.clutter"
    clutter.mkdir()
    Image.fromarray(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)).save(clutter / "x.jpg")
    return root


@pytest.fixture
def dataset(tmp_path):
    return make_dataset(tmp_path / "data")


@pytest.fixture
def settings(tmp_path, dataset):
    return Settings(data_root=dataset, raw_root=dataset, runs_dir=tmp_path / "runs", device="cpu")


@pytest.fixture
def fake_backbone(monkeypatch):
    monkeypatch.setitem(models.BACKBONES, "fake", lambda device: FakeEmbedder(device))
    return "fake"


def unit_vectors(n, d, seed=0):
    X = np.random.default_rng(seed).normal(size=(n, d)).astype(np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)
