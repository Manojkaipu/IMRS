"""Image (and text) encoders. Every encoder returns L2-normalised float32 embeddings, so
inner product == cosine similarity everywhere downstream."""
from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

log = logging.getLogger(__name__)


class Embedder(nn.Module):
    name: str = "base"
    dim: int = 0
    supports_text: bool = False

    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device

    @property
    def preprocess(self) -> Callable[[Image.Image], torch.Tensor]:
        raise NotImplementedError

    def encode_images(self, batch: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def encode_text(self, texts: Sequence[str]) -> torch.Tensor:
        raise NotImplementedError(f"{self.name} has no text encoder; use a CLIP backbone for text search")

    @torch.no_grad()
    def embed_pils(self, imgs: Sequence[Image.Image]) -> np.ndarray:
        if not imgs:
            return np.zeros((0, self.dim), dtype=np.float32)
        x = torch.stack([self.preprocess(im.convert("RGB")) for im in imgs]).to(self.device)
        return self.encode_images(x).float().cpu().numpy()

    @torch.no_grad()
    def embed_text(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode_text(texts).float().cpu().numpy()


class CLIPEmbedder(Embedder):
    supports_text = True

    def __init__(self, variant: str = "ViT-B/32", device: str = "cpu"):
        super().__init__(device)
        import clip

        self._clip = clip
        self.model, self._prep = clip.load(variant, device=device)
        self.model.eval()
        self.dim = self.model.visual.output_dim

    @property
    def preprocess(self):
        return self._prep

    @torch.no_grad()
    def encode_images(self, batch):
        return F.normalize(self.model.encode_image(batch).float(), dim=-1)

    @torch.no_grad()
    def encode_text(self, texts):
        tokens = self._clip.tokenize(list(texts), truncate=True).to(self.device)
        return F.normalize(self.model.encode_text(tokens).float(), dim=-1)


class TorchvisionEmbedder(Embedder):
    """ImageNet-pretrained torchvision classifier with the classification head removed."""

    def __init__(self, arch: str, device: str = "cpu"):
        super().__init__(device)
        import torchvision as tv

        if arch == "resnet50":
            weights = tv.models.ResNet50_Weights.IMAGENET1K_V2
            self.model = tv.models.resnet50(weights=weights)
            self.model.fc = nn.Identity()
            self.dim = 2048
        elif arch == "vit_b16":
            weights = tv.models.ViT_B_16_Weights.IMAGENET1K_V1
            self.model = tv.models.vit_b_16(weights=weights)
            self.model.heads = nn.Identity()
            self.dim = 768
        else:
            raise ValueError(arch)
        self._prep = weights.transforms()
        self.model.to(device).eval()

    @property
    def preprocess(self):
        return self._prep

    @torch.no_grad()
    def encode_images(self, batch):
        return F.normalize(self.model(batch), dim=-1)


class DINOv2Embedder(Embedder):
    """Self-supervised DINOv2 (Meta) loaded through torch.hub; CLS-token features."""

    def __init__(self, arch: str = "dinov2_vits14", device: str = "cpu"):
        super().__init__(device)
        import torchvision.transforms as T

        self.model = torch.hub.load("facebookresearch/dinov2", arch)
        self.model.to(device).eval()
        self.dim = self.model.embed_dim
        self._prep = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    @property
    def preprocess(self):
        return self._prep

    @torch.no_grad()
    def encode_images(self, batch):
        return F.normalize(self.model(batch), dim=-1)


BACKBONES: dict[str, Callable[[str], Embedder]] = {
    "clip_vit_b32": lambda device: CLIPEmbedder("ViT-B/32", device),
    "clip_vit_b16": lambda device: CLIPEmbedder("ViT-B/16", device),
    "clip_vit_l14": lambda device: CLIPEmbedder("ViT-L/14", device),
    "resnet50": lambda device: TorchvisionEmbedder("resnet50", device),
    "vit_b16": lambda device: TorchvisionEmbedder("vit_b16", device),
    "dinov2_vits14": lambda device: DINOv2Embedder("dinov2_vits14", device),
    "dinov2_vitb14": lambda device: DINOv2Embedder("dinov2_vitb14", device),
}

FINETUNED_SUFFIX = "_ft"


# ----------------------------------------------------------------------------------------
# Fine-tuned projection head
# ----------------------------------------------------------------------------------------

class ResidualProjection(nn.Module):
    """x -> normalize(x + MLP(x)).

    The last layer is zero-initialised, so training starts from exactly the pretrained
    embedding space and can only move away from it when the loss says so. This keeps the
    adapter from destroying the backbone's general features on classes it never trains on.
    """

    def __init__(self, dim: int, hidden: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x + self.mlp(x), dim=-1)


class ProjectedEmbedder(Embedder):
    supports_text = False  # the adapter is trained on image embeddings only

    def __init__(self, base: Embedder, head: ResidualProjection, name: str):
        super().__init__(base.device)
        self.base, self.head, self.name, self.dim = base, head.to(base.device).eval(), name, base.dim

    @property
    def preprocess(self):
        return self.base.preprocess

    @torch.no_grad()
    def encode_images(self, batch):
        return self.head(self.base.encode_images(batch))



@torch.no_grad()
def apply_projection(head: ResidualProjection, z: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Apply a trained head to cached base embeddings (no backbone or image decoding needed)."""
    head = head.to(device).eval()
    t = torch.from_numpy(np.ascontiguousarray(z, dtype=np.float32)).to(device)
    return head(t).cpu().numpy()


def checkpoint_path(checkpoint_dir: Path, base_name: str) -> Path:
    return Path(checkpoint_dir) / f"proj_{base_name}.pt"


def save_projection(head: ResidualProjection, checkpoint_dir: Path, base_name: str, meta: dict) -> Path:
    fp = checkpoint_path(checkpoint_dir, base_name)
    fp.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "meta": meta}, fp)
    fp.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return fp


def load_projection(checkpoint_dir: Path, base_name: str, dim: int) -> tuple[ResidualProjection, dict]:
    fp = checkpoint_path(checkpoint_dir, base_name)
    if not fp.exists():
        raise FileNotFoundError(f"{fp} not found - run `imrs finetune --backbone {base_name}` first")
    ckpt = torch.load(fp, map_location="cpu", weights_only=True)
    meta = ckpt["meta"]
    head = ResidualProjection(dim, hidden=meta.get("hidden", 1024))
    head.load_state_dict(ckpt["state_dict"])
    return head, meta


def available_backbones(checkpoint_dir: Path | None = None) -> list[str]:
    names = list(BACKBONES)
    if checkpoint_dir is not None:
        names += [n + FINETUNED_SUFFIX for n in BACKBONES if checkpoint_path(checkpoint_dir, n).exists()]
    return names


def load_embedder(name: str, device: str = "cpu", checkpoint_dir: Path | None = None) -> Embedder:
    if name.endswith(FINETUNED_SUFFIX):
        base_name = name[: -len(FINETUNED_SUFFIX)]
        base = load_embedder(base_name, device)
        head, _ = load_projection(checkpoint_dir or Path("runs/checkpoints"), base_name, base.dim)
        return ProjectedEmbedder(base, head, name)
    if name not in BACKBONES:
        raise ValueError(f"Unknown backbone '{name}'. Choose from: {', '.join(BACKBONES)}")
    log.info("loading %s on %s", name, device)
    emb = BACKBONES[name](device)
    emb.name = name
    return emb
