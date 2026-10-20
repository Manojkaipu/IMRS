"""Metric-learning adapter on top of a frozen backbone.

Trains a residual projection head on cached backbone embeddings using only the *training*
classes of a class-disjoint split, then reports retrieval on the held-out classes. Because
the backbone is frozen and embeddings are cached, an epoch takes seconds on a CPU.

Losses:
  supcon   supervised contrastive loss (Khosla et al., 2020)
  triplet  batch-hard triplet loss (Hermans et al., 2017)
Batches are P classes x K images so every anchor has positives and hard negatives in-batch.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .config import Settings
from .data import class_disjoint_split, load_image_set, sample_queries
from .evaluate import embeddings_for
from .index import exact_search
from .metrics import evaluate_rankings
from .models import ResidualProjection, apply_projection, save_projection

log = logging.getLogger(__name__)


@dataclass
class FinetuneConfig:
    backbone: str = "clip_vit_b32"
    loss: str = "supcon"  # supcon | triplet
    epochs: int = 30
    classes_per_batch: int = 32  # P
    samples_per_class: int = 4  # K
    lr: float = 1e-3
    weight_decay: float = 1e-4
    temperature: float = 0.1  # supcon
    margin: float = 0.2  # triplet
    hidden: int = 1024
    val_frac: float = 0.15  # of training classes; selects the best epoch (early stopping)
    patience: int = 5  # stop after this many epochs without a val improvement
    split_seed: int = 0
    seed: int = 0


def supcon_loss(z: torch.Tensor, y: torch.Tensor, temperature: float) -> torch.Tensor:
    sim = z @ z.T / temperature
    self_mask = torch.eye(len(z), dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(self_mask, -1e9)
    pos = (y[:, None] == y[None, :]) & ~self_mask
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    return -(log_prob * pos).sum(1).div(pos.sum(1).clamp(min=1)).mean()


def batch_hard_triplet_loss(z: torch.Tensor, y: torch.Tensor, margin: float) -> torch.Tensor:
    d = torch.cdist(z, z)
    same = y[:, None] == y[None, :]
    hardest_pos = (d * same).max(1).values
    hardest_neg = d.masked_fill(same, float("inf")).min(1).values
    return F.relu(hardest_pos - hardest_neg + margin).mean()


def pk_batches(labels: np.ndarray, P: int, K: int, rng: np.random.Generator):
    """One epoch of P-classes x K-samples batches (sampling with replacement inside small classes)."""
    by_class = {c: np.flatnonzero(labels == c) for c in np.unique(labels)}
    classes = np.array([c for c, idx in by_class.items() if len(idx) >= 2])
    n_batches = max(1, len(labels) // (P * K))
    for _ in range(n_batches):
        chosen = rng.choice(classes, size=min(P, len(classes)), replace=False)
        yield np.concatenate([rng.choice(by_class[c], size=K, replace=len(by_class[c]) < K) for c in chosen])


def retrieval_map(Z: np.ndarray, labels: np.ndarray, k: int = 10, n_queries: int = 1000) -> float:
    q = sample_queries(labels, n_queries)
    _, pos = exact_search(Z, Z[q], k + 1)
    pos = np.array([[p for p in row if p != qi][:k] for row, qi in zip(pos, q)])
    counts = np.bincount(labels)
    return evaluate_rankings(labels[pos], labels[q], counts[labels[q]] - 1, k)[f"mAP@{k}"]


def finetune(settings: Settings, cfg: FinetuneConfig) -> dict:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    Z, images = embeddings_for(settings, cfg.backbone, load_image_set(settings.data_root))
    labels = images.labels

    train_classes, test_classes = class_disjoint_split(len(images.classes), seed=cfg.split_seed)
    n_val = max(1, int(len(train_classes) * cfg.val_frac))
    val_classes, fit_classes = train_classes[:n_val], train_classes[n_val:]
    fit, val, test = (np.isin(labels, c) for c in (fit_classes, val_classes, test_classes))
    log.info("classes: fit=%d val=%d heldout=%d | images: fit=%d val=%d heldout=%d",
             len(fit_classes), len(val_classes), len(test_classes), fit.sum(), val.sum(), test.sum())

    device = settings.device
    head = ResidualProjection(Z.shape[1], hidden=cfg.hidden).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    X_fit = torch.from_numpy(Z[fit]).to(device)
    y_fit = torch.from_numpy(labels[fit]).to(device)

    history = []
    base_val = retrieval_map(Z[val], labels[val])
    log.info("epoch 0 | val mAP@10 %.4f (frozen backbone)", base_val)
    # epoch 0 is the identity (zero-initialised last layer), so the adapter can't end up worse than frozen on val
    best_val, best_epoch, best_state = base_val, 0, copy.deepcopy(head.state_dict())
    for epoch in range(1, cfg.epochs + 1):
        head.train()
        losses = []
        for b in pk_batches(labels[fit], cfg.classes_per_batch, cfg.samples_per_class, rng):
            z = head(X_fit[b])
            y = y_fit[b]
            loss = (supcon_loss(z, y, cfg.temperature) if cfg.loss == "supcon"
                    else batch_hard_triplet_loss(z, y, cfg.margin))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        val_map = retrieval_map(apply_projection(head, Z[val], device), labels[val])
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_mAP@10": val_map})
        log.info("epoch %d | loss %.4f | val mAP@10 %.4f", epoch, history[-1]["loss"], val_map)
        if val_map > best_val:
            best_val, best_epoch, best_state = val_map, epoch, copy.deepcopy(head.state_dict())
        elif epoch - best_epoch >= cfg.patience:
            log.info("early stop: no val improvement since epoch %d", best_epoch)
            break

    head.load_state_dict(best_state)
    result = {
        "heldout_mAP@10_before": retrieval_map(Z[test], labels[test]),
        "heldout_mAP@10_after": retrieval_map(apply_projection(head, Z[test], device), labels[test]),
        "val_mAP@10_before": base_val,
        "val_mAP@10_best": best_val,
        "best_epoch": best_epoch,
    }
    meta = {**asdict(cfg), **result, "dim": int(Z.shape[1]), "history": history}
    fp = save_projection(head.cpu(), settings.checkpoint_dir, cfg.backbone, meta)
    log.info("held-out mAP@10: %.4f -> %.4f | saved %s", result["heldout_mAP@10_before"],
             result["heldout_mAP@10_after"], fp)
    return meta
