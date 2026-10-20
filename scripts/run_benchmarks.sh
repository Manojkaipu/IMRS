#!/usr/bin/env bash
# Reproduce every number in the README. CPU-only end to end is ~3 h (embedding dominates;
# everything after the first `embed` is cached). Set IMRS_DEVICE=cuda to go much faster.
set -euo pipefail

BACKBONES="clip_vit_b32 dinov2_vits14 resnet50 vit_b16"

imrs embed --backbones $BACKBONES
imrs bench --backbones $BACKBONES --indexers flat hnsw ivf_flat ivf_pq annoy
imrs bench --backbones clip_vit_b32 dinov2_vits14 --indexers flat --qe 2
imrs sweep --backbone clip_vit_b32

for loss in supcon triplet; do   # triplet last: best on held-out classes, so it stays active
  imrs finetune --backbone clip_vit_b32 --loss "$loss"
  cp runs/checkpoints/proj_clip_vit_b32.json "runs/checkpoints/proj_clip_vit_b32_${loss}.json"
done
imrs bench --backbones clip_vit_b32 clip_vit_b32_ft --eval_classes heldout --indexers flat

imrs calibrate --backbone clip_vit_b32
imrs report
