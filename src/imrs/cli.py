"""`imrs` command-line interface.

    imrs preprocess   raw Caltech-256 -> resized, de-duplicated copy + manifest.csv
    imrs embed        fill the embedding cache for one or more backbones
    imrs bench        retrieval quality / speed for backbone x index combinations
    imrs sweep        recall-vs-QPS curves for approximate indexes
    imrs finetune     train a metric-learning adapter on held-in classes
    imrs calibrate    measure / tune the web-expansion trigger
    imrs search       query the collection from the terminal (text or image)
    imrs expand       fetch web images for new concepts
    imrs report       markdown leaderboard + plots from saved results
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from .config import Settings
from .index import INDEXERS
from .models import BACKBONES, FINETUNED_SUFFIX


def backbone(name: str) -> str:
    """argparse type: a registered backbone, optionally with the fine-tuned `_ft` suffix."""
    base = name[: -len(FINETUNED_SUFFIX)] if name.endswith(FINETUNED_SUFFIX) else name
    if base not in BACKBONES:
        raise argparse.ArgumentTypeError(f"unknown backbone '{name}' (choose from {', '.join(BACKBONES)}, "
                                         f"or <name>{FINETUNED_SUFFIX} after fine-tuning)")
    return name


def base_backbone(name: str) -> str:
    if name.endswith(FINETUNED_SUFFIX):
        raise argparse.ArgumentTypeError("expected a base backbone, not a fine-tuned one")
    return backbone(name)


def _settings(args) -> Settings:
    s = Settings()
    if getattr(args, "data_root", None):
        s.data_root = Path(args.data_root)
    if getattr(args, "runs_dir", None):
        s.runs_dir = Path(args.runs_dir)
    if getattr(args, "device", None):
        s.device = args.device
    return s


def _append_csv(rows: list[dict], fp: Path) -> None:
    fp.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(fp, mode="a", header=not fp.exists(), index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n-> {fp}")


def cmd_preprocess(args):
    from .data import preprocess_dataset

    s = _settings(args)
    preprocess_dataset(args.raw_root or s.raw_root, s.data_root, args.img_size, args.phash_threshold,
                       use_clahe=args.clahe, color_norm=args.color_norm)


def cmd_embed(args):
    from .data import load_image_set
    from .evaluate import embeddings_for

    s = _settings(args)
    images = load_image_set(s.data_root, include_dynamic=args.include_dynamic)
    for b in args.backbones:
        Z, ok_images = embeddings_for(s, b, images, args.batch_size, args.workers)
        print(f"{b}: {len(ok_images)}/{len(images)} images embedded, dim={Z.shape[1]}")


def cmd_bench(args):
    from .evaluate import BenchConfig, run_benchmark

    s = _settings(args)
    cfg = BenchConfig(k=args.k, n_queries=args.queries, eval_classes=args.eval_classes,
                      query_expansion=args.qe, index_params=json.loads(args.index_params))
    rows = run_benchmark(s, args.backbones, args.indexers, cfg, args.batch_size, args.workers)
    _append_csv(rows, s.runs_dir / "bench" / "results.csv")


def cmd_sweep(args):
    from .evaluate import BenchConfig, run_sweep

    s = _settings(args)
    cfg = BenchConfig(k=args.k, n_queries=args.queries, eval_classes=args.eval_classes)
    rows = run_sweep(s, args.backbone, args.indexers, cfg)
    fp = s.runs_dir / "bench" / f"sweep_{args.backbone}.csv"
    fp.unlink(missing_ok=True)
    _append_csv(rows, fp)


def cmd_finetune(args):
    from .finetune import FinetuneConfig, finetune

    s = _settings(args)
    cfg = FinetuneConfig(backbone=args.backbone, loss=args.loss, epochs=args.epochs, lr=args.lr,
                         classes_per_batch=args.P, samples_per_class=args.K,
                         temperature=args.temperature, margin=args.margin)
    meta = finetune(s, cfg)
    print(json.dumps({k: v for k, v in meta.items() if k != "history"}, indent=2))


def cmd_calibrate(args):
    from .evaluate import calibrate_coverage

    s = _settings(args)
    summary, rows = calibrate_coverage(s, args.backbone, args.min_good)
    (s.runs_dir / "bench").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(s.runs_dir / "bench" / f"calibration_{args.backbone}_per_class.csv", index=False)
    _append_csv([summary], s.runs_dir / "bench" / "calibration.csv")


def cmd_search(args):
    import numpy as np
    from PIL import Image

    from .data import load_image_set
    from .evaluate import embeddings_for
    from .index import build_index
    from .models import load_embedder

    s = _settings(args)
    images = load_image_set(s.data_root, include_dynamic=True)
    Z, images = embeddings_for(s, args.backbone, images)
    emb = load_embedder(args.backbone, s.device, s.checkpoint_dir)
    q = emb.embed_text([args.text]) if args.text else emb.embed_pils([Image.open(args.image)])
    sims, pos = build_index(args.indexer, Z).search(q.astype(np.float32), args.k)
    for rank, (sc, p) in enumerate(zip(sims[0], pos[0]), 1):
        print(f"{rank:>3}. {sc:.3f}  {images.classes[images.labels[p]]:<28} {images.paths[p]}")


def cmd_expand(args):
    from .expansion import fetch_term

    s = _settings(args)
    terms = list(args.terms)
    if args.terms_file:
        terms += [t.strip() for t in Path(args.terms_file).read_text(encoding="utf-8").splitlines() if t.strip()]
    total = 0
    for t in terms:
        total += len(fetch_term(s.data_root, t, n=args.per_term))
    print(f"added {total} images for {len(terms)} term(s); run `imrs embed --include_dynamic` to index them")


def cmd_report(args):
    from .report import build_report

    print(build_report(_settings(args).runs_dir))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="imrs", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, workers=False):
        sp.add_argument("--data_root", help="preprocessed dataset (env: IMRS_DATA_ROOT)")
        sp.add_argument("--runs_dir", help="outputs + caches (env: IMRS_RUNS_DIR)")
        sp.add_argument("--device", help="cuda / cpu / mps (env: IMRS_DEVICE)")
        if workers:
            sp.add_argument("--batch_size", type=int, default=64)
            sp.add_argument("--workers", type=int, default=0, help="DataLoader worker processes")
        return sp

    sp = common(sub.add_parser("preprocess"))
    sp.add_argument("--raw_root", help="raw dataset (env: IMRS_RAW_ROOT)")
    sp.add_argument("--img_size", type=int, default=224)
    sp.add_argument("--phash_threshold", type=int, default=4)
    sp.add_argument("--clahe", action="store_true")
    sp.add_argument("--color_norm", action="store_true")
    sp.set_defaults(func=cmd_preprocess)

    sp = common(sub.add_parser("embed"), workers=True)
    sp.add_argument("--backbones", nargs="+", default=["clip_vit_b32"], type=base_backbone)
    sp.add_argument("--include_dynamic", action="store_true", help="also embed web-expansion folders")
    sp.set_defaults(func=cmd_embed)

    sp = common(sub.add_parser("bench"), workers=True)
    sp.add_argument("--backbones", nargs="+", default=["clip_vit_b32", "resnet50", "vit_b16"], type=backbone)
    sp.add_argument("--indexers", nargs="+", default=["flat", "hnsw", "ivf_flat", "ivf_pq", "annoy"], choices=INDEXERS)
    sp.add_argument("--k", type=int, default=10)
    sp.add_argument("--queries", type=int, default=2000)
    sp.add_argument("--eval_classes", choices=["all", "heldout"], default="all")
    sp.add_argument("--qe", type=int, default=0, help="average query expansion with top-n neighbours")
    sp.add_argument("--index_params", default="{}", help='JSON, e.g. \'{"hnsw": {"ef_search": 128}}\'')
    sp.set_defaults(func=cmd_bench)

    sp = common(sub.add_parser("sweep"))
    sp.add_argument("--backbone", default="clip_vit_b32", type=backbone)
    sp.add_argument("--indexers", nargs="+", default=["hnsw", "ivf_flat", "ivf_pq", "annoy"], choices=INDEXERS)
    sp.add_argument("--k", type=int, default=10)
    sp.add_argument("--queries", type=int, default=2000)
    sp.add_argument("--eval_classes", choices=["all", "heldout"], default="all")
    sp.set_defaults(func=cmd_sweep)

    sp = common(sub.add_parser("finetune"))
    sp.add_argument("--backbone", default="clip_vit_b32", type=base_backbone)
    sp.add_argument("--loss", choices=["supcon", "triplet"], default="supcon")
    sp.add_argument("--epochs", type=int, default=30)
    sp.add_argument("--lr", type=float, default=1e-3)
    sp.add_argument("--P", type=int, default=32, help="classes per batch")
    sp.add_argument("--K", type=int, default=4, help="images per class per batch")
    sp.add_argument("--temperature", type=float, default=0.1)
    sp.add_argument("--margin", type=float, default=0.2)
    sp.set_defaults(func=cmd_finetune)

    sp = common(sub.add_parser("calibrate"))
    sp.add_argument("--backbone", default="clip_vit_b32", type=base_backbone)
    sp.add_argument("--min_good", type=int, default=3)
    sp.set_defaults(func=cmd_calibrate)

    sp = common(sub.add_parser("search"))
    g = sp.add_mutually_exclusive_group(required=True)
    g.add_argument("--text")
    g.add_argument("--image")
    sp.add_argument("--backbone", default="clip_vit_b32", type=backbone)
    sp.add_argument("--indexer", default="flat", choices=INDEXERS)
    sp.add_argument("--k", type=int, default=10)
    sp.set_defaults(func=cmd_search)

    sp = common(sub.add_parser("expand"))
    sp.add_argument("terms", nargs="*")
    sp.add_argument("--terms_file", help="one term per line")
    sp.add_argument("--per_term", type=int, default=20)
    sp.set_defaults(func=cmd_expand)

    sp = common(sub.add_parser("report"))
    sp.set_defaults(func=cmd_report)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
