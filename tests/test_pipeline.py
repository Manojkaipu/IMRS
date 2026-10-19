"""End-to-end: embed -> cache -> benchmark -> fine-tune -> report, with the fake backbone."""
import numpy as np
import pandas as pd
import pytest

from imrs import cli
from imrs.data import load_image_set
from imrs.embeddings import EmbeddingCache, embed_files
from imrs.evaluate import BenchConfig, calibrate_coverage, embeddings_for, run_benchmark, run_sweep
from imrs.finetune import FinetuneConfig, finetune

from .conftest import FakeEmbedder


def test_embed_files_keeps_alignment_when_images_are_corrupt(dataset):
    files = sorted((dataset / "001.class0").glob("*.jpg"))[:4]
    bad = dataset / "001.class0" / "broken.jpg"
    bad.write_bytes(b"not an image")
    emb = FakeEmbedder()
    Z, ok = embed_files(emb, files[:2] + [bad] + files[2:], batch_size=2, progress=False)
    assert ok.tolist() == [True, True, False, True, True]
    assert Z.shape == (4, emb.dim)
    np.testing.assert_allclose(Z[2], embed_files(emb, [files[2]], progress=False)[0][0], atol=1e-6)


def test_cache_is_incremental(settings, fake_backbone):
    images = load_image_set(settings.data_root)
    calls = []

    def factory():
        calls.append(1)
        return FakeEmbedder()

    cache = EmbeddingCache(settings.cache_dir, fake_backbone)
    Z1, _ = cache.get(factory, images.root, images.paths, progress=False)
    Z2, _ = cache.get(factory, images.root, images.paths, progress=False)
    assert len(calls) == 1  # second call never loads the model
    np.testing.assert_array_equal(Z1, Z2)
    # order follows the request, not the cache
    Z3, _ = cache.get(factory, images.root, images.paths[::-1], progress=False)
    np.testing.assert_array_equal(Z3, Z1[::-1])


def test_benchmark_end_to_end(settings, fake_backbone):
    cfg = BenchConfig(k=5, n_queries=40, n_latency_queries=5)
    rows = run_benchmark(settings, [fake_backbone], ["flat", "hnsw", "annoy"], cfg)
    flat = rows[0]
    assert flat["mAP@5"] > 0.8  # classes are separable colours
    assert flat["ann_recall@5"] == pytest.approx(1.0)
    assert 0 < flat["R@5"] < 1  # 11 relevant per query, only 5 retrieved
    assert all(r["n_gallery"] == 96 for r in rows)


def test_query_expansion_runs(settings, fake_backbone):
    rows = run_benchmark(settings, [fake_backbone], ["flat"], BenchConfig(k=5, n_queries=20, query_expansion=2))
    assert rows[0]["mAP@5"] > 0.8


def test_sweep(settings, fake_backbone):
    rows = run_sweep(settings, fake_backbone, ["hnsw", "flat"], BenchConfig(k=5, n_queries=20))
    assert {r["indexer"] for r in rows} == {"hnsw"}  # flat has no knob
    assert all(0 <= r["ann_recall@5"] <= 1 for r in rows)


def test_finetune_then_bench_on_heldout(settings, fake_backbone):
    meta = finetune(settings, FinetuneConfig(backbone=fake_backbone, epochs=2, classes_per_batch=2,
                                             samples_per_class=4, hidden=16))
    assert (settings.checkpoint_dir / "proj_fake.pt").exists()
    assert len(meta["history"]) == 2
    rows = run_benchmark(settings, [fake_backbone + "_ft"], ["flat"], BenchConfig(k=5, n_queries=20))
    assert rows[0]["eval_classes"] == "heldout"  # forced: never evaluate on training classes
    assert rows[0]["n_gallery"] == 48


def test_projection_starts_as_identity(settings, fake_backbone):
    from imrs.models import ResidualProjection, apply_projection

    Z, _ = embeddings_for(settings, fake_backbone, load_image_set(settings.data_root), progress=False)
    np.testing.assert_allclose(apply_projection(ResidualProjection(Z.shape[1]), Z), Z, atol=1e-5)


def test_calibrate(settings, fake_backbone):
    summary, rows = calibrate_coverage(settings, fake_backbone, min_good=3, k=5)
    assert len(rows) == 8 and 0 <= summary["roc_auc"] <= 1


def test_cli_bench_and_report(settings, fake_backbone, monkeypatch):
    monkeypatch.setenv("IMRS_DATA_ROOT", str(settings.data_root))
    monkeypatch.setenv("IMRS_RUNS_DIR", str(settings.runs_dir))
    monkeypatch.setenv("IMRS_DEVICE", "cpu")
    cli.main(["bench", "--backbones", "fake", "--indexers", "flat", "ivf_flat", "--queries", "20"])
    df = pd.read_csv(settings.runs_dir / "bench" / "results.csv")
    assert list(df["indexer"]) == ["flat", "ivf_flat"]

    cli.main(["report"])
    report = (settings.runs_dir / "bench" / "REPORT.md").read_text(encoding="utf-8")
    assert "| fake | flat |" in report
    assert (settings.runs_dir / "bench" / "map_by_backbone.png").exists()
