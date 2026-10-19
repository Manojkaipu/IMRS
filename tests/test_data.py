import numpy as np
from PIL import Image

from imrs.data import class_disjoint_split, load_image_set, preprocess_dataset, pretty_class, sample_queries

from .conftest import N_CLASSES, PER_CLASS


def test_load_image_set_skips_clutter_and_dynamic(dataset):
    (dataset / "dynamic_red-car").mkdir()
    Image.new("RGB", (8, 8)).save(dataset / "dynamic_red-car" / "a.jpg")

    base = load_image_set(dataset)
    assert len(base.classes) == N_CLASSES and len(base) == N_CLASSES * PER_CLASS
    assert "257.clutter" not in base.classes

    full = load_image_set(dataset, include_dynamic=True)
    assert full.classes[:N_CLASSES] == base.classes  # base label ids unchanged
    assert full.classes[-1] == "dynamic_red-car" and len(full) == len(base) + 1
    assert all("\\" not in p for p in full.paths)  # portable posix paths


def test_class_disjoint_split():
    tr, te = class_disjoint_split(256, 0.5, seed=0)
    assert len(tr) == len(te) == 128
    assert not set(tr) & set(te) and set(tr) | set(te) == set(range(256))
    np.testing.assert_array_equal(tr, class_disjoint_split(256, 0.5, seed=0)[0])  # deterministic


def test_sample_queries_is_stratified_and_unique():
    labels = np.repeat(np.arange(10), [100] + [10] * 9)  # one dominant class
    q = sample_queries(labels, 50)
    assert len(q) == len(set(q)) == 50
    assert np.bincount(labels[q], minlength=10).max() <= 10  # big class doesn't dominate


def test_preprocess_dedupes_and_writes_manifest(dataset, tmp_path):
    cls = dataset / "001.class0"
    Image.open(cls / "img_000.jpg").save(cls / "copy_of_000.jpg")  # exact duplicate
    out = tmp_path / "proc"
    df = preprocess_dataset(dataset, out, img_size=64)
    assert (out / "manifest.csv").exists()
    assert (df["class_name"] == "001.class0").sum() == PER_CLASS
    assert Image.open(out / df["path"].iloc[0]).size == (64, 64)
    assert len(load_image_set(out)) == N_CLASSES * PER_CLASS


def test_pretty_class():
    assert pretty_class("003.backpack") == "backpack"
    assert pretty_class("dynamic_red-sports-car") == "red sports car"
    assert pretty_class("145.motorbikes-101") == "motorbikes"
