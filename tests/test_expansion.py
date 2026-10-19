import numpy as np
from PIL import Image

from imrs import expansion
from imrs.expansion import CoveragePolicy, fetch_term, slug


def test_coverage_policy():
    p = CoveragePolicy(min_score=0.3, min_good=3)
    assert p.is_weak([0.9, 0.8, 0.1])
    assert not p.is_weak([0.31, 0.32, 0.35, 0.0])
    assert p.is_weak([])


def test_slug():
    assert slug("  Red Sports-Car!! ") == "red-sports-car"
    assert slug("???") == "query"


def test_fetch_term_dedupes_and_saves_thumbnails(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    distinct = [Image.fromarray(rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)) for _ in range(3)]
    served = {f"http://x/{i}": img for i, img in enumerate(distinct)}
    served["http://x/dup"] = distinct[0].copy()  # perceptual duplicate of #0
    served["http://x/broken"] = None

    monkeypatch.setattr(expansion, "image_urls", lambda term, n: list(served))
    monkeypatch.setattr(expansion, "download_image", lambda url, session=None: served[url])
    monkeypatch.setattr(expansion.time, "sleep", lambda s: None)

    files = fetch_term(tmp_path, "Red Car", n=10, max_side=128)
    assert len(files) == 3
    assert all(f.parent.name == "dynamic_red-car" for f in files)
    assert max(Image.open(files[0]).size) == 128

    # a second fetch of the same concept doesn't re-add what's already on disk
    assert fetch_term(tmp_path, "Red Car", n=10) == []


def test_verify_with_text_drops_off_topic_downloads(tmp_path):
    from imrs.expansion import verify_with_text

    from .conftest import FakeEmbedder, class_color

    on = tmp_path / "on.jpg"
    off = tmp_path / "off.jpg"
    Image.new("RGB", (32, 32), class_color(3)).save(on)
    Image.new("RGB", (32, 32), (255 - class_color(3)[0], 0, 255)).save(off)
    kept, rejected = verify_with_text(FakeEmbedder(), [on, off], "class3", min_score=0.99)
    assert kept == [on] and rejected == [off]
    assert not off.exists()
