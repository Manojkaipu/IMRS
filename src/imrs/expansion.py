"""Dynamic dataset expansion: when the index has no good match for a query, fetch a small
batch of web images for it, de-duplicate them, and add them to the collection.

Images come from Wikimedia Commons (official API, freely licensed), topped up from DuckDuckGo
image search when Commons has too few. Requests use an honest User-Agent and polite delays.
"""
from __future__ import annotations

import hashlib
import io
import logging
import os
import random
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .config import DYNAMIC_PREFIX

log = logging.getLogger(__name__)

# Wikimedia asks API clients to identify themselves; set IMRS_USER_AGENT to include your contact URL/email.
USER_AGENT = os.environ.get("IMRS_USER_AGENT", "IMRS/1.0 (open-source image retrieval demo)")
COMMONS_API = "https://commons.wikimedia.org/w/api.php"


@dataclass
class CoveragePolicy:
    """When is the collection 'missing' a concept?  Too few results above a cosine threshold.

    The threshold is backbone-specific: CLIP text->image cosines sit around 0.2-0.35, while
    image->image cosines are much higher. Calibrate it with `imrs calibrate`.
    """

    min_score: float = 0.25
    min_good: int = 3

    def is_weak(self, scores: Sequence[float]) -> bool:
        return sum(s >= self.min_score for s in scores) < self.min_good


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.strip().lower()).strip("-") or "query"


def dynamic_folder(data_root: Path, term: str) -> Path:
    return Path(data_root) / f"{DYNAMIC_PREFIX}{slug(term)}"


# ----------------------------------------------------------------------------------------
# URL sources
# ----------------------------------------------------------------------------------------

def _ddgs():
    try:
        from ddgs import DDGS  # current package name
    except ImportError:
        from duckduckgo_search import DDGS  # older name
    return DDGS


def ddg_image_urls(query: str, n: int, max_tries: int = 2) -> list[str]:
    DDGS = _ddgs()
    for attempt in range(max_tries):
        try:
            with DDGS() as ddgs:
                results = ddgs.images(query, max_results=n * 2, safesearch="moderate")
                urls = [r.get("image") or r.get("thumbnail") for r in results]
                return [u for u in urls if u][:n]
        except Exception as e:  # rate limits surface as generic exceptions
            log.info("ddg attempt %d failed: %s", attempt + 1, e)
            time.sleep(2 ** attempt + random.random())
    return []


_FILLER = re.compile(r"^(an? |the )?((photo|picture|image|close-up) of (an? |the )?)?", re.I)


def search_query(term: str) -> str:
    """'a photo of a golden retriever' -> 'golden retriever' (search engines match keywords)."""
    return _FILLER.sub("", term.strip()).strip() or term


def commons_image_urls(term: str, n: int, session=None) -> list[str]:
    """Wikimedia Commons file search: an official API, and every result is freely licensed."""
    import requests

    s = session or requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    try:
        js = s.get(COMMONS_API, params={
            "action": "query", "format": "json", "generator": "search", "gsrnamespace": 6,
            "gsrsearch": f"{search_query(term)} filetype:bitmap", "gsrlimit": min(50, n * 2),
            "prop": "imageinfo", "iiprop": "url|mime", "iiurlwidth": 512,
        }, timeout=20).json()
    except Exception as e:
        log.info("commons lookup failed for %r: %s", term, e)
        return []
    pages = sorted(js.get("query", {}).get("pages", {}).values(), key=lambda p: p.get("index", 0))
    urls = [ii.get("thumburl") or ii.get("url") for p in pages for ii in p.get("imageinfo", [])
            if ii.get("mime") in ("image/jpeg", "image/png", "image/webp")]
    return [u for u in urls if u][:n]


_PREPOSITION = re.compile(r"\s+(on|in|at|with|near|under|over|by|of|from)\s+", re.I)


def image_urls(term: str, n: int) -> list[str]:
    urls = commons_image_urls(term, n)
    head = _PREPOSITION.split(search_query(term))[0]
    if len(urls) < n and head != search_query(term):
        # "red sports car on a racetrack" -> "red sports car": fewer AND-ed keywords, more hits
        urls += commons_image_urls(head, n - len(urls))
    if len(urls) < n:
        try:
            urls += ddg_image_urls(search_query(term), n - len(urls))
        except ImportError:
            log.info("ddgs / duckduckgo-search not installed; using Wikimedia Commons only")
    return list(dict.fromkeys(urls))[:n]


# ----------------------------------------------------------------------------------------
# Download + de-duplicate + save
# ----------------------------------------------------------------------------------------

def download_image(url: str, session=None, timeout: int = 15, min_side: int = 64) -> Image.Image | None:
    import requests

    s = session or requests
    try:
        r = s.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        return img if min(img.size) >= min_side else None
    except Exception:
        return None


def save_thumbnail(img: Image.Image, out_path: Path, max_side: int = 256, quality: int = 85) -> None:
    img = img.copy()
    img.thumbnail((max_side, max_side), Image.BICUBIC)
    img.save(out_path, format="JPEG", quality=quality, optimize=True)


def fetch_term(data_root: Path, term: str, n: int = 20, max_side: int = 256, quality: int = 85,
               phash_threshold: int = 4, progress: Callable[[float], None] | None = None) -> list[Path]:
    """Download up to `n` new, non-duplicate images for `term` into `dynamic_<slug>/`."""
    import requests
    from imagehash import phash

    out_dir = dynamic_folder(data_root, term)
    out_dir.mkdir(parents=True, exist_ok=True)
    seen = []
    for p in out_dir.glob("*.jpg"):
        try:
            seen.append(phash(Image.open(p)))
        except Exception:
            pass

    urls = image_urls(term, n * 3)
    saved: list[Path] = []
    with requests.Session() as session:
        for i, url in enumerate(urls):
            if progress:
                progress(i / max(1, len(urls)))
            if len(saved) >= n:
                break
            img = download_image(url, session)
            if img is None:
                continue
            h = phash(img)
            if any(h - o <= phash_threshold for o in seen):
                continue
            seen.append(h)
            out = out_dir / f"{slug(term)}-{hashlib.md5(url.encode()).hexdigest()[:10]}.jpg"
            save_thumbnail(img, out, max_side, quality)
            saved.append(out)
            time.sleep(random.uniform(0.1, 0.3))
    if progress:
        progress(1.0)
    log.info("expansion %r: %d urls -> %d new images in %s", term, len(urls), len(saved), out_dir)
    return saved


def zero_shot_label(embedder, query_vec: np.ndarray, class_names: Sequence[str]) -> tuple[str, float]:
    """Best-matching class name for an image embedding via CLIP text prompts."""
    prompts = [f"a photo of a {c}" for c in class_names]
    T = embedder.embed_text(prompts)
    scores = T @ np.asarray(query_vec).reshape(-1)
    best = int(np.argmax(scores))
    return class_names[best], float(scores[best])


def verify_with_text(embedder, files: Sequence[Path], term: str, min_score: float) -> tuple[list[Path], list[Path]]:
    """Drop downloads whose CLIP image-text similarity to `term` is below `min_score`.

    Search engines return logos, diagrams and off-topic pages; this keeps label noise out of
    the index. Returns (kept, rejected); rejected files are deleted from disk.
    """
    if not files or not getattr(embedder, "supports_text", False):
        return list(files), []
    t = embedder.embed_text([f"a photo of {term}"])[0]
    imgs = [Image.open(f).convert("RGB") for f in files]
    scores = embedder.embed_pils(imgs) @ t
    kept = [f for f, s in zip(files, scores) if s >= min_score]
    rejected = [f for f, s in zip(files, scores) if s < min_score]
    for f in rejected:
        f.unlink(missing_ok=True)
    log.info("verify %r: kept %d / %d (min_score=%.2f)", term, len(kept), len(files), min_score)
    return kept, rejected
