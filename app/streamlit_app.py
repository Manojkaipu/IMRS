"""Streamlit front-end: text/image search over a collection that grows itself from the web.

    streamlit run app/streamlit_app.py

State model: the collection is the image files on disk (base classes + `dynamic_*` folders)
and the embedding cache in `runs/cache`. The in-memory VectorStore is rebuilt from those two
on startup, so there is no separate snapshot to keep in sync.
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from imrs.config import DYNAMIC_PREFIX, Settings
from imrs.data import ImageSet, load_image_set, pretty_class
from imrs.evaluate import embeddings_for
from imrs.expansion import CoveragePolicy, fetch_term, verify_with_text, zero_shot_label
from imrs.index import INDEXERS, VectorStore
from imrs.models import FINETUNED_SUFFIX, available_backbones, load_embedder

st.set_page_config(page_title="IMRS · Self-expanding image search", page_icon="🔎", layout="wide")
settings = Settings()


# ----------------------------------------------------------------------------------------
# Cached resources
# ----------------------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model…")
def get_embedder(name: str, device: str):
    return load_embedder(name, device, settings.checkpoint_dir)


def base_of(embedder):
    return getattr(embedder, "base", embedder)


@st.cache_resource(show_spinner="Indexing collection (first run embeds every image; later runs use the cache)…")
def get_store(backbone: str, data_root: str, device: str) -> VectorStore:
    images = load_image_set(data_root, include_dynamic=True)
    emb = get_embedder(backbone, device)
    Z, images = embeddings_for(settings, backbone, images, base_embedder=base_of(emb), progress=False)
    store = VectorStore(Z.shape[1] if len(Z) else emb.dim)
    store.add(Z, images.paths, [pretty_class(images.classes[l]) for l in images.labels])
    return store


def fetch_verified(term: str, n: int, progress=None) -> tuple[list[Path], int]:
    """Download images for `term`, then drop the ones CLIP says don't match it."""
    files = fetch_term(settings.data_root, term, n=n, progress=progress)
    if verify_min <= 0:
        return files, 0
    kept, rejected = verify_with_text(get_embedder("clip_vit_b32", settings.device), files, term, verify_min)
    return kept, len(rejected)


def add_images(store: VectorStore, backbone: str, files: list[Path]) -> int:
    rel = [f.relative_to(settings.data_root).as_posix() for f in files]
    imgs = ImageSet(settings.data_root, rel, np.zeros(len(rel), dtype=np.int64), ["new"])
    Z, imgs = embeddings_for(settings, backbone, imgs, base_embedder=base_of(get_embedder(backbone, settings.device)),
                             progress=False)
    labels = [pretty_class(Path(p).parent.name) for p in imgs.paths]
    return store.add(Z, imgs.paths, labels)


# ----------------------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")
    data_root = st.text_input("Dataset root", value=str(settings.data_root),
                              help="Preprocessed dataset folder (env: IMRS_DATA_ROOT)")
    settings.data_root = Path(data_root)
    if not settings.data_root.is_dir():
        st.error(f"Dataset root not found: `{data_root}`. Run `imrs preprocess` or set IMRS_DATA_ROOT.")
        st.stop()
    st.caption(f"Device: `{settings.device}`")

    backbones = available_backbones(settings.checkpoint_dir)
    backbone = st.selectbox("Backbone", backbones, index=0,
                            help="CLIP backbones support text queries. `_ft` = fine-tuned adapter.")
    indexer = st.selectbox("Index", INDEXERS, index=0,
                           help="flat = exact; hnsw/ivf/annoy = approximate (faster on large collections)")
    topk = st.slider("Results", 4, 48, 12, step=4)

    st.divider()
    st.subheader("Self-expansion")
    auto_expand = st.toggle("Fetch web images when coverage is weak", value=True)
    min_score = st.slider("Coverage threshold (cosine)", 0.0, 1.0, 0.25, 0.01,
                          help="Text→image CLIP scores are low (≈0.2–0.35). Tune with `imrs calibrate`.")
    min_good = st.number_input("…needed above threshold", 1, 20, 3)
    per_term = st.number_input("Images per expansion", 5, 100, 20)
    verify_min = st.slider("Verify downloads: min CLIP score", 0.0, 0.4, 0.2, 0.01,
                           help="Downloads whose CLIP similarity to the query text is below this are discarded. "
                                "0 disables verification.")
    policy = CoveragePolicy(min_score=min_score, min_good=int(min_good))

    if st.button("Re-sync with disk", help="Re-list files and embed anything new"):
        get_store.clear()

store = get_store(backbone, str(settings.data_root), settings.device)
store.set_indexer(indexer)
embedder = get_embedder(backbone, settings.device)
st.sidebar.metric("Images indexed", f"{len(store):,}")


# ----------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------

def show_results(hits, policy: CoveragePolicy | None = None):
    cols = st.columns(6)
    for i, (path, label, score) in enumerate(hits):
        with cols[i % 6]:
            fp = settings.data_root / path
            good = policy is None or score >= policy.min_score
            badge = "🆕 " if path.startswith(DYNAMIC_PREFIX) else ""
            if fp.exists():
                st.image(str(fp), width="stretch")
            st.caption(f"{badge}**{label}** · {score:.3f}{'' if good else ' ·  weak'}")


def search(q: np.ndarray):
    t0 = time.perf_counter()
    hits = store.search(q, topk)[0]
    return hits, (time.perf_counter() - t0) * 1e3


def expand_and_retry(term: str, q: np.ndarray):
    with st.status(f"Coverage is weak — fetching web images for **{term}**", expanded=True) as status:
        bar = st.progress(0.0)
        files, rejected = fetch_verified(term, int(per_term), progress=lambda f: bar.progress(min(1.0, f)))
        added = add_images(store, backbone, files)
        status.update(label=f"Added {added} new images for '{term}' ({rejected} rejected by CLIP check)",
                      state="complete" if added else "error")
    if added:
        hits, ms = search(q)
        st.markdown(f"**After expansion** · {ms:.1f} ms")
        show_results(hits, policy)


# ----------------------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------------------

tab_search, tab_batch, tab_manage, tab_bench = st.tabs(["🔎 Search", "🌱 Batch expansion", "🗂️ Manage", "📊 Benchmarks"])

with tab_search:
    mode = st.radio("Query type", ["Text → Image", "Image → Image"], horizontal=True,
                    disabled=not embedder.supports_text, index=0 if embedder.supports_text else 1)
    if not embedder.supports_text:
        st.info(f"`{backbone}` has no text encoder; pick a CLIP backbone for text queries.")

    if mode == "Text → Image" and embedder.supports_text:
        text = st.text_input("Describe what you're looking for", placeholder="a red sports car on a racetrack")
        if text:
            q = embedder.embed_text([text])
            hits, ms = search(q)
            st.markdown(f"**Top {len(hits)}** · {ms:.1f} ms · {indexer}")
            show_results(hits, policy)
            if policy.is_weak([s for _, _, s in hits]):
                if auto_expand:
                    expand_and_retry(text, q)
                else:
                    st.warning("Coverage looks weak. Enable self-expansion to fetch web images for this query.")
    else:
        up = st.file_uploader("Upload a query image", type=["jpg", "jpeg", "png", "bmp", "webp"])
        if up:
            img = Image.open(io.BytesIO(up.read())).convert("RGB")
            c1, c2 = st.columns([1, 5])
            c1.image(img, caption="Query", width="stretch")
            q = embedder.embed_pils([img])
            hits, ms = search(q)
            with c2:
                st.markdown(f"**Top {len(hits)}** · {ms:.1f} ms · {indexer}")
                show_results(hits)
            # Image->image cosines run much higher than text->image ones, so the text threshold
            # doesn't transfer; weak = even the best match is far away.
            img_policy = CoveragePolicy(min_score=max(policy.min_score, 0.75), min_good=policy.min_good)
            if img_policy.is_weak([s for _, _, s in hits]) and embedder.supports_text:
                names = sorted({l for l in store.labels})
                term, conf = zero_shot_label(embedder, q[0], names)
                st.caption(f"Zero-shot guess for the query: **{term}** ({conf:.2f})")
                if auto_expand:
                    expand_and_retry(term, q)

with tab_batch:
    st.subheader("Add new concepts in bulk")
    raw = st.text_area("One concept per line", height=140,
                       placeholder="electric scooter\nrubik's cube\ngolden retriever")
    terms = [t.strip() for t in raw.splitlines() if t.strip()]
    if st.button("Fetch & index", disabled=not terms):
        total = 0
        for t in terms:
            with st.spinner(f"Fetching '{t}'…"):
                files, rejected = fetch_verified(t, int(per_term))
                n = add_images(store, backbone, files)
            st.write(f"• **{t}**: {n} new images ({rejected} rejected)")
            total += n
        st.success(f"Added {total} images across {len(terms)} concept(s). Collection size: {len(store):,}")

with tab_manage:
    st.subheader("Web-expansion folders")
    dyn_dirs = sorted(d for d in settings.data_root.glob(f"{DYNAMIC_PREFIX}*") if d.is_dir())
    if not dyn_dirs:
        st.caption("Nothing fetched from the web yet.")
    for d in dyn_dirs:
        files = sorted(d.glob("*.jpg"))
        with st.expander(f"{pretty_class(d.name)} — {len(files)} images"):
            rel = {f.name: f.relative_to(settings.data_root).as_posix() for f in files}
            cols = st.columns(8)
            for i, f in enumerate(files[:48]):
                cols[i % 8].image(str(f), caption=f.name[-14:], width="stretch")
            picked = st.multiselect("Remove images", list(rel), key=f"rm_{d.name}")
            c1, c2 = st.columns(2)
            if c1.button("Remove selected", key=f"btn_{d.name}", disabled=not picked):
                for name in picked:
                    (d / name).unlink(missing_ok=True)
                n = store.remove([rel[name] for name in picked])
                st.success(f"Removed {n} images (index updated, nothing re-embedded).")
                st.rerun()
            if c2.button("Delete whole concept", key=f"del_{d.name}"):
                n = store.remove(list(rel.values()))
                for f in files:
                    f.unlink(missing_ok=True)
                d.rmdir()
                st.success(f"Deleted '{pretty_class(d.name)}' ({n} images).")
                st.rerun()

with tab_bench:
    bench_dir = settings.runs_dir / "bench"
    results = bench_dir / "results.csv"
    if not results.exists():
        st.info("No benchmark results yet. Run `imrs bench` and `imrs report`.")
    else:
        df = pd.read_csv(results)
        st.dataframe(df, width="stretch", hide_index=True)
        for png in sorted(bench_dir.glob("*.png")):
            st.image(str(png), caption=png.stem)

st.caption(f"Backbone `{backbone}`{' (fine-tuned adapter)' if backbone.endswith(FINETUNED_SUFFIX) else ''} · "
           f"index `{indexer}` · data `{settings.data_root}` · cache `{settings.cache_dir}`")
