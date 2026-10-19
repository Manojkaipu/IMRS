import argparse, os, io, csv, time, glob, json, hashlib, re
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

import torch
import requests

from retrieval_framework import BACKBONES, embed_paths, list_images_by_class, build_index, AnnoyIP, ensure_dir
try:
    import faiss
except Exception:
    faiss = None

ImageFile.LOAD_TRUNCATED_IMAGES = True

def slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return re.sub(r"-+", "-", s).strip("-")

def download_image(url: str, timeout=15):
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent":"Mozilla/5.0"})
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        return None

def save_small(img: Image.Image, out_path: str, max_side: int = 256, quality: int = 85):
    w, h = img.size
    scale = max_side / max(w, h)
    nw, nh = max(1, int(round(w*scale))), max(1, int(round(h*scale)))
    img = img.resize((nw, nh), Image.BICUBIC)
    img.save(out_path, format="JPEG", quality=quality, optimize=True, progressive=True)

def perceptual_hash(img: Image.Image):
    try:
        from imagehash import phash
        return phash(img)
    except Exception:
        return None

def already_seen(ph, seen_list, th=4):
    if ph is None: return False
    return any((ph - x) <= th for x in seen_list)

def ddg_image_urls(query: str, n: int):
    try:
        from duckduckgo_search import DDGS
    except Exception as e:
        raise ImportError("Please install duckduckgo-search: pip install duckduckgo-search") from e
    urls=[]
    with DDGS() as ddgs:
        for r in ddgs.images(keywords=query, max_results=n, safesearch="off"):
            u = r.get("image") or r.get("thumbnail")
            if u: urls.append(u)
            if len(urls) >= n: break
    return urls

def make_index(indexer: str, d: int):
    indexer = indexer.lower()
    if indexer == "annoy": return AnnoyIP(d, n_trees=50)
    if faiss is None: raise ImportError("pip install faiss-cpu")
    if indexer in ("flat_ip","flat"): return faiss.IndexFlatIP(d)
    if indexer == "hnsw":
        idx = faiss.IndexHNSWFlat(d, 32); idx.hnsw.efSearch = 64; return idx
    raise ValueError("indexer must be one of: flat_ip, hnsw, annoy")

def embed_existing(data_root: str, backbone: str, device: str, cache_npz: str):
    paths, _, _ = list_images_by_class(data_root)
    print(f"[embed-existing] images={len(paths)} backbone={backbone}")
    emb = BACKBONES[backbone](device)
    X = embed_paths(emb, paths, batch=64, device=device)
    np.savez_compressed(cache_npz, embeddings=X, paths=np.array(paths, dtype=object))
    print(f"[embed-existing] saved -> {cache_npz}")
    return paths, X

def build_or_update_index(indexer: str, X: np.ndarray):
    d = X.shape[1]
    idx = make_index(indexer, d)
    idx.add(X.astype("float32"))
    return idx

def batch_retrieve(indexer_obj, Xdb: np.ndarray, Q: np.ndarray, k: int = 12):
    sims, idxs = indexer_obj.search(Q.astype("float32"), k)
    return sims, idxs

def expand_from_web(data_root: str, term: str, per_term: int, max_side: int, quality: int, phash_th: int = 4):
    batch_dir = os.path.join(data_root, "_dynamic_batches", slug(term))
    ensure_dir(batch_dir)
    seen=[]
    existing = glob.glob(os.path.join(batch_dir, "*.jpg"))
    for p in existing:
        try:
            h = perceptual_hash(Image.open(p).convert("RGB"))
            if h: seen.append(h)
        except Exception: pass
    urls = ddg_image_urls(term, per_term*3)
    saved=[]
    for u in tqdm(urls, desc=f"dl:{term}"):
        if len(saved) >= per_term: break
        im = download_image(u)
        if im is None: continue
        h = perceptual_hash(im)
        if already_seen(h, seen, th=phash_th): continue
        fname = slug(term)+"-"+hashlib.md5(u.encode()).hexdigest()[:10]+".jpg"
        outp = os.path.join(batch_dir, fname)
        save_small(im, outp, max_side=max_side, quality=quality)
        saved.append(outp)
        if h: seen.append(h)
    print(f"[expand] term='{term}' saved={len(saved)} dir={batch_dir}")
    return saved

def embed_new(paths: List[str], backbone: str, device: str):
    emb = BACKBONES[backbone](device)
    from PIL import Image
    ims=[]
    for p in paths:
        try: ims.append(emb.preprocess(Image.open(p).convert("RGB")))
        except Exception: pass
    if not ims: return np.zeros((0, emb.out_dim), dtype="float32")
    with torch.no_grad():
        x = torch.stack(ims).to(device)
        z = emb(x).cpu().numpy().astype("float32")
    return z

def main():
    ap = argparse.ArgumentParser("Batch expansion for dynamic image retrieval")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p0 = sub.add_parser("embed-existing")
    p0.add_argument("--data_root", required=True)
    p0.add_argument("--backbone", default="clip_vit_b32", choices=list(BACKBONES.keys()))
    p0.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p0.add_argument("--cache_npz", default="runs/batch/cache_existing.npz")

    p1 = sub.add_parser("retrieve")
    p1.add_argument("--data_root", required=True)
    p1.add_argument("--backbone", default="clip_vit_b32", choices=list(BACKBONES.keys()))
    p1.add_argument("--indexer", default="flat_ip", choices=["flat_ip","hnsw","annoy"])
    p1.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p1.add_argument("--cache_npz", default="runs/batch/cache_existing.npz")
    p1.add_argument("--query_image")
    p1.add_argument("--query_text")
    p1.add_argument("--topk", type=int, default=12)

    p2 = sub.add_parser("expand-from-web")
    p2.add_argument("--data_root", required=True)
    p2.add_argument("--terms_csv", required=True)
    p2.add_argument("--per_term", type=int, default=20)
    p2.add_argument("--max_side", type=int, default=256)
    p2.add_argument("--quality", type=int, default=85)
    p2.add_argument("--phash_threshold", type=int, default=4)
    p2.add_argument("--backbone", default="clip_vit_b32", choices=list(BACKBONES.keys()))
    p2.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p2.add_argument("--cache_npz", default="runs/batch/cache_existing.npz")

    args = ap.parse_args()

    if args.cmd == "embed-existing":
        os.makedirs(os.path.dirname(args.cache_npz), exist_ok=True)
        embed_existing(args.data_root, args.backbone, args.device, args.cache_npz)

    elif args.cmd == "retrieve":
        if not os.path.exists(args.cache_npz):
            raise FileNotFoundError(f"Cache not found: {args.cache_npz}. Run embed-existing first.")
        cache = np.load(args.cache_npz, allow_pickle=True)
        X = cache["embeddings"]; paths = cache["paths"].tolist()
        emb = BACKBONES[args.backbone](args.device)
        idx = build_or_update_index(args.indexer, X)
        if args.query_image:
            from PIL import Image
            im = Image.open(args.query_image).convert("RGB")
            with torch.no_grad():
                q = emb.preprocess(im).unsqueeze(0).to(args.device)
                qz = emb(q).cpu().numpy().astype("float32")
            sims, idxs = idx.search(qz, args.topk)
        elif args.query_text:
            if not hasattr(emb, "encode_text"): raise ValueError("query_text requires CLIP backbone.")
            with torch.no_grad():
                qz = emb.encode_text([args.query_text]).cpu().numpy().astype("float32")
            sims, idxs = idx.search(qz, args.topk)
        else:
            raise ValueError("Provide --query_image or --query_text")
        print("\nTop results:")
        for rank, (i, s) in enumerate(zip(idxs[0], sims[0]), start=1):
            print(f"{rank:>2}. {paths[int(i)]}  score={float(s):.3f}")

    elif args.cmd == "expand-from-web":
        os.makedirs(os.path.dirname(args.cache_npz), exist_ok=True)
        if os.path.exists(args.cache_npz):
            cache = np.load(args.cache_npz, allow_pickle=True)
            X = cache["embeddings"]; paths = cache["paths"].tolist()
        else:
            paths, X0 = embed_existing(args.data_root, args.backbone, args.device, args.cache_npz)
            X = X0
        terms=[]
        with open(args.terms_csv, newline="", encoding="utf-8") as f:
            rdr = csv.reader(f)
            for row in rdr:
                if not row: continue
                if row[0].strip().lower()=="term": continue
                terms.append(row[0].strip())
        print(f"[expand] terms={len(terms)} per_term={args.per_term}")
        new_paths=[]
        for t in terms:
            new_paths += expand_from_web(args.data_root, t, args.per_term, args.max_side, args.quality, args.phash_threshold)
        if new_paths:
            Z = embed_new(new_paths, args.backbone, args.device)
            X = np.concatenate([X,Z], axis=0); paths.extend(new_paths)
            np.savez_compressed(args.cache_npz, embeddings=X, paths=np.array(paths, dtype=object))
            print(f"[expand] updated cache -> {args.cache_npz} | total_db={len(paths)}")
        else:
            print("[expand] no new images downloaded; cache unchanged.")

if __name__ == "__main__":
    main()
