import os, io, json, time, glob, pickle, hashlib, re, random
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from PIL import Image, ImageFile

import streamlit as st

import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as T

ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    import faiss
except Exception:
    faiss = None
try:
    from annoy import AnnoyIndex
except Exception:
    AnnoyIndex = None
try:
    from duckduckgo_search import DDGS
except Exception:
    DDGS = None
try:
    import requests
except Exception:
    requests = None
try:
    from imagehash import phash as _phash
except Exception:
    _phash = None

DEFAULT_DATA_ROOT = r"C:/Users/punna/Downloads/IMRS/256_ObjectCategories_proc"
CP2_NPZ_DIR = r"C:/Users/punna/Downloads/IMRS/runs/ckpt2/cache"
APP_STATE_DIR = "runs/streamlit_state"
os.makedirs(APP_STATE_DIR, exist_ok=True)

SIM_THRESHOLD = 0.22
MIN_GOOD = 3
AUTO_PER_TERM = 20
AUTO_MAX_SIDE = 256
AUTO_JPEG_QUALITY = 85
PHASH_TH = 4

DDG_MAX_TRIES = 5
DDG_BASE_SLEEP = 1.0
DDG_JITTER = (0.25, 0.75)
HTTP_PER_DOWNLOAD_SLEEP = (0.1, 0.35)

UA_LIST = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_3) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
]

def _coverage_is_weak(scores, sim_thr=SIM_THRESHOLD, min_good=MIN_GOOD):
    scores = list(scores) if scores is not None else []
    good = sum(1 for s in scores if s >= sim_thr)
    top = max(scores) if scores else 0.0
    return (top < sim_thr) or (good < min_good)

def cache_tag(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_-]+','_', name)

def backbone_for_dim(dim: int) -> str | None:
    return {512: "CLIP ViT-B/32", 2048: "ResNet-50", 768: "ViT-B/16 (torchvision)"} .get(dim, None)

def backbone_tag(name: str) -> str:
    if "CLIP" in name: return "clip_vit_b32"
    if "ResNet-50" in name: return "resnet50"
    if "ViT-B/16" in name: return "vit_b16"
    return cache_tag(name)

def slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return re.sub(r"-+", "-", s).strip("-")

# --------- ID helpers (signed int64, with coercion) ----------
def id_from_path(p: str) -> int:
    norm = os.path.normpath(p).replace("\\", "/")
    h = hashlib.sha1(norm.encode("utf-8")).digest()[:8]
    return int.from_bytes(h, "little", signed=True)

def coerce_ids_int64(ids_list):
    try:
        return np.array(ids_list, dtype=np.int64)
    except OverflowError:
        fixed = [int(((int(x) + (1 << 63)) % (1 << 64)) - (1 << 63)) for x in ids_list]
        return np.array(fixed, dtype=np.int64)

# ------------------------------------------------------------

def list_images_by_class(data_root: str) -> Tuple[List[str], List[int], List[str]]:
    man = os.path.join(data_root, "manifest.csv")
    if os.path.exists(man):
        df = pd.read_csv(man)
        classes = sorted(df["class_name"].unique())
        c2i = {c: i for i, c in enumerate(classes)}
        paths = df["path"].tolist()
        labels = [c2i[c] for c in df["class_name"].tolist()]
        paths = [p for p in paths if isinstance(p, str) and os.path.exists(p)]
        labels = labels[:len(paths)]
        return paths, labels, classes
    classes = sorted([
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
        and not d.startswith("dynamic_")
    ])
    class_to_id = {c: i for i, c in enumerate(classes)}
    paths, labels = [], []
    for c in classes:
        folder = os.path.join(data_root, c)
        for p in glob.glob(os.path.join(folder, "*")):
            if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")):
                if os.path.exists(p):
                    paths.append(p); labels.append(class_to_id[c])
    return paths, labels, classes

def load_dynamic_set(data_root: str) -> List[str]:
    dyn_roots = sorted([
        os.path.join(data_root, d) for d in os.listdir(data_root)
        if d.startswith("dynamic_") and os.path.isdir(os.path.join(data_root, d))
    ])
    out = []
    for dr in dyn_roots:
        out.extend(sorted(
            p for p in glob.glob(os.path.join(dr, "*"))
            if p.lower().endswith((".jpg",".jpeg",".png",".bmp",".gif",".webp")) and os.path.exists(p)
        ))
    return out

class Embedder(nn.Module):
    def __init__(self, dim: int):
        super().__init__(); self.out_dim = dim
    def forward(self, img_batch: torch.Tensor) -> torch.Tensor: raise NotImplementedError
    @property
    def preprocess(self): raise NotImplementedError

class CLIPViTB32(Embedder):
    def __init__(self, device):
        super().__init__(dim=512)
        import clip
        self.device = device
        self.clip, self.prep = clip.load("ViT-B/32", device=device)
    @torch.no_grad()
    def forward(self, img_batch):
        z = self.clip.encode_image(img_batch)
        return z / z.norm(dim=-1, keepdim=True)
    @property
    def preprocess(self): return self.prep
    @torch.no_grad()
    def encode_text(self, texts: List[str]):
        import clip
        toks = clip.tokenize(texts).to(self.device)
        z = self.clip.encode_text(toks)
        return z / z.norm(dim=-1, keepdim=True)

class ResNet50GAP(Embedder):
    def __init__(self, device):
        super().__init__(dim=2048)
        self.backbone = tv.models.resnet50(weights=tv.models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone.fc = nn.Identity()
        self.backbone.to(device).eval()
        self._prep = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
    @torch.no_grad()
    def forward(self, img_batch):
        z = self.backbone(img_batch)
        return z / z.norm(dim=-1, keepdim=True)
    @property
    def preprocess(self): return self._prep

class ViT_B16_TorchVision(Embedder):
    def __init__(self, device):
        super().__init__(dim=768)
        weights = tv.models.ViT_B_16_Weights.DEFAULT
        self.model = tv.models.vit_b_16(weights=weights)
        self.model.heads = nn.Identity()
        self.model.to(device).eval()
        self._prep = weights.transforms()
    @torch.no_grad()
    def forward(self, img_batch):
        z = self.model(img_batch)
        return z / z.norm(dim=-1, keepdim=True)
    @property
    def preprocess(self): return self._prep

BACKBONES = {
    "CLIP ViT-B/32": CLIPViTB32,
    "ResNet-50": ResNet50GAP,
    "ViT-B/16 (torchvision)": ViT_B16_TorchVision,
}

class IndexManager:
    def __init__(self, indexer: str, dim: int, state_dir: str, hnsw_m=32, ef=64, annoy_trees=50):
        self.indexer=indexer; self.dim=dim; self.state_dir=state_dir
        os.makedirs(self.state_dir, exist_ok=True)
        self.hnsw_m=hnsw_m; self.ef=ef; self.annoy_trees=annoy_trees
        self.backend=None; self.idx=None
        self._annoy_ids: List[int] = []

    def fit_with_ids(self, X: np.ndarray, ids: np.ndarray):
        if self.indexer.startswith("FAISS"):
            if faiss is None: raise ImportError("Install faiss-cpu")
            if self.indexer=="FAISS Flat":
                base = faiss.IndexFlatIP(self.dim)
            elif self.indexer=="FAISS HNSW":
                base = faiss.IndexHNSWFlat(self.dim, self.hnsw_m)
                base.hnsw.efSearch = self.ef
            else:
                raise ValueError("Unknown FAISS indexer")
            idx = faiss.IndexIDMap2(base)
            idx.add_with_ids(X.astype("float32"), ids.astype(np.int64))
            self.idx = idx; self.backend="faiss"; self._annoy_ids=[]
        elif self.indexer=="Annoy":
            if AnnoyIndex is None: raise ImportError("Install annoy")
            self.idx = AnnoyIndex(self.dim, metric='angular')
            self._annoy_ids = ids.astype(np.int64).tolist()
            for i, vec in enumerate(X.astype("float32")):
                self.idx.add_item(i, vec.tolist())
            self.idx.build(self.annoy_trees)
            self.backend="annoy"
        else:
            raise ValueError("Unknown indexer")

    def rebuild(self, X: np.ndarray, ids: np.ndarray):
        self.fit_with_ids(X, ids)

    def search(self, Q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.backend=="faiss":
            return self.idx.search(Q.astype("float32"), k)
        elif self.backend=="annoy":
            sims, id_hits = [], []
            for q in Q.astype("float32"):
                ii, dd = self.idx.get_nns_by_vector(q.tolist(), int(k), include_distances=True)
                ss = [1 - (d**2)/2 for d in dd]
                sims.append(ss)
                id_hits.append([self._annoy_ids[i] if 0 <= i < len(self._annoy_ids) else -1 for i in ii])
            return np.array(sims, np.float32), np.array(id_hits, np.int64)
        else:
            raise ValueError("Index not initialized")

    def save(self):
        meta = {"indexer":self.indexer,"dim":self.dim,"backend":self.backend,
                "hnsw_m":self.hnsw_m,"ef":self.ef,"annoy_trees":self.annoy_trees}
        with open(os.path.join(self.state_dir,"index_meta.json"),"w") as f: json.dump(meta,f,indent=2)
        if self.backend=="faiss":
            faiss.write_index(self.idx, os.path.join(self.state_dir,"index.faiss"))
        elif self.backend=="annoy":
            self.idx.save(os.path.join(self.state_dir,"index.ann"))
            with open(os.path.join(self.state_dir,"annoy_ids.pkl"),"wb") as f: pickle.dump(self._annoy_ids,f)

    def load(self) -> bool:
        meta_path = os.path.join(self.state_dir,"index_meta.json")
        if not os.path.exists(meta_path): return False
        meta = json.load(open(meta_path))
        self.indexer=meta["indexer"]; self.dim=meta["dim"]; self.backend=meta["backend"]
        self.hnsw_m=meta.get("hnsw_m",32); self.ef=meta.get("ef",64); self.annoy_trees=meta.get("annoy_trees",50)
        if self.backend=="faiss":
            if faiss is None: return False
            self.idx = faiss.read_index(os.path.join(self.state_dir,"index.faiss"))
            return True
        elif self.backend=="annoy":
            if AnnoyIndex is None: return False
            self.idx = AnnoyIndex(self.dim, metric='angular')
            self.idx.load(os.path.join(self.state_dir,"index.ann"))
            self._annoy_ids = pickle.load(open(os.path.join(self.state_dir,"annoy_ids.pkl"),"rb"))
            return True
        return False

@torch.no_grad()
def embed_images(embedder: 'Embedder', paths: List[str], device: str, batch: int = 64) -> np.ndarray:
    out = []
    for i in range(0, len(paths), batch):
        xs=[]
        for p in paths[i:i+batch]:
            try:
                img = Image.open(p).convert("RGB")
                xs.append(embedder.preprocess(img))
            except Exception:
                continue
        if not xs: continue
        x = torch.stack(xs).to(device)
        z = embedder(x).cpu().numpy().astype("float32")
        out.append(z)
    if not out: return np.zeros((0, embedder.out_dim), dtype="float32")
    return np.vstack(out)

@torch.no_grad()
def embed_pils(embedder: 'Embedder', imgs: List[Image.Image], device: str) -> np.ndarray:
    if not imgs: return np.zeros((0, embedder.out_dim), dtype="float32")
    x = torch.stack([embedder.preprocess(im.convert("RGB")) for im in imgs]).to(device)
    z = embedder(x).cpu().numpy().astype("float32")
    return z

def _sleep_with_jitter(base: float, attempt: int):
    delay = base * (2 ** (attempt - 1)) + random.uniform(*DDG_JITTER)
    time.sleep(delay)

def ddg_image_urls(query: str, n: int) -> List[str]:
    if DDGS is None: raise ImportError("Install duckduckgo-search")
    urls = []
    err = None
    for attempt in range(1, DDG_MAX_TRIES + 1):
        try:
            with DDGS() as ddgs:
                for r in ddgs.images(keywords=query, max_results=n*2, safesearch="off"):
                    u = r.get("image") or r.get("thumbnail")
                    if u:
                        urls.append(u)
                        if len(urls) >= n:
                            return urls[:n]
            if len(urls) >= n:
                return urls[:n]
        except Exception as e:
            err = e
            _sleep_with_jitter(DDG_BASE_SLEEP, attempt)
            continue
        if len(urls) < n:
            _sleep_with_jitter(DDG_BASE_SLEEP, attempt)
    if urls:
        return urls[:n]
    if err:
        raise err
    return urls

def wiki_search_titles(term: str, limit: int = 5) -> List[str]:
    if requests is None: return []
    try:
        headers = {"User-Agent": random.choice(UA_LIST)}
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action":"opensearch","search":term,"limit":limit,"namespace":0,"format":"json"},
            headers=headers, timeout=15
        )
        r.raise_for_status()
        js = r.json()
        return js[1] if isinstance(js, list) and len(js) >= 2 else []
    except Exception:
        return []

def wiki_images_for_title(title: str, max_n: int = 10) -> List[str]:
    if requests is None: return []
    urls = []
    try:
        headers = {"User-Agent": random.choice(UA_LIST)}
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action":"query","titles":title,"prop":"pageimages|images","pithumbsize":512,"format":"json"},
            headers=headers, timeout=15
        )
        r.raise_for_status()
        js = r.json()
        pages = js.get("query", {}).get("pages", {})
        for _, page in pages.items():
            thumb = page.get("thumbnail", {}).get("source")
            if thumb: urls.append(thumb)
            imgs = page.get("images", []) or []
            file_titles = [im.get("title") for im in imgs if im.get("title","").lower().startswith("file:")]
            for chunk_i in range(0, len(file_titles), 20):
                chunk = file_titles[chunk_i:chunk_i+20]
                r2 = requests.get(
                    "https://en.wikipedia.org/w/api.php",
                    params={"action":"query","titles":"|".join(chunk),"prop":"imageinfo","iiprop":"url","format":"json"},
                    headers=headers, timeout=15
                )
                r2.raise_for_status()
                js2 = r2.json()
                for _, p2 in js2.get("query", {}).get("pages", {}).items():
                    ii = p2.get("imageinfo", [])
                    if ii:
                        u = ii[0].get("url")
                        if u: urls.append(u)
            if len(urls) >= max_n: break
    except Exception:
        return urls[:max_n]
    return urls[:max_n]

def wikipedia_image_urls(term: str, n: int) -> List[str]:
    results = []
    titles = wiki_search_titles(term, limit=6)
    for t in titles:
        results += wiki_images_for_title(t, max_n=max(4, n))
        if len(results) >= n:
            break
    return results[:n]

def get_image_urls(term: str, n: int) -> List[str]:
    try:
        urls = ddg_image_urls(term, n)
        if len(urls) < n:
            more = wikipedia_image_urls(term, n - len(urls))
            urls += more
        if not urls:
            urls = wikipedia_image_urls(term, n)
        return urls[:n]
    except Exception:
        urls = wikipedia_image_urls(term, n)
        return urls[:n]

def download_image(url: str, timeout=15, max_tries=3):
    if requests is None: raise ImportError("Install requests")
    for attempt in range(1, max_tries+1):
        try:
            headers = {"User-Agent": random.choice(UA_LIST), "Referer": "https://www.google.com/"}
            r = requests.get(url, timeout=timeout, headers=headers)
            r.raise_for_status()
            time.sleep(random.uniform(*HTTP_PER_DOWNLOAD_SLEEP))
            return Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception:
            time.sleep(0.25 * attempt + random.uniform(0.05, 0.25))
            continue
    return None

def save_small(img: Image.Image, out_path: str, max_side: int = 256, quality: int = 85):
    w, h = img.size
    scale = max_side / max(w, h)
    nw, nh = max(1, int(round(w*scale))), max(1, int(round(h*scale)))
    img = img.resize((nw, nh), Image.BICUBIC)
    img.save(out_path, format="JPEG", quality=quality, optimize=True, progressive=True)

def p_hash(img: Image.Image):
    if _phash is None: return None
    try: return _phash(img)
    except Exception: return None

def ensure_base_state(data_root: str):
    st.session_state.setdefault("paths", None)
    st.session_state.setdefault("labels", None)
    st.session_state.setdefault("classes", None)
    st.session_state.setdefault("embedder_name", None)
    st.session_state.setdefault("embedder", None)
    st.session_state.setdefault("indexer", None)
    st.session_state.setdefault("embeddings", None)
    st.session_state.setdefault("ids", None)
    st.session_state.setdefault("catalog", {})
    st.session_state.setdefault("manager", None)

    if st.session_state.paths is None:
        paths, labels, classes = list_images_by_class(data_root)
        dyn_paths = load_dynamic_set(data_root)
        paths = paths + dyn_paths
        labels = (labels[:len(paths)] + [-1]*max(0, len(paths)-len(labels)))
        st.session_state.paths = paths
        st.session_state.labels = labels
        st.session_state.classes = classes

def ensure_embedder(backbone_name: str, device: str):
    if (st.session_state.embedder is None) or (st.session_state.embedder_name != backbone_name):
        st.session_state.embedder_name = backbone_name
        st.session_state.embedder = BACKBONES[backbone_name](device)
        st.session_state.manager = None

def dynamic_folder_for_term(data_root: str, term: str) -> str:
    return os.path.join(data_root, f"dynamic_{slug(term)}")

def expand_to_dynamic_folder(data_root: str, term: str, per_term: int, max_side: int, quality: int, phash_threshold: int = 4) -> List[str]:
    dyn_dir = dynamic_folder_for_term(data_root, term)
    os.makedirs(dyn_dir, exist_ok=True)
    seen=[]
    for p in glob.glob(os.path.join(dyn_dir, "*.jpg")):
        try:
            h = p_hash(Image.open(p).convert("RGB"))
            if h: seen.append(h)
        except Exception: pass
    urls = get_image_urls(term, per_term*3)
    saved=[]
    prog = st.progress(0, text=f"Downloading to {os.path.basename(dyn_dir)}")
    for i, u in enumerate(urls):
        if len(saved) >= per_term: break
        im = download_image(u)
        if im is None:
            prog.progress(min(100, int((i+1)/max(1,len(urls))*100))); continue
        h = p_hash(im)
        if h and any((h - x) <= phash_threshold for x in seen):
            prog.progress(min(100, int((i+1)/max(1,len(urls))*100))); continue
        fname = slug(term)+"-"+hashlib.md5(u.encode()).hexdigest()[:10]+".jpg"
        outp = os.path.join(dyn_dir, fname)
        save_small(im, outp, max_side=max_side, quality=quality)
        saved.append(outp)
        if h: seen.append(h)
        prog.progress(min(100, int((i+1)/max(1,len(urls))*100)))
    prog.empty()
    return saved

def persist_dynamic_embeddings(term: str, Z: np.ndarray, paths: List[str], embedder_name: str):
    tag = slug(term)
    outp = os.path.join(APP_STATE_DIR, f"dyn_emb_{tag}_{cache_tag(embedder_name)}.npz")
    np.savez_compressed(outp, embeddings=Z, paths=np.array(paths, dtype=object))
    return outp

def scan_and_merge_npzs(backbone_name: str, device: str) -> dict:
    dim = BACKBONES[backbone_name](device).out_dim
    tag = backbone_tag(backbone_name)
    merged_from = {"base": None, "dyn_added": [], "skipped_no_path": 0, "dedup_dropped": 0}

    vecs = []
    paths = []
    seen_ids = set()

    if os.path.isdir(CP2_NPZ_DIR):
        candidates = sorted(glob.glob(os.path.join(CP2_NPZ_DIR, "emb_*.npz")))
        preferred = [p for p in candidates if tag in os.path.basename(p)]
        chosen = preferred[0] if preferred else None
        if not chosen and candidates:
            for p in candidates:
                try:
                    if np.load(p)["embeddings"].shape[1] == dim:
                        chosen = p; break
                except Exception:
                    continue
        if chosen:
            try:
                data = np.load(chosen, allow_pickle=True)
                Z = data["embeddings"]
                P = data["paths"].tolist() if "paths" in data.files else None
                if P is None:
                    merged_from["skipped_no_path"] += len(Z)
                else:
                    for z, pth in zip(Z, P):
                        if isinstance(pth, str) and os.path.exists(pth):
                            pid = id_from_path(pth)
                            if pid in seen_ids:
                                merged_from["dedup_dropped"] += 1; continue
                            vecs.append(z.astype("float32")); paths.append(pth); seen_ids.add(pid)
                        else:
                            merged_from["skipped_no_path"] += 1
                merged_from["base"] = chosen
            except Exception:
                pass

    dyn_npzs = sorted(glob.glob(os.path.join(APP_STATE_DIR, f"dyn_emb_*_{cache_tag(backbone_name)}.npz")))
    if not dyn_npzs:
        alt_dyn = sorted(glob.glob(os.path.join(APP_STATE_DIR, "dyn_emb_*.npz")))
        for p in alt_dyn:
            try:
                if np.load(p)["embeddings"].shape[1] == dim:
                    dyn_npzs.append(p)
            except Exception:
                continue
    for p in dyn_npzs:
        try:
            d = np.load(p, allow_pickle=True)
            Z = d["embeddings"]; P = d["paths"].tolist() if "paths" in d.files else None
            if Z.shape[1] != dim: continue
            if P is None:
                merged_from["skipped_no_path"] += len(Z); continue
            added_any = False
            for z, pth in zip(Z, P):
                if isinstance(pth, str) and os.path.exists(pth):
                    pid = id_from_path(pth)
                    if pid in seen_ids:
                        merged_from["dedup_dropped"] += 1; continue
                    vecs.append(z.astype("float32")); paths.append(pth); seen_ids.add(pid); added_any = True
                else:
                    merged_from["skipped_no_path"] += 1
            if added_any:
                merged_from["dyn_added"].append(p)
        except Exception:
            continue

    if not vecs:
        merged = np.zeros((0, dim), dtype="float32"); out_paths = []
    else:
        merged = np.vstack(vecs).astype("float32"); out_paths = paths

    merged_npz = os.path.join(APP_STATE_DIR, f"emb_merged_{cache_tag(backbone_name)}.npz")
    np.savez_compressed(merged_npz, embeddings=merged, paths=np.array(out_paths, dtype=object))

    return {"embeddings": merged, "paths": out_paths, "merged_npz": merged_npz, "sources": merged_from}

def build_catalog(paths: List[str]) -> Tuple[np.ndarray, Dict[int, Dict[str,str]]]:
    ids = []
    catalog = {}
    for p in paths:
        if isinstance(p, str) and os.path.exists(p):
            pid = id_from_path(p)
            if pid not in catalog:
                catalog[pid] = {"path": p}
                ids.append(pid)
    return coerce_ids_int64(ids), catalog

def auto_expand_and_refresh(term: str, *, data_root: str, device: str, embedder, manager: IndexManager):
    if not term or term.strip()=="":
        return []
    st.info(f"Expanding database for: **{term}**")
    new_paths = expand_to_dynamic_folder(
        data_root=data_root, term=term,
        per_term=AUTO_PER_TERM, max_side=AUTO_MAX_SIDE, quality=AUTO_JPEG_QUALITY,
        phash_threshold=PHASH_TH,
    )
    if not new_paths:
        st.warning("No new images were added (duplicates or errors).")
        return []
    z_new = embed_images(embedder, new_paths, device=device, batch=64)
    persist_dynamic_embeddings(term, z_new, new_paths, st.session_state.embedder_name)

    new_ids = coerce_ids_int64([id_from_path(p) for p in new_paths])
    st.session_state.catalog.update({int(i):{"path":p} for i,p in zip(new_ids, new_paths)})

    if st.session_state.embeddings is None or st.session_state.ids is None:
        st.session_state.embeddings = z_new
        st.session_state.ids = new_ids
    else:
        st.session_state.embeddings = np.concatenate([st.session_state.embeddings, z_new], axis=0)
        st.session_state.ids = np.concatenate([st.session_state.ids, new_ids], axis=0)

    manager.rebuild(st.session_state.embeddings, st.session_state.ids)
    np.savez_compressed(os.path.join(APP_STATE_DIR, f"emb_{cache_tag(st.session_state.embedder_name)}.npz"),
                        embeddings=st.session_state.embeddings,
                        paths=np.array([st.session_state.catalog[int(i)]["path"] for i in st.session_state.ids], dtype=object))
    st.success(f"Added {len(new_paths)} images for '{term}'. Index rebuilt.")
    return new_paths

st.set_page_config(page_title="Dynamic Image Retrieval (Caltech-256)", layout="wide")
st.title("🔎 Dynamic Image Retrieval — Caltech-256")

with st.sidebar:
    st.header("Settings")
    data_root = st.text_input("Dataset root", value=DEFAULT_DATA_ROOT)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.markdown(f"**Device:** `{device}`")
    backbone_name = st.selectbox("Backbone", list(BACKBONES.keys()), index=0, key="bb_select")
    indexer = st.selectbox("Indexer", ["FAISS Flat", "FAISS HNSW", "Annoy"], index=0)
    topk = st.slider("Top-K", 1, 50, 12)
    rebuild = st.button("Rebuild Index (full)")
    save_idx = st.button("💾 Save Snapshot")
    load_idx = st.button("📥 Load Snapshot")

with st.sidebar:
    st.divider()
    st.subheader("NPZ Auto-Merge")
    auto_merge_npz = st.checkbox("Auto-load CP2 NPZs and merge dynamic NPZs", value=True)
    merge_now = st.button("🔄 Rescan & Merge NPZs")

with st.sidebar:
    st.divider()
    use_external_npz = st.checkbox("Use a specific NPZ (override)", value=False)
    external_npz = st.text_input("NPZ path", value="", help="If set, overrides auto-merge for this session.")

tabs = st.tabs(["Search (Image/Text)", "Batch Expansion", "Manage Dynamic Images"])

def ensure_base_and_embedder():
    ensure_base_state(data_root)
    ensure_embedder(backbone_name, device)

ensure_base_and_embedder()

merged_report = None

if use_external_npz and external_npz and os.path.exists(external_npz):
    try:
        data = np.load(external_npz, allow_pickle=True)
        Z = data["embeddings"]; P = data["paths"].tolist() if "paths" in data.files else None
        dim = Z.shape[1]
        expected_dim = BACKBONES[backbone_name](device).out_dim
        if dim != expected_dim:
            auto = backbone_for_dim(dim)
            if auto is None:
                st.error(f"NPZ has dim {dim}, which doesn’t match the selected backbone ({backbone_name}, dim {expected_dim}).")
                st.stop()
            if auto != backbone_name:
                st.warning(f"NPZ dim {dim} suggests backbone '{auto}'. Switching.")
                backbone_name = auto
                ensure_embedder(backbone_name, device)
        if P is None:
            st.error("Override NPZ does not include 'paths'. This version requires paths for stable IDs.")
            st.stop()
        keep_vecs, keep_paths = [], []
        skipped = 0
        for z, pth in zip(Z, P):
            if isinstance(pth, str) and os.path.exists(pth):
                keep_vecs.append(z.astype("float32")); keep_paths.append(pth)
            else:
                skipped += 1
        st.session_state.embeddings = np.vstack(keep_vecs).astype("float32") if keep_vecs else np.zeros((0, expected_dim), dtype="float32")
        st.session_state.ids, st.session_state.catalog = build_catalog(keep_paths)
        st.session_state.manager = None
        st.success(f"Loaded NPZ with {len(keep_paths)} vectors (skipped {skipped} without path).")
    except Exception as e:
        st.error(f"Failed to load NPZ: {e}")
        st.stop()
elif auto_merge_npz or merge_now:
    merged = scan_and_merge_npzs(backbone_name, device)
    Z = merged["embeddings"]; P = merged["paths"]
    st.session_state.embeddings = Z.astype("float32")
    st.session_state.ids, st.session_state.catalog = build_catalog(P)
    st.session_state.manager = None
    merged_report = merged
    src = merged["sources"]
    st.success(f"NPZ merged. Base: {src['base'] or 'None'} • Dynamic added: {len(src['dyn_added'])} • Kept: {len(P)} • Skipped(no path): {src['skipped_no_path']} • Dedup: {src['dedup_dropped']}")
    st.info(f"Merged NPZ saved: {merged['merged_npz']}")

if st.session_state.embeddings is None:
    st.info("Computing embeddings for on-disk dataset…")
    paths = st.session_state.paths or []
    Z = embed_images(st.session_state.embedder, paths, device=device, batch=64)
    keep_paths = [p for p in paths if isinstance(p, str) and os.path.exists(p)]
    st.session_state.embeddings = Z.astype("float32")
    st.session_state.ids, st.session_state.catalog = build_catalog(keep_paths)
    st.session_state.manager = None

if (st.session_state.manager is None) or (st.session_state.indexer != indexer) or rebuild or merge_now:
    st.session_state.indexer = indexer
    st.session_state.manager = IndexManager(indexer=indexer, dim=st.session_state.embeddings.shape[1],
                                            state_dir=APP_STATE_DIR, hnsw_m=32, ef=64, annoy_trees=50)
    st.session_state.manager.fit_with_ids(st.session_state.embeddings, st.session_state.ids)
    st.success(f"Index built: {indexer} | DB size: {st.session_state.embeddings.shape[0]}")
    if merged_report is not None:
        st.toast("Embeddings updated from NPZ merge. You can search again now.", icon="✅")

with st.sidebar:
    if save_idx:
        st.session_state.manager.save()
        with open(os.path.join(APP_STATE_DIR, "catalog.pkl"), "wb") as f:
            pickle.dump(st.session_state.catalog, f)
        with open(os.path.join(APP_STATE_DIR, "ids.npy"), "wb") as f:
            np.save(f, st.session_state.ids)
        st.success("Snapshot saved.")
    if load_idx:
        ok = st.session_state.manager.load()
        cat_path = os.path.join(APP_STATE_DIR, "catalog.pkl")
        ids_path = os.path.join(APP_STATE_DIR, "ids.npy")
        if ok and os.path.exists(cat_path) and os.path.exists(ids_path):
            st.session_state.catalog = pickle.load(open(cat_path, "rb"))
            st.session_state.ids = np.load(ids_path)
            paths = [st.session_state.catalog[int(i)]["path"] for i in st.session_state.ids if int(i) in st.session_state.catalog]
            st.session_state.embeddings = embed_images(st.session_state.embedder, paths, device=device, batch=64)
            st.session_state.manager.rebuild(st.session_state.embeddings, st.session_state.ids)
            st.success("Snapshot loaded and index refreshed.")
        elif ok:
            st.warning("Index loaded but missing catalog/ids; please keep both files.")
        else:
            st.error("No snapshot to load.")

with tabs[0]:
    st.subheader("Search")
    tab_img, tab_text = st.tabs(["Image → Image", "Text → Image (CLIP)"])

    def show_hits(id_rows: np.ndarray, sim_rows: np.ndarray):
        ids = id_rows[0].tolist()
        scores = sim_rows[0].tolist()
        cols = st.columns(6)
        for i, (img_id, sc) in enumerate(zip(ids, scores)):
            with cols[i % 6]:
                meta = st.session_state.catalog.get(int(img_id))
                if meta and os.path.exists(meta["path"]):
                    st.image(meta["path"], caption=f"{os.path.basename(meta['path'])}\nscore={sc:.3f}", use_container_width=True)
                else:
                    st.caption(f"[{img_id}] score={sc:.3f} (missing)")

    with tab_text:
        txt = st.text_input("Describe what you want (e.g., 'a helicopter in the sky')")
        if txt:
            if hasattr(st.session_state.embedder, "encode_text"):
                with torch.no_grad():
                    q = st.session_state.embedder.encode_text([txt]).cpu().numpy().astype("float32")
                db_dim = st.session_state.embeddings.shape[1]
                if q.shape[1] != db_dim:
                    st.error(f"Query dim {q.shape[1]} ≠ index dim {db_dim}. Load a matching NPZ or switch backbone.")
                    st.stop()
                sims, id_hits = st.session_state.manager.search(q, topk)
                show_hits(id_hits, sims)
                if _coverage_is_weak(list(sims[0])):
                    st.warning("Coverage looks weak. Expanding the DB with a small web batch…")
                    added = auto_expand_and_refresh(
                        term=txt, data_root=data_root, device=device,
                        embedder=st.session_state.embedder, manager=st.session_state.manager
                    )
                    if added:
                        sims, id_hits = st.session_state.manager.search(q, topk)
                        show_hits(id_hits, sims)
                        st.toast("Database expanded. Try searching again if needed.", icon="🧩")
            else:
                st.warning("Text search requires the CLIP backbone.")

    with tab_img:
        q_img = st.file_uploader("Upload a query image", type=["jpg","jpeg","png","bmp","gif","webp"], key="imgq")
        if q_img:
            im = Image.open(io.BytesIO(q_img.read())).convert("RGB")
            st.image(im, caption="Query", width=256)
            z = embed_pils(st.session_state.embedder, [im], device=device)
            db_dim = st.session_state.embeddings.shape[1]
            if z.shape[1] != db_dim:
                st.error(f"Query dim {z.shape[1]} ≠ index dim {db_dim}. Load a matching NPZ or switch backbone.")
                st.stop()
            sims, id_hits = st.session_state.manager.search(z, topk)
            show_hits(id_hits, sims)
            need_expand = _coverage_is_weak(list(sims[0]))
            auto_term = None
            if need_expand and isinstance(st.session_state.embedder, CLIPViTB32) and len(st.session_state.classes) > 0:
                import clip
                prompts = [f"a photo of {c.replace('_',' ')}" for c in st.session_state.classes]
                with torch.no_grad():
                    tok = clip.tokenize(prompts).to(device)
                    class_text = st.session_state.embedder.clip.encode_text(tok)
                    class_text = class_text / class_text.norm(dim=-1, keepdim=True)
                    qv = z / np.linalg.norm(z, axis=1, keepdims=True)
                    sims_ct = (torch.from_numpy(qv).to(device) @ class_text.t()).cpu().numpy()[0]
                    best_idx = int(np.argmax(sims_ct))
                    auto_term = prompts[best_idx]
            elif need_expand and len(id_hits[0])>0:
                first_id = int(id_hits[0][0])
                meta = st.session_state.catalog.get(first_id)
                if meta:
                    cls_name = os.path.basename(os.path.dirname(meta["path"]))
                    auto_term = f"a photo of {cls_name.replace('_',' ')}"
            if need_expand and auto_term:
                st.warning("Coverage looks weak. Expanding the DB with a small web batch…")
                added = auto_expand_and_refresh(
                    term=auto_term, data_root=data_root, device=device,
                    embedder=st.session_state.embedder, manager=st.session_state.manager
                )
                if added:
                    sims, id_hits = st.session_state.manager.search(z, topk)
                    show_hits(id_hits, sims)
                    st.toast("Database expanded. Try searching again if needed.", icon="🧩")

with tabs[1]:
    st.subheader("Batch Expansion (Web)")
    if DDGS is None or requests is None:
        st.error("Please install: `pip install duckduckgo-search requests imagehash`")
    terms_src = st.radio("Provide terms via…", ["Textarea", "CSV upload"], horizontal=True)
    terms=[]
    if terms_src=="Textarea":
        raw = st.text_area("One term per line", height=140, placeholder="electric scooter\ngolden retriever puppy\nblack DSLR camera on table")
        terms = [t.strip() for t in raw.splitlines() if t.strip()]
    else:
        up = st.file_uploader("Upload CSV with a header 'term'", type=["csv"], key="terms_csv_up")
        if up:
            try:
                df = pd.read_csv(up)
                col = "term" if "term" in df.columns else df.columns[0]
                terms = [str(x).strip() for x in df[col].tolist() if str(x).strip()]
                st.success(f"Loaded {len(terms)} terms")
            except Exception as e:
                st.error(f"CSV parse failed: {e}")
    c1,c2,c3,c4 = st.columns(4)
    with c1: per_term = st.number_input("Images per term", 1, 100, 20)
    with c2: max_side = st.number_input("Max side (px)", 64, 1024, 256, step=32)
    with c3: quality = st.number_input("JPEG quality", 40, 95, 85)
    with c4: phash_th = st.number_input("pHash threshold", 0, 16, 4)
    if st.button("Run Batch Expansion"):
        if not terms: st.warning("Add at least one term.")
        elif DDGS is None or requests is None: st.error("Install duckduckgo-search and requests first.")
        else:
            new_paths_total=[]
            term_failures = 0
            for t in terms:
                try:
                    added = expand_to_dynamic_folder(data_root, t, int(per_term), int(max_side), int(quality), int(phash_th))
                    if not added:
                        st.warning(f"No images added for '{t}' (duplicates/errors).")
                    new_paths_total += added
                except Exception as e:
                    term_failures += 1
                    st.warning(f"Failed '{t}': {e}")
                time.sleep(random.uniform(0.25, 0.6))
            if new_paths_total:
                st.info("Embedding new images…")
                z_new = embed_images(st.session_state.embedder, new_paths_total, device=device, batch=64)
                np.savez_compressed(os.path.join(APP_STATE_DIR, f"dyn_emb_batch_{cache_tag(st.session_state.embedder_name)}.npz"),
                                    embeddings=z_new, paths=np.array(new_paths_total, dtype=object))
                new_ids = coerce_ids_int64([id_from_path(p) for p in new_paths_total])
                st.session_state.catalog.update({int(i):{"path":p} for i,p in zip(new_ids, new_paths_total)})

                if st.session_state.embeddings is None or st.session_state.ids is None:
                    st.session_state.embeddings = z_new
                    st.session_state.ids = new_ids
                else:
                    st.session_state.embeddings = np.concatenate([st.session_state.embeddings, z_new], axis=0)
                    st.session_state.ids = np.concatenate([st.session_state.ids, new_ids], axis=0)

                st.session_state.manager.rebuild(st.session_state.embeddings, st.session_state.ids)
                np.savez_compressed(os.path.join(APP_STATE_DIR, f"emb_{cache_tag(st.session_state.embedder_name)}.npz"),
                                    embeddings=st.session_state.embeddings,
                                    paths=np.array([st.session_state.catalog[int(i)]["path"] for i in st.session_state.ids], dtype=object))
                st.success(f"Added {len(new_paths_total)} images over {len(terms)} term(s). Failures: {term_failures}")
                st.toast("New images embedded & index updated. You can search again now.", icon="✅")
            else:
                st.info(f"No new images added. Failures: {term_failures}")

with tabs[2]:
    st.subheader("Manage Dynamic Images")
    dyn_dirs = sorted([
        os.path.join(data_root, d) for d in os.listdir(data_root)
        if d.startswith("dynamic_") and os.path.isdir(os.path.join(data_root, d))
    ])
    all_dyn = []
    for d in dyn_dirs:
        all_dyn += sorted(glob.glob(os.path.join(d, "*.jpg")))
    if not all_dyn:
        st.caption("No dynamic images yet.")
    else:
        sel = st.multiselect("Select dynamic images to remove", all_dyn, default=[])
        if st.button("🗑️ Remove selected"):
            removed=0
            for p in sel:
                try: os.remove(p); removed+=1
                except Exception: pass
            paths, labels, classes = list_images_by_class(data_root)
            dyn_paths = load_dynamic_set(data_root)
            full_paths = paths + dyn_paths
            st.info("Re-embedding after delete…")
            Z = embed_images(st.session_state.embedder, full_paths, device=device, batch=64)
            st.session_state.embeddings = Z.astype("float32")
            st.session_state.ids, st.session_state.catalog = build_catalog(full_paths)
            st.session_state.manager.rebuild(st.session_state.embeddings, st.session_state.ids)
            np.savez_compressed(os.path.join(APP_STATE_DIR, f"emb_{cache_tag(st.session_state.embedder_name)}.npz"),
                                embeddings=st.session_state.embeddings,
                                paths=np.array([st.session_state.catalog[int(i)]["path"] for i in st.session_state.ids], dtype=object))
            st.success(f"Removed {removed} files. DB size: {len(st.session_state.ids)}")

st.markdown("---")
st.caption(
    "• CP2 NPZs auto-loaded from: C:/Users/punna/Downloads/IMRS/runs/ckpt2/cache "
    "• Dynamic NPZs merged from: runs/streamlit_state "
    "• Merged snapshot: runs/streamlit_state/emb_merged_<backbone>.npz "
    "• Index is ID-mapped, merges are de-duped and path-validated."
)
