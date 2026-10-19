import argparse
import os
import time
import csv
import json
import glob
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm

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
    import cv2
except Exception:
    cv2 = None
try:
    from imagehash import phash as _phash
except Exception:
    _phash = None
try:
    from annoy import AnnoyIndex
except Exception:
    AnnoyIndex = None

def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_json(obj, fp):
    with open(fp, "w") as f:
        json.dump(obj, f, indent=2)

def _aspect_pad_resize(img: Image.Image, size: int = 224) -> Image.Image:
    w, h = img.size
    scale = size / max(w, h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    img = img.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return canvas

def _to_cv(img: Image.Image):
    if cv2 is None:
        return None
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def _to_pil(img_cv):
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

def _apply_clahe(img: Image.Image, clip=2.0, tile=8) -> Image.Image:
    if cv2 is None:
        return img
    cv = _to_cv(img)
    lab = cv2.cvtColor(cv, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    L = clahe.apply(L)
    lab = cv2.merge([L, A, B])
    cv = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return _to_pil(cv)

def _grayscale(img: Image.Image) -> Image.Image:
    if cv2 is None:
        return img.convert("L").convert("RGB")
    cv = _to_cv(img)
    gray = cv2.cvtColor(cv, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return _to_pil(gray3)

def _color_norm(img: Image.Image) -> Image.Image:
    if cv2 is None:
        return img
    cv = _to_cv(img).astype(np.float32)
    for c in range(3):
        ch = cv[:, :, c]
        ch = (ch - ch.mean()) / (ch.std() + 1e-6) * 50 + 128
        cv[:, :, c] = np.clip(ch, 0, 255)
    cv = cv.astype(np.uint8)
    return _to_pil(cv)

def preprocess_cp1(raw_root: str, proc_root: str, img_size: int = 224,
                   phash_threshold: int = 4, use_clahe: bool = False,
                   grayscale: bool = False, color_norm: bool = False):
    if _phash is None:
        raise ImportError("imagehash is required for preprocess: pip install imagehash")
    ensure_dir(proc_root)
    classes = sorted([d for d in os.listdir(raw_root) if os.path.isdir(os.path.join(raw_root, d))])
    seen = {c: [] for c in classes}
    rows = []
    kept = 0
    skipped = 0
    for cid, cls in enumerate(classes):
        src = os.path.join(raw_root, cls)
        dst = os.path.join(proc_root, cls)
        ensure_dir(dst)
        for p in glob.glob(os.path.join(src, "*")):
            if not p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
                continue
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                skipped += 1
                continue
            if use_clahe:
                img = _apply_clahe(img)
            if grayscale:
                img = _grayscale(img)
            if color_norm:
                img = _color_norm(img)
            h = _phash(img)
            if any((h - oh) <= phash_threshold for oh in seen[cls]):
                skipped += 1
                continue
            seen[cls].append(h)
            out_img = _aspect_pad_resize(img, img_size)
            out_name = os.path.splitext(os.path.basename(p))[0] + ".jpg"
            out_path = os.path.join(dst, out_name)
            out_img.save(out_path, quality=95)
            rows.append({"path": out_path, "class_id": cid, "class_name": cls})
            kept += 1
    manifest = pd.DataFrame(rows)
    manifest.to_csv(os.path.join(proc_root, "manifest.csv"), index=False)
    print(f"[preprocess] kept={kept}, skipped={skipped}, classes={len(classes)}")

def list_images_by_class(data_root: str) -> Tuple[List[str], List[int], List[str]]:
    man = os.path.join(data_root, "manifest.csv")
    if os.path.exists(man):
        df = pd.read_csv(man)
        classes = sorted(df["class_name"].unique())
        c2i = {c: i for i, c in enumerate(classes)}
        paths = df["path"].tolist()
        labels = [c2i[c] for c in df["class_name"].tolist()]
        return paths, labels, classes
    classes = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
    class_to_id = {c: i for i, c in enumerate(classes)}
    paths, labels = [], []
    for c in classes:
        folder = os.path.join(data_root, c)
        for p in glob.glob(os.path.join(folder, "*")):
            if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
                paths.append(p)
                labels.append(class_to_id[c])
    return paths, labels, classes

class Embedder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.out_dim = dim
    def forward(self, img_batch: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    @property
    def preprocess(self):
        raise NotImplementedError

class CLIPViTB32(Embedder):
    def __init__(self, device):
        super().__init__(dim=512)
        import clip
        self.device = device
        self.clip, self.prep = clip.load("ViT-B/32", device=device)
    @torch.no_grad()
    def forward(self, img_batch: torch.Tensor) -> torch.Tensor:
        feats = self.clip.encode_image(img_batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats
    @property
    def preprocess(self):
        return self.prep
    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        import clip
        tokens = clip.tokenize(texts).to(self.device)
        feats = self.clip.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

class ResNet50GAP(Embedder):
    def __init__(self, device):
        super().__init__(dim=2048)
        self.backbone = tv.models.resnet50(weights=tv.models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone.fc = nn.Identity()
        self.device = device
        self.backbone.to(device).eval()
        self._prep = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    @torch.no_grad()
    def forward(self, img_batch: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(img_batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats
    @property
    def preprocess(self):
        return self._prep

class ViT_B16_TorchVision(Embedder):
    def __init__(self, device):
        super().__init__(dim=768)
        weights = tv.models.ViT_B_16_Weights.DEFAULT
        self.model = tv.models.vit_b_16(weights=weights)
        self.model.heads = nn.Identity()
        self.model.to(device).eval()
        self._prep = weights.transforms()
        self.device = device
    @torch.no_grad()
    def forward(self, img_batch):
        feats = self.model(img_batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats
    @property
    def preprocess(self):
        return self._prep

BACKBONES = {
    "clip_vit_b32": CLIPViTB32,
    "resnet50": ResNet50GAP,
    "vit_b16": ViT_B16_TorchVision,
}

class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, out_dim)
        )
    def forward(self, x):
        x = self.net(x)
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
        return x

class AnnoyIP:
    def __init__(self, d, n_trees=50, **_ignored):
        if AnnoyIndex is None:
            raise ImportError("annoy is required for Annoy index: pip install annoy")
        self.d = d
        self.n_trees = n_trees
        self.ann = AnnoyIndex(d, metric='angular')
        self._built = False
    def train(self, X):
        for i in range(X.shape[0]):
            self.ann.add_item(i, X[i].tolist())
        self.ann.build(self.n_trees)
        self._built = True
    def add(self, X):
        if not self._built:
            self.train(X)
    def search(self, Q, k):
        idxs = []
        sims = []
        for q in Q:
            ii, dd = self.ann.get_nns_by_vector(q.tolist(), int(k), include_distances=True)
            ss = [1 - (d**2) / 2 for d in dd]
            idxs.append(ii)
            sims.append(ss)
        return np.array(sims, dtype=np.float32), np.array(idxs, dtype=np.int64)

def build_index(name: str, d: int, ef: int = 64, nlist: int = 4096, pq_m: int = 32, hnsw_m: int = 32, **kw):
    name = name.lower()
    if name == "annoy":
        return AnnoyIP(d, **kw)
    if faiss is None:
        raise ImportError("faiss-cpu is required for FAISS indexers: pip install faiss-cpu")
    if name in ("flat_ip", "flat"):
        return faiss.IndexFlatIP(d)
    if name == "ivf_flat":
        quantizer = faiss.IndexFlatIP(d)
        return faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    if name == "ivf_pq":
        quantizer = faiss.IndexFlatIP(d)
        return faiss.IndexIVFPQ(quantizer, d, nlist, pq_m, 8, faiss.METRIC_INNER_PRODUCT)
    if name == "hnsw":
        index = faiss.IndexHNSWFlat(d, hnsw_m)
        index.hnsw.efSearch = ef
        return index
    raise ValueError(f"Unknown indexer: {name}")

def embed_paths(embedder: Embedder, img_paths: List[str], batch: int = 64, device: str = "cpu") -> np.ndarray:
    embs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(img_paths), batch), desc="Embedding"):
            chunk = img_paths[i:i+batch]
            imgs = []
            for p in chunk:
                try:
                    img = Image.open(p).convert("RGB")
                except Exception:
                    continue
                imgs.append(embedder.preprocess(img))
            if not imgs:
                continue
            imgs = torch.stack(imgs).to(device)
            feats = embedder(imgs)
            embs.append(feats.cpu().numpy().astype("float32"))
    return np.vstack(embs)

def average_precision(true: int, retrieved: List[int], k: int) -> float:
    ap, hits = 0.0, 0
    for i, y in enumerate(retrieved[:k], start=1):
        if y == true:
            hits += 1
            ap += hits / i
    return ap / k

def precision_recall_at_k(true: int, retrieved: List[int], k: int) -> Tuple[float, float]:
    topk = retrieved[:k]
    hits = sum(1 for y in topk if y == true)
    precision = hits / k
    denom = max(1, retrieved.count(true))
    recall = hits / denom
    return precision, recall

def dcg_at_k(rel: List[int], k: int) -> float:
    r = np.array(rel[:k], dtype=float)
    if r.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, r.size + 2))
    return float((r * discounts).sum())

def ndcg_at_k(true: int, retrieved: List[int], k: int) -> float:
    rel = [1 if y == true else 0 for y in retrieved]
    dcg = dcg_at_k(rel, k)
    idcg = dcg_at_k(sorted(rel, reverse=True), k)
    return 0.0 if idcg == 0 else dcg / idcg

def k_reciprocal_rerank(query_idx, neighbors, k=20, alpha=0.1):
    boosts = {}
    for n in neighbors[query_idx][:k]:
        if query_idx in set(neighbors[n][:k]):
            boosts[n] = alpha
    return boosts

@dataclass
class BenchConfig:
    backbone: str
    indexer: str
    topk: int
    sample_queries: int
    nlist: int = 4096
    pq_m: int = 32
    hnsw_m: int = 32
    ef: int = 64
    n_trees: int = 50

def run_bench(cfg: BenchConfig, data_root: str, out_dir: str, device: str = "cpu", persist_index: bool = False) -> Dict:
    seed_everything(42)
    ensure_dir(out_dir)
    cache_dir = os.path.join(out_dir, "cache")
    ensure_dir(cache_dir)
    paths, labels, classes = list_images_by_class(data_root)
    labels = np.asarray(labels, dtype=np.int32)
    if cfg.backbone.endswith("_ft"):
        base = cfg.backbone.replace("_ft", "")
        emb = BACKBONES[base](device)
        head_fp = os.path.join(out_dir, "train", f"proj_{base}.pt")
        if not os.path.exists(head_fp):
            raise FileNotFoundError("Fine-tuned projection not found. Run 'finetune' first.")
        head = ProjectionHead(emb.out_dim)
        head.load_state_dict(torch.load(head_fp, map_location=device))
        head.to(device).eval()
        class FTEmbedder(Embedder):
            def __init__(self, base_emb, head):
                super().__init__(dim=512)
                self.base = base_emb
                self.head = head
            @property
            def preprocess(self):
                return self.base.preprocess
            @torch.no_grad()
            def forward(self, x):
                z = self.base(x)
                return self.head(z)
        embedder = FTEmbedder(emb, head)
    else:
        embedder = BACKBONES[cfg.backbone](device)
    emb_cache = os.path.join(cache_dir, f"emb_{cfg.backbone}.npz")
    if os.path.exists(emb_cache):
        embeddings = np.load(emb_cache)["embeddings"]
    else:
        embeddings = embed_paths(embedder, paths, batch=64, device=device)
        np.savez_compressed(emb_cache, embeddings=embeddings)
    d = embeddings.shape[1]
    t_build0 = time.time()
    index = build_index(cfg.indexer, d, ef=cv2 if False else cfg.ef, nlist=cfg.nlist, pq_m=cfg.pq_m, hnsw_m=cfg.hnsw_m, n_trees=cfg.n_trees)
    if faiss is not None and isinstance(index, (faiss.IndexIVFFlat, faiss.IndexIVFPQ)):
        n_train = min(100000, embeddings.shape[0])
        sample = embeddings[np.random.choice(embeddings.shape[0], n_train, replace=False)]
        index.train(sample)
    index.add(embeddings)
    build_time = time.time() - t_build0
    smallK = min(50, embeddings.shape[0]-1)
    sims_tmp, pre_neighbors = index.search(embeddings, smallK)
    rng = np.random.default_rng(123)
    num_q = min(cfg.sample_queries, len(paths))
    q_idx = rng.choice(len(paths), size=num_q, replace=False)
    t0 = time.time()
    sims, nbrs = index.search(embeddings[q_idx], cfg.topk + 1)
    t_search = time.time() - t0
    clean_neighbors = []
    for qi, neigh in enumerate(nbrs):
        neigh = [n for n in neigh if n != q_idx[qi]]
        clean_neighbors.append(neigh[:cfg.topk])
    use_rerank = False
    if use_rerank:
        for j, neigh in enumerate(clean_neighbors):
            boosts = k_reciprocal_rerank(q_idx[j], pre_neighbors, k=20, alpha=0.1)
            for r, n in enumerate(neigh):
                if n in boosts:
                    sims[j][r] += boosts[n]
            order = np.argsort(-sims[j][:len(neigh)])
            clean_neighbors[j] = [neigh[o] for o in order]
    K = cfg.topk
    aps, recs, precs, ndcgs = [], [], [], []
    per_class = {i: {"n": 0, "ap": 0.0, "rec": 0.0, "prec": 0.0, "ndcg": 0.0} for i in range(len(classes))}
    for qi, neigh in enumerate(clean_neighbors):
        y_true = int(labels[q_idx[qi]])
        retrieved_labels = [int(labels[n]) for n in neigh]
        ap = average_precision(y_true, retrieved_labels, K)
        prec, rec = precision_recall_at_k(y_true, retrieved_labels, K)
        nd = ndcg_at_k(y_true, retrieved_labels, K)
        aps.append(ap); precs.append(prec); recs.append(rec); ndcgs.append(nd)
        pc = per_class[y_true]
        pc["n"] += 1
        pc["ap"] += ap; pc["prec"] += prec; pc["rec"] += rec; pc["ndcg"] += nd
    for cid in per_class:
        if per_class[cid]["n"] > 0:
            for m in ("ap", "prec", "rec", "ndcg"):
                per_class[cid][m] /= per_class[cid]["n"]
    summary = {
        "backbone": cfg.backbone,
        "indexer": cfg.indexer,
        "topk": cfg.topk,
        "sample_queries": num_q,
        "mean_ap@K": float(np.mean(aps)),
        "mean_precision@K": float(np.mean(precs)),
        "mean_recall@K": float(np.mean(recs)),
        "mean_ndcg@K": float(np.mean(ndcgs)),
        "build_time_sec": build_time,
        "search_time_sec": t_search,
        "qps": num_q / t_search if t_search > 0 else float("inf"),
        "num_db": len(paths),
        "dim": d,
    }
    ensure_dir(out_dir)
    summ_csv = os.path.join(out_dir, "summary.csv")
    exists = os.path.exists(summ_csv)
    with open(summ_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(summary)
    pcm = pd.DataFrame([
        {"class_id": cid, "class_name": classes[cid], **per_class[cid]} for cid in range(len(classes))
    ])
    pcm_out = os.path.join(out_dir, f"per_class_{cfg.backbone}_{cfg.indexer}.csv")
    pcm.to_csv(pcm_out, index=False)
    ex_rows = []
    for j in range(min(25, len(q_idx))):
        qi = q_idx[j]
        yq = int(labels[qi])
        row = {"query_path": paths[qi], "query_class": classes[yq]}
        for r, n in enumerate(clean_neighbors[j][:10]):
            row[f"rank{r+1}_path"] = paths[n]
            row[f"rank{r+1}_class"] = classes[int(labels[n])]
        ex_rows.append(row)
    pd.DataFrame(ex_rows).to_csv(os.path.join(out_dir, f"examples_{cfg.backbone}_{cfg.indexer}.csv"), index=False)
    if persist_index and (faiss is not None) and hasattr(index, "add") and not isinstance(index, AnnoyIP):
        idx_dir = os.path.join(out_dir, "index"); ensure_dir(idx_dir)
        faiss.write_index(index, os.path.join(idx_dir, f"{cfg.backbone}_{cfg.indexer}.faiss"))
    print("Summary:", json.dumps(summary, indent=2))
    return summary

def make_triplets(labels: np.ndarray, n_triplets: int = 4096):
    rng = np.random.default_rng(0)
    by_class = {}
    for i, y in enumerate(labels):
        by_class.setdefault(int(y), []).append(i)
    classes = list(by_class.keys())
    triplets = []
    for _ in range(n_triplets):
        c_pos = rng.choice(classes)
        c_neg = rng.choice([c for c in classes if c != c_pos])
        a, p = rng.choice(by_class[c_pos], size=2, replace=False)
        n = rng.choice(by_class[c_neg])
        triplets.append((a, p, n))
    return triplets

def finetune_projection(data_root: str, out_dir: str, backbone: str, device: str,
                        epochs: int, batch_size: int, lr: float, margin: float):
    seed_everything(123)
    ensure_dir(os.path.join(out_dir, "train"))
    paths, labels, _ = list_images_by_class(data_root)
    labels = np.asarray(labels, dtype=np.int64)
    base = BACKBONES[backbone](device)
    cache_dir = os.path.join(out_dir, "cache"); ensure_dir(cache_dir)
    emb_cache = os.path.join(cache_dir, f"emb_{backbone}.npz")
    if os.path.exists(emb_cache):
        base_embs = np.load(emb_cache)["embeddings"]
    else:
        base_embs = embed_paths(base, paths, batch=64, device=device)
        np.savez_compressed(emb_cache, embeddings=base_embs)
    base_embs_t = torch.from_numpy(base_embs).to(device)
    head = ProjectionHead(in_dim=base.out_dim, out_dim=512).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    criterion = nn.TripletMarginLoss(margin=margin, p=2)
    for ep in range(epochs):
        head.train()
        triplets = make_triplets(labels, n_triplets=8192)
        losses = []
        for i in range(0, len(triplets), batch_size):
            batch = triplets[i:i+batch_size]
            if not batch:
                continue
            a_idx, p_idx, n_idx = zip(*batch)
            a = head(base_embs_t[list(a_idx)])
            p = head(base_embs_t[list(p_idx)])
            n = head(base_embs_t[list(n_idx)])
            loss = criterion(a, p, n)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.item()))
        print(f"[FT] epoch {ep+1}/{epochs} | loss={np.mean(losses):.4f} | batches={len(losses)}")
    torch.save(head.state_dict(), os.path.join(out_dir, "train", f"proj_{backbone}.pt"))
    save_json({"backbone": backbone, "epochs": epochs, "batch_size": batch_size, "lr": lr, "margin": margin},
              os.path.join(out_dir, "train", f"proj_{backbone}.json"))

def launch_demo(data_root: str, out_dir: str, device: str,
                backbone: str = "clip_vit_b32", indexer: str = "flat_ip"):
    import gradio as gr
    paths, labels, classes = list_images_by_class(data_root)
    labels = np.asarray(labels, dtype=np.int32)
    emb_cls = BACKBONES[backbone]
    embedder = emb_cls(device)
    cache_fp = os.path.join(out_dir, "cache", f"emb_{backbone}.npz")
    if os.path.exists(cache_fp):
        embeddings = np.load(cache_fp)["embeddings"]
    else:
        ensure_dir(os.path.join(out_dir, "cache"))
        embeddings = embed_paths(embedder, paths, batch=64, device=device)
        np.savez_compressed(cache_fp, embeddings=embeddings)
    d = embeddings.shape[1]
    index = build_index(indexer, d)
    index.add(embeddings)
    def search_image(img, k):
        img = img.convert("RGB")
        with torch.no_grad():
            x = embedder.preprocess(img).unsqueeze(0).to(device)
            q = embedder(x).cpu().numpy().astype("float32")
        sims, idxs = index.search(q, int(k))
        idxs = idxs[0].tolist(); ss = sims[0].tolist()
        return [(paths[i], f"{classes[int(labels[i])]} | score={ss[j]:.3f}") for j, i in enumerate(idxs)]
    def search_text(prompt, k):
        if not hasattr(embedder, "encode_text"):
            return []
        with torch.no_grad():
            q = embedder.encode_text([prompt]).cpu().numpy().astype("float32")
        sims, idxs = index.search(q, int(k))
        idxs = idxs[0].tolist(); ss = sims[0].tolist()
        return [(paths[i], f"{classes[int(labels[i])]} | score={ss[j]:.3f}") for j, i in enumerate(idxs)]
    with gr.Blocks() as demo:
        gr.Markdown("# Caltech-256 Retrieval Demo (CP2)")
        with gr.Tab("Image → Image"):
            inp = gr.Image(type="pil", label="Query image")
            k = gr.Slider(1, 50, value=10, step=1, label="Top-K")
            out = gr.Gallery(label="Results").style(grid=[5], height=600)
            gr.Button("Search").click(search_image, inputs=[inp, k], outputs=out)
        with gr.Tab("Text → Image (CLIP)"):
            txt = gr.Textbox(label="Query text (e.g., 'a motorbike on a road')")
            k2 = gr.Slider(1, 50, value=10, step=1, label="Top-K")
            out2 = gr.Gallery(label="Results").style(grid=[5], height=600)
            gr.Button("Search").click(search_text, inputs=[txt, k2], outputs=out2)
    demo.launch()

def main():
    parser = argparse.ArgumentParser(description="Caltech-256 Retrieval – CP2 Framework")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_p = sub.add_parser("preprocess", help="Resize+pad to 224, pHash de-dup, optional CLAHE/gray/color-norm, write manifest.csv")
    p_p.add_argument("--raw_root", type=str, required=True)
    p_p.add_argument("--proc_root", type=str, required=True)
    p_p.add_argument("--img_size", type=int, default=224)
    p_p.add_argument("--phash_threshold", type=int, default=4)
    p_p.add_argument("--use_clahe", action="store_true")
    p_p.add_argument("--grayscale", action="store_true")
    p_p.add_argument("--color_norm", action="store_true")
    p_b = sub.add_parser("bench", help="Benchmark backbones/indexers on retrieval metrics")
    p_b.add_argument("--data_root", type=str, required=True)
    p_b.add_argument("--out_dir", type=str, default="runs/ckpt2")
    p_b.add_argument("--backbones", nargs="+", default=["clip_vit_b32", "resnet50", "vit_b16"],
                     choices=list(BACKBONES.keys()) + ["clip_vit_b32_ft"])
    p_b.add_argument("--indexers", nargs="+", default=["flat_ip", "hnsw", "ivf_pq", "annoy"],
                     choices=["flat_ip", "ivf_flat", "ivf_pq", "hnsw", "annoy"])
    p_b.add_argument("--topk", type=int, default=10)
    p_b.add_argument("--sample_queries", type=int, default=1000)
    p_b.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p_b.add_argument("--persist_index", action="store_true")
    p_f = sub.add_parser("finetune", help="Fine-tune a projection head with Triplet loss")
    p_f.add_argument("--data_root", type=str, required=True)
    p_f.add_argument("--out_dir", type=str, default="runs/ckpt2_ft")
    p_f.add_argument("--backbone", type=str, default="clip_vit_b32", choices=["clip_vit_b32", "resnet50", "vit_b16"])
    p_f.add_argument("--epochs", type=int, default=5)
    p_f.add_argument("--batch_size", type=int, default=64)
    p_f.add_argument("--lr", type=float, default=1e-3)
    p_f.add_argument("--margin", type=float, default=0.2)
    p_f.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p_d = sub.add_parser("demo", help="Launch Gradio demo")
    p_d.add_argument("--data_root", type=str, required=True)
    p_d.add_argument("--out_dir", type=str, default="runs/ckpt2")
    p_d.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p_d.add_argument("--backbone", type=str, default="clip_vit_b32", choices=list(BACKBONES.keys()))
    p_d.add_argument("--indexer", type=str, default="flat_ip", choices=["flat_ip", "ivf_flat", "ivf_pq", "hnsw", "annoy"])
    args = parser.parse_args()
    if args.cmd == "preprocess":
        preprocess_cp1(args.raw_root, args.proc_root, img_size=args.img_size,
                       phash_threshold=args.phash_threshold,
                       use_clahe=args.use_clahe, grayscale=args.grayscale, color_norm=args.color_norm)
    elif args.cmd == "bench":
        results = []
        for b in args.backbones:
            for ix in args.indexers:
                print(f"\n=== bench: backbone={b}, indexer={ix} ===")
                cfg = BenchConfig(backbone=b, indexer=ix, topk=args.topk, sample_queries=args.sample_queries)
                results.append(run_bench(cfg, args.data_root, args.out_dir,
                                        device=args.device, persist_index=args.persist_index))
        agg = pd.DataFrame(results)
        agg.to_csv(os.path.join(args.out_dir, "all_experiments.csv"), index=False)
        print("All experiments →", os.path.join(args.out_dir, "all_experiments.csv"))
    elif args.cmd == "finetune":
        finetune_projection(args.data_root, args.out_dir, args.backbone, args.device,
                            args.epochs, args.batch_size, args.lr, args.margin)
        print("Saved fine-tuned projection head. Use backbone '<name>_ft' in bench.")
    elif args.cmd == "demo":
        launch_demo(args.data_root, args.out_dir, args.device, backbone=args.backbone, indexer=args.indexer)

if __name__ == "__main__":
    main()
