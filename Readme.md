# Dynamic Image Retrieval System on Caltech-256

This project implements a **scalable, dynamically expanding image retrieval system** on the **Caltech-256** dataset. It combines:

- Strong **visual feature extractors** (CLIP ViT-B/32, ResNet-50, ViT-B/16)
- Fast **similarity search backends** (FAISS Flat, FAISS HNSW, Annoy)
- **ID-mapped indexing** for safe merging and updates
- **Dynamic web expansion** (fetch missing concepts from the internet)
- A full **Streamlit UI** for search, dataset growth, and management

The system starts from the **Checkpoint 2** embeddings (NPZ files) and continuously improves itself as users issue new queries.

---

## 📁 1. Project Structure

Lets take the basic folder structure example is:

```text
C:\Users\punna\Downloads\IMRS
The key layout is:
IMRS/
├─ 256_ObjectCategories/              
├─ 256_ObjectCategories_proc/         # Preprocessed dataset (Checkpoint 1)
├─ retrieval_framework.py             # CP2 benchmarking/embeddings
├─ app_streamlit.py                   # Final dynamic retrieval app
├─ runs/
│  ├─ ckpt2/
│  │  └─ cache/                       # Checkpoint 2 NPZ embeddings: emb_*.npz
│  └─ streamlit_state/                # App state: dynamic NPZs, merged NPZs, index snapshots
└─ .venv/                             # Python virtual environment
Important paths inside the app:

Preprocessed data root
C:/Users/punna/Downloads/IMRS/256_ObjectCategories_proc

Checkpoint 2 NPZ cache
C:/Users/punna/Downloads/IMRS/runs/ckpt2/cache

Streamlit state
runs/streamlit_state

⚙️ 2. Environment Setup
From the IMRS folder:

use powershell and run:
cd C:\Users\punna\Downloads\IMRS

# (If not already done) create venv
python -m venv .venv

# Activate venv
.\.venv\Scripts\Activate

# Upgrade pip
python -m pip install --upgrade pip

# Install core libraries
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install faiss-cpu annoy streamlit pillow numpy pandas imagehash
pip install duckduckgo-search requests python-docx
pip install git+https://github.com/openai/CLIP.git
You may already have most of these from Checkpoint 2; just fill in the missing ones.

🧪 3. Re-run Checkpoint Benchmarks
If you want to regenerate embeddings / NPZs:

Use powershell to run:

cd C:\Users\punna\Downloads\IMRS
.\.venv\Scripts\Activate

python retrieval_framework.py bench `
  --data_root "C:/Users/punna/Downloads/IMRS/256_ObjectCategories_proc" `
  --out_dir runs/ckpt2 `
  --backbones clip_vit_b32 resnet50 vit_b16 `
  --indexers flat_ip hnsw ivf_pq annoy `
  --topk 50 --sample_queries 1500
This writes NPZ embedding caches to:
runs/ckpt2/cache/emb_*.npz
The Streamlit app will reuse and merge these automatically.

🚀 4. Running the Streamlit App
Open powershell:
cd C:\Users\punna\Downloads\IMRS
.\.venv\Scripts\Activate

streamlit run app_streamlit.py
Then open the browser (usually auto-opens) at:

http://localhost:8501

The app has:

A Search (Image/Text) tab

A Batch Expansion tab

A Manage Dynamic Images tab

Sidebar controls for backbone, indexer, NPZ merging, and snapshots.

🧠 5. How the System Works (High-Level)
Each image is embedded into a vector using the selected backbone.

Every image is assigned a stable signed 64-bit ID derived from its normalized file path.

Embeddings + IDs are stored in:

FAISS (IndexIDMap2) or

Annoy (with a separate position→ID mapping).

A central catalog maps IDs to image paths:
catalog[id] = {"path": ".../image.jpg"}
When you query:

The index returns top-k IDs and scores.

The app looks up catalog[id]["path"] and displays those images.

This design is robust to:

Merging multiple NPZ embedding files

Dynamic dataset growth

Deletion of dynamic images

🧩 6. Sidebar Configuration (Step-by-Step)
6.1 Dataset Root
Default:
C:/Users/punna/Downloads/IMRS/256_ObjectCategories_proc

Change only if your preprocessed dataset moved.

6.2 Device
Shows cuda if a GPU is available, else cpu.

Determined automatically.

6.3 Backbone
Choose a feature extractor:

CLIP ViT-B/32

Required for text-to-image search.

ResNet-50

ViT-B/16 (torchvision)

6.4 Indexer
Similarity backend:

FAISS Flat – exact cosine/IP nearest neighbors.

FAISS HNSW – graph-based approximate search.

Annoy – tree-based approximate search on CPU.

6.5 Top-K
Slider to control how many results to retrieve (e.g., 12, 20, etc.).

6.6 Index Control
Rebuild Index (full)

Rebuilds FAISS/Annoy index from current embeddings + IDs.

💾 Save Snapshot

Saves index + catalog + IDs into runs/streamlit_state.

📥 Load Snapshot

Loads previously saved index + catalog + IDs.

6.7 NPZ Auto-Merge
Auto-load CP2 NPZs and merge dynamic NPZs

When checked, the app:

Reads NPZs from runs/ckpt2/cache (Checkpoint 2).

Merges them with dynamic NPZs from runs/streamlit_state.

Drops vectors with missing paths.

De-duplicates by ID.

Produces a merged NPZ:

runs/streamlit_state/emb_merged_<backbone>.npz

🔄 Rescan & Merge NPZs

Manually triggers the NPZ merging process.

6.8 Use Specific NPZ (Override)
Optionally specify a single NPZ file path to use as the embedding source for the session.

🔍 7. Tab 1 – Search (Image/Text)
The Search tab has two modes:

7.1 Text → Image (CLIP-only)
Requires CLIP ViT-B/32 as the selected backbone.

Go to Search (Image/Text) → Text → Image (CLIP).

In the input box, type a query such as:

a helicopter in the sky

a yellow school bus

a red sports car on a racetrack

Press Enter.

The app:

Encodes the text with CLIP.

Searches the index for nearest embeddings.

Displays top-k results as image thumbnails with scores.

Automatic dynamic expansion:
If similarity scores indicate weak coverage (low values, few strong matches), the app:

Warns that coverage is weak.

Fetches a small batch of new images from the web using that text.

Deduplicates with pHash.

Saves them under dynamic_<slug> folders (JPEG, max side 256 px).

Embeds them and appends them to the database.

Rebuilds the index and re-runs the search.

Shows improved results.

7.2 Image → Image
Go to Search (Image/Text) → Image → Image.

Upload a query image (.jpg, .jpeg, .png, etc.).

The app:

Displays the query image.

Embeds it using the currently selected backbone.

Performs nearest-neighbor search.

Displays the most similar images and their scores.

If coverage is weak:

If using CLIP and class names are available, the system may infer a text label (e.g., a photo of helicopter) and run a web expansion using that inferred term.

Otherwise, it may use folder/class names from the nearest hits.

🌱 8. Tab 2 – Batch Expansion
Use Batch Expansion to add many concepts at once.

Go to the Batch Expansion tab.

Provide terms:

Textarea: type one term per line, e.g.:
minecraft sword
rubik cube
mario character
CSV upload: upload a CSV with at least one column (preferably named term).

Configure parameters:

Images per term – e.g., 20.

Max side (px) – e.g., 256.

JPEG quality – e.g., 80–90.

pHash threshold – e.g., 4 (lower = stricter dedupe).

Click Run Batch Expansion.

The app:

Fetches images for each term (DuckDuckGo + Wikipedia fallback).

Filters duplicates via pHash.

Saves each image under dynamic_<term_slug>/.

Embeds them using the selected backbone.

Assigns IDs and updates the catalog.

Rebuilds the index once at the end.

Saves a dynamic NPZ:

runs/streamlit_state/dyn_emb_<term>_<backbone>.npz

A summary shows how many images were successfully added and how many terms failed.

🗂️ 9. Tab 3 – Manage Dynamic Images
Use this tab to clean up the dynamically downloaded data.

Click Manage Dynamic Images.

The app scans data_root for dynamic_* folders.

All dynamic images are listed.

Select any images you want to remove.

Click 🗑️ Remove selected.

Then the app:

Deletes those files from disk.

Re-lists ALL dataset images (base + dynamic).

Re-embeds them with the selected backbone.

Rebuilds the index.

Saves an updated NPZ snapshot in runs/streamlit_state.

This helps keep the dynamic dataset high quality.

🧬 10. NPZ Reuse & Safe Merging
The app automatically merges embeddings from:

Checkpoint 2 NPZs in runs/ckpt2/cache

Dynamic NPZs in runs/streamlit_state

Merge process:

Load all relevant NPZs for the chosen backbone.

Require that each embedding has a valid image path.

Convert all paths to signed 64-bit IDs.

Remove duplicates by ID (first occurrence wins).

Save merged file:
runs/streamlit_state/emb_merged_<backbone>.npz
This merged NPZ becomes the active embedding source and keeps the system consistent across sessions.

💾 11. Snapshots
Save Snapshot
Saves:

Index (FAISS or Annoy)

Catalog (catalog.pkl)

ID array (ids.npy)

Load Snapshot
Restores the index and ID mapping.

Recomputes embeddings if needed, to keep everything in sync.

Use snapshots to:

Resume from a specific state.

Avoid recomputing everything after a restart.

🛠️ 12. Troubleshooting Tips
Path changes?
Update the constants at the top of app_streamlit.py:

DEFAULT_DATA_ROOT

CP2_NPZ_DIR

APP_STATE_DIR

ID or path mismatch errors?
Delete cached state:

runs/streamlit_state/ids.npy

runs/streamlit_state/catalog.pkl

Any stale emb_*.npz if needed
Then rerun the app and let it re-merge NPZs.

Rate limits when downloading images?
The code already does backoff and uses Wikipedia fallback. For very aggressive batch expansions, some terms may still fail—these are reported in the UI.

✅ 13. Summary
This project delivers:

A research-grade embedding and retrieval pipeline (Checkpoint 2).

A production-style dynamic retrieval system via Streamlit:

Strong encoders (CLIP / ResNet / ViT)

FAISS & Annoy indexing

ID-mapped, merge-safe embeddings

Dynamic web expansion and cleanup tools

NPZ merging and snapshotting

To use everything:

Activate the virtual environment.

Run:

powershell
Copy code
streamlit run app_streamlit.py
Explore via:

Search (Image/Text) tab

Batch Expansion tab

Manage Dynamic Images tab

Save snapshots and merged NPZs for later reuse.

Enjoy exploring and extending your dynamically growing Caltech-256 image retrieval system 🚀
