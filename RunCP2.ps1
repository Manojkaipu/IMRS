# RunCP2.ps1
$PROJECT   = "C:\Users\punna\Downloads\IMRS"
$DATA_RAW  = "C:\Users\punna\Downloads\IMRS\256_ObjectCategories"
$DATA_PROC = "C:\Users\punna\Downloads\IMRS\256_ObjectCategories_proc"
$RUNS_OUT  = "runs\ckpt2"
$RUNS_FT   = "runs\ckpt2_ft"

Set-Location $PROJECT

if (-not (Test-Path ".\.venv")) { python -m venv .venv }
. ".\.venv\Scripts\Activate.ps1"

pip install --upgrade pip
pip install -r requirements.txt

$manifest = Join-Path $DATA_PROC "manifest.csv"
if (-not (Test-Path $manifest)) {
  python retrieval_framework.py preprocess `
    --raw_root "C:/Users/punna/Downloads/IMRS/256_ObjectCategories" `
    --proc_root "C:/Users/punna/Downloads/IMRS/256_ObjectCategories_proc" `
    --img_size 224 --phash_threshold 4 `
    --use_clahe --color_norm
}

python retrieval_framework.py bench `
  --data_root "C:/Users/punna/Downloads/IMRS/256_ObjectCategories_proc" `
  --out_dir $RUNS_OUT `
  --backbones clip_vit_b32 resnet50 vit_b16 `
  --indexers flat_ip hnsw ivf_pq annoy `
  --topk 50 --sample_queries 1500

python retrieval_framework.py finetune `
  --data_root "C:/Users/punna/Downloads/IMRS/256_ObjectCategories_proc" `
  --out_dir $RUNS_FT `
  --backbone clip_vit_b32 --epochs 5 --batch_size 64 --lr 1e-3 --margin 0.2

python retrieval_framework.py bench `
  --data_root "C:/Users/punna/Downloads/IMRS/256_ObjectCategories_proc" `
  --out_dir $RUNS_FT `
  --backbones clip_vit_b32_ft `
  --indexers flat_ip hnsw annoy `
  --topk 50 --sample_queries 1500
