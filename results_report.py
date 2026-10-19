"""
results_report.py
Reads CP2 outputs and prints a concise leaderboard + makes simple charts.

Install:
  pip install pandas matplotlib

Run:
  python results_report.py --runs runs/ckpt2
"""

import argparse, os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="Path to runs/ckpt2 or runs/ckpt2_ft")
    args = ap.parse_args()

    all_csv = os.path.join(args.runs, "all_experiments.csv")
    if not os.path.exists(all_csv):
        raise FileNotFoundError(f"{all_csv} not found. Run the bench step first.")

    df = pd.read_csv(all_csv)
    # Sort by mean_ap@K (desc), then by qps (desc)
    df = df.sort_values(by=["mean_ap@K", "qps"], ascending=[False, False])
    print("\n=== Leaderboard (top 10) ===")
    print(df.head(10)[["backbone","indexer","topk","sample_queries","mean_ap@K","mean_ndcg@K","qps"]])

    # Simple bar chart for mAP
    plt.figure()
    labels = [f"{b}-{ix}" for b,ix in zip(df["backbone"], df["indexer"])]
    plt.bar(labels, df["mean_ap@K"])
    plt.xticks(rotation=45, ha="right")
    plt.title("mAP@K by Method")
    plt.tight_layout()
    out_png = os.path.join(args.runs, "map_leaderboard.png")
    plt.savefig(out_png, dpi=150)
    print(f"\nSaved: {out_png}")

if __name__ == "__main__":
    main()
