"""Turn benchmark CSVs into a markdown leaderboard and plots."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

SHOW = ["backbone", "indexer", "eval_classes", "mAP@10", "P@10", "R@10", "nDCG@10", "Hit@1",
        "ann_recall@10", "latency_ms_p50", "qps_batch", "build_s", "index_mb"]


def leaderboard_markdown(df: pd.DataFrame) -> str:
    cols = [c for c in SHOW if c in df.columns]
    df = df[cols].sort_values(["eval_classes", "mAP@10"], ascending=[True, False]) if "mAP@10" in df else df[cols]
    fmt = {c: "{:.3f}" for c in ("mAP@10", "P@10", "R@10", "nDCG@10", "Hit@1", "ann_recall@10")}
    fmt.update({"latency_ms_p50": "{:.2f}", "qps_batch": "{:,.0f}", "build_s": "{:.2f}", "index_mb": "{:.1f}"})
    out = df.copy()
    for c, f in fmt.items():
        if c in out:
            out[c] = out[c].map(lambda v, f=f: "–" if pd.isna(v) else f.format(v))
    return markdown_table(out)


def markdown_table(df: pd.DataFrame) -> str:
    """Minimal GitHub-markdown table (avoids a `tabulate` dependency)."""
    header = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    return header + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False))


def plot_map(df: pd.DataFrame, out_png: Path) -> None:
    import matplotlib.pyplot as plt

    flat = df[df["indexer"] == "flat"].sort_values("mAP@10")
    if flat.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(flat) + 1.2))
    labels = [f"{b} ({e})" for b, e in zip(flat["backbone"], flat["eval_classes"])]
    ax.barh(labels, flat["mAP@10"], color="#4C72B0")
    for i, v in enumerate(flat["mAP@10"]):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("mAP@10 (exact search)")
    ax.set_xlim(0, min(1.0, flat["mAP@10"].max() + 0.1))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_sweep(df: pd.DataFrame, out_png: Path, k: int = 10) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for indexer, g in df.groupby("indexer"):
        g = g.sort_values("value")
        ax.plot(g[f"ann_recall@{k}"], g["qps_batch"], marker="o", label=indexer)
    ax.set_xlabel(f"neighbour recall@{k} vs exact search")
    ax.set_ylabel("queries / second (batched, log)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title(f"Speed / accuracy trade-off ({df['backbone'].iloc[0]})")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def build_report(runs_dir: Path) -> str:
    runs_dir = Path(runs_dir)
    parts = []
    bench = runs_dir / "bench" / "results.csv"
    if bench.exists():
        df = pd.read_csv(bench)
        parts.append("## Retrieval benchmark\n\n" + leaderboard_markdown(df))
        plot_map(df, runs_dir / "bench" / "map_by_backbone.png")
    for sweep in sorted((runs_dir / "bench").glob("sweep_*.csv")):
        df = pd.read_csv(sweep)
        plot_sweep(df, sweep.with_suffix(".png"))
        parts.append(f"## ANN sweep ({sweep.stem})\n\n![]({sweep.with_suffix('.png').name})")
    calib = runs_dir / "bench" / "calibration.csv"
    if calib.exists():
        parts.append("## Expansion trigger calibration\n\n" + markdown_table(pd.read_csv(calib).round(3)))
    text = "\n\n".join(parts) or "No results yet - run `imrs bench` first."
    (runs_dir / "bench").mkdir(parents=True, exist_ok=True)
    (runs_dir / "bench" / "REPORT.md").write_text(text, encoding="utf-8")
    return text
