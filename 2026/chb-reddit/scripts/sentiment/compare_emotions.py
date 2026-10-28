#!/usr/bin/env python3
"""
NRCLex vs RoBERTa comparison — generates visualizations comparing the two
emotion-analysis methods applied to the IRaMuTeQ corpus.

Emotions in common (basis for the comparison):
  anger, disgust, fear, joy, sadness, surprise

Outputs (in --output-dir):
  comparison_summary.csv          — both methods' means in a single table
  plot_comparison_overall.png     — side-by-side bars of the common emotions
  plot_correlation_heatmap.png    — correlation between the two methods, per document

Prerequisite: run nrclex_emotions.py and roberta_emotions.py first.

Usage:
  python compare_emotions.py
  python compare_emotions.py \
      --nrclex  data/nrclex_output/emotions_per_doc.csv \
      --roberta data/roberta_output/emotions_per_doc.csv \
      --output-dir data/compare_output
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Emotions present in both methods
COMMON_EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def _set_style():
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({"figure.dpi": 150})


def plot_comparison_overall(nrc_means: pd.Series, rob_means: pd.Series,
                             out_dir: Path) -> None:
    """Side-by-side bars: NRCLex vs RoBERTa for each common emotion."""
    x = np.arange(len(COMMON_EMOTIONS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    bars1 = ax.bar(x - width / 2, nrc_means.values, width, label="NRCLex",
                   color=sns.color_palette("muted")[0])
    bars2 = ax.bar(x + width / 2, rob_means.values, width, label="RoBERTa",
                   color=sns.color_palette("muted")[1])
    ax.bar_label(bars1, fmt="%.3f", padding=2, fontsize=7)
    ax.bar_label(bars2, fmt="%.3f", padding=2, fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(COMMON_EMOTIONS)
    ax.set_title("NRCLex vs RoBERTa comparison — mean emotions", fontsize=14, pad=12)
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Mean score / relative frequency")
    ax.legend()
    max_val = max(nrc_means.max(), rob_means.max())
    ax.set_ylim(0, max_val * 1.22)
    plt.tight_layout()
    fig.savefig(out_dir / "plot_comparison_overall.png")
    plt.close(fig)


def plot_correlation_heatmap(df_nrc: pd.DataFrame, df_rob: pd.DataFrame,
                              out_dir: Path) -> None:
    """Pearson correlation heatmap between the two methods, per emotion."""
    # align by document id
    merged = df_nrc[["id"] + COMMON_EMOTIONS].merge(
        df_rob[["id"] + COMMON_EMOTIONS],
        on="id",
        suffixes=("_nrc", "_rob"),
    )

    corr_data = {}
    for e in COMMON_EMOTIONS:
        col_nrc = f"{e}_nrc"
        col_rob = f"{e}_rob"
        if col_nrc in merged.columns and col_rob in merged.columns:
            corr = merged[col_nrc].corr(merged[col_rob])
            corr_data[e] = corr

    corr_series = pd.Series(corr_data, name="Pearson r").round(4)

    # display as a 1×N heatmap
    fig, ax = plt.subplots(figsize=(10, 2.5))
    data = corr_series.values.reshape(1, -1)
    sns.heatmap(data, annot=True, fmt=".3f", cmap="coolwarm",
                xticklabels=corr_series.index, yticklabels=["Correlation"],
                vmin=-1, vmax=1, linewidths=0.5, ax=ax,
                cbar_kws={"label": "Pearson r"})
    ax.set_title("NRCLex × RoBERTa correlation per emotion (per document)",
                 fontsize=13, pad=10)
    plt.tight_layout()
    fig.savefig(out_dir / "plot_correlation_heatmap.png")
    plt.close(fig)

    return corr_series


def plot_scatter_grid(df_nrc: pd.DataFrame, df_rob: pd.DataFrame,
                      out_dir: Path) -> None:
    """NRCLex × RoBERTa scatter plots for each common emotion."""
    merged = df_nrc[["id"] + COMMON_EMOTIONS].merge(
        df_rob[["id"] + COMMON_EMOTIONS],
        on="id",
        suffixes=("_nrc", "_rob"),
    )

    n = len(COMMON_EMOTIONS)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    axes = axes.flatten()

    for i, e in enumerate(COMMON_EMOTIONS):
        ax = axes[i]
        x = merged[f"{e}_nrc"]
        y = merged[f"{e}_rob"]
        ax.scatter(x, y, alpha=0.3, s=10, color=sns.color_palette("muted")[i % 8])
        # trend line
        if x.std() > 0 and y.std() > 0:
            m, b = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, m * x_line + b, color="red", linewidth=1)
        corr = x.corr(y)
        ax.set_title(f"{e}  (r={corr:.3f})", fontsize=10)
        ax.set_xlabel("NRCLex")
        ax.set_ylabel("RoBERTa")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("NRCLex × RoBERTa — per-emotion scatter", fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(out_dir / "plot_scatter_grid.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_args() -> argparse.Namespace:
    script_dir = Path(__file__).parent
    default_nrc   = script_dir / "data" / "nrclex_output"  / "emotions_per_doc.csv"
    default_rob   = script_dir / "data" / "roberta_output" / "emotions_per_doc.csv"
    default_out   = script_dir / "data" / "compare_output"

    p = argparse.ArgumentParser(
        description="NRCLex vs RoBERTa comparison over the IRaMuTeQ corpus."
    )
    p.add_argument("--nrclex",     type=Path, default=default_nrc,
                   help="NRCLex emotions_per_doc CSV")
    p.add_argument("--roberta",    type=Path, default=default_rob,
                   help="RoBERTa emotions_per_doc CSV")
    p.add_argument("--output-dir", type=Path, default=default_out,
                   help="Output directory")
    return p.parse_args()


def main() -> None:
    args = build_args()

    for path, label in [(args.nrclex, "NRCLex"), (args.roberta, "RoBERTa")]:
        if not path.exists():
            print(f"[error] {label} CSV not found: {path}", file=sys.stderr)
            sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df_nrc = pd.read_csv(args.nrclex)
    df_rob = pd.read_csv(args.roberta)

    print(f"NRCLex:  {len(df_nrc)} documents")
    print(f"RoBERTa: {len(df_rob)} documents")

    # ---- means of the common emotions ----
    nrc_means = df_nrc[COMMON_EMOTIONS].mean().round(4)
    rob_means = df_rob[COMMON_EMOTIONS].mean().round(4)

    # ---- summary table ----
    summary = pd.DataFrame({"NRCLex": nrc_means, "RoBERTa": rob_means})
    summary["difference"] = (summary["RoBERTa"] - summary["NRCLex"]).round(4)
    summary_path = args.output_dir / "comparison_summary.csv"
    summary.to_csv(summary_path)
    print(f"\n--- Mean comparison (common emotions) ---")
    print(summary.to_string())
    print(f"\n  Saved: {summary_path}")

    # ---- plots ----
    print("\nGenerating plots…")
    _set_style()

    plot_comparison_overall(nrc_means, rob_means, args.output_dir)
    print(f"  {args.output_dir}/plot_comparison_overall.png")

    corr = plot_correlation_heatmap(df_nrc, df_rob, args.output_dir)
    print(f"  {args.output_dir}/plot_correlation_heatmap.png")

    plot_scatter_grid(df_nrc, df_rob, args.output_dir)
    print(f"  {args.output_dir}/plot_scatter_grid.png")

    print("\n--- Pearson correlations per emotion ---")
    for e, r in corr.items():
        print(f"  {e:<12} r = {r:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
