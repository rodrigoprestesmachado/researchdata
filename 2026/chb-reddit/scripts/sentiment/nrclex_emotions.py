#!/usr/bin/env python3
"""
NRCLex Emotion Analysis — detects emotions in each text of the IRaMuTeQ corpus.

Reads the corpus in IRaMuTeQ format (corpus_iramutec.txt), applies NRCLex to
each document, and generates aggregated CSVs and visualizations.

Emotions detected (NRC Emotion Lexicon):
  fear, anger, anticipation, trust, surprise, sadness, disgust, joy
  positive, negative

Outputs (in --output-dir):
  emotions_per_doc.csv        — one row per document
  emotions_by_subreddit.csv   — mean per subreddit
  emotions_by_type.csv        — post vs comment mean
  plot_overall.png            — bar chart of mean emotions across the corpus
  plot_by_subreddit.png       — comparison by subreddit
  plot_by_type.png            — post vs comment
  plot_heatmap.png            — subreddit × emotion heatmap

Usage:
  python nrclex_emotions.py
  python nrclex_emotions.py --corpus data/corpus_iramutec.txt --output-dir data/nrclex_output
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from nrclex import NRCLex
from tqdm import tqdm

EMOTIONS = ["fear", "anger", "anticipation", "trust", "surprise",
            "sadness", "disgust", "joy", "positive", "negative"]

HEADER_RE = re.compile(
    r"^\*\*\*\*\s+"
    r"\*id_(\S+)\s+"
    r"\*period_(\d+)\s+"
    r"\*r_(\S+)\s+"
    r"\*type_(\S+)"
    r"(?:\s+\*tier_(\S+))?"
)


# ---------------------------------------------------------------------------
# IRaMuTeQ corpus parser
# ---------------------------------------------------------------------------

def parse_corpus(corpus_path: Path) -> list[dict]:
    """Reads the corpus and returns a list of dicts with metadata + text."""
    docs = []
    current_meta = None
    current_lines: list[str] = []

    def flush():
        if current_meta is not None:
            text = " ".join(current_lines).strip()
            docs.append({**current_meta, "text": text})

    for line in corpus_path.read_text(encoding="utf-8").splitlines():
        m = HEADER_RE.match(line)
        if m:
            flush()
            current_meta = {
                "id":        m.group(1),
                "period":    m.group(2),
                "subreddit": m.group(3),
                "type":      m.group(4),
                "tier":      m.group(5) or "",
            }
            current_lines = []
        else:
            if current_meta is not None:
                current_lines.append(line)

    flush()
    return docs


# ---------------------------------------------------------------------------
# NRCLex analysis
# ---------------------------------------------------------------------------

_NRC = None  # reusable instance (NRCLex 4.x: constructor loads the lexicon, not the text)


def _get_nrc() -> NRCLex:
    global _NRC
    if _NRC is None:
        _NRC = NRCLex()   # no argument → uses the package's default lexicon
    return _NRC


def analyze_doc(text: str) -> dict[str, float]:
    """Applies NRCLex to the text and returns each emotion's relative frequency.

    NRCLex 4.x: the constructor loads the lexicon; load_raw_text() processes the text.
    Reusing the same instance avoids reloading the lexicon on every call.
    """
    if not text.strip():
        return {e: 0.0 for e in EMOTIONS}

    nrc = _get_nrc()
    nrc.load_raw_text(text)
    freq = nrc.affect_frequencies          # dict emotion → proportion (0..1)
    return {e: freq.get(e, 0.0) for e in EMOTIONS}


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def _set_style():
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({"figure.dpi": 150})


def plot_overall(df: pd.DataFrame, out_dir: Path) -> None:
    means = df[EMOTIONS].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(means.index, means.values, color=sns.color_palette("muted", len(means)))
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_title("Mean emotions — full corpus", fontsize=14, pad=12)
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Mean relative frequency")
    ax.set_ylim(0, means.max() * 1.18)
    plt.tight_layout()
    fig.savefig(out_dir / "plot_overall.png")
    plt.close(fig)


def plot_by_subreddit(agg: pd.DataFrame, out_dir: Path) -> None:
    basic_emotions = ["joy", "trust", "anticipation", "surprise",
                      "fear", "sadness", "disgust", "anger"]
    data = agg[basic_emotions]
    fig, ax = plt.subplots(figsize=(11, 6))
    data.T.plot(kind="bar", ax=ax, width=0.75)
    ax.set_title("Mean emotions by subreddit", fontsize=14, pad=12)
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Mean relative frequency")
    ax.legend(title="Subreddit", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(out_dir / "plot_by_subreddit.png")
    plt.close(fig)


def plot_by_type(agg: pd.DataFrame, out_dir: Path) -> None:
    basic_emotions = ["joy", "trust", "anticipation", "surprise",
                      "fear", "sadness", "disgust", "anger"]
    data = agg[basic_emotions]
    x = range(len(basic_emotions))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (idx, row) in enumerate(data.iterrows()):
        offset = (i - len(data) / 2 + 0.5) * width
        ax.bar([v + offset for v in x], row.values, width, label=str(idx))
    ax.set_xticks(list(x))
    ax.set_xticklabels(basic_emotions, rotation=30, ha="right")
    ax.set_title("Mean emotions: post vs comment", fontsize=14, pad=12)
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Mean relative frequency")
    ax.legend(title="Type")
    plt.tight_layout()
    fig.savefig(out_dir / "plot_by_type.png")
    plt.close(fig)


def plot_heatmap(agg: pd.DataFrame, out_dir: Path) -> None:
    basic_emotions = ["joy", "trust", "anticipation", "surprise",
                      "fear", "sadness", "disgust", "anger"]
    data = agg[basic_emotions]
    fig, ax = plt.subplots(figsize=(10, max(4, len(data) * 0.8 + 1)))
    sns.heatmap(data, annot=True, fmt=".3f", cmap="YlOrRd",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Relative frequency"})
    ax.set_title("Heatmap: subreddit × emotion", fontsize=14, pad=12)
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Subreddit")
    plt.tight_layout()
    fig.savefig(out_dir / "plot_heatmap.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_args() -> argparse.Namespace:
    script_dir = Path(__file__).parent
    default_corpus = script_dir / "data" / "corpus_iramutec.txt"
    default_out    = script_dir / "data" / "nrclex_output"

    p = argparse.ArgumentParser(
        description="NRCLex emotion analysis over the IRaMuTeQ corpus."
    )
    p.add_argument("--corpus",     type=Path, default=default_corpus,
                   help="Path to corpus_iramutec.txt")
    p.add_argument("--output-dir", type=Path, default=default_out,
                   help="Output directory for CSVs and plots")
    p.add_argument("--no-plots",   action="store_true",
                   help="Skips plot generation")
    return p.parse_args()


def main() -> None:
    args = build_args()

    if not args.corpus.exists():
        print(f"[error] Corpus not found: {args.corpus}", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- parse corpus ----
    print(f"Reading corpus: {args.corpus}")
    docs = parse_corpus(args.corpus)
    print(f"  {len(docs)} documents found.")

    # ---- analyze emotions ----
    print("Analyzing emotions with NRCLex…")
    rows = []
    for doc in tqdm(docs, unit="doc"):
        scores = analyze_doc(doc["text"])
        rows.append({**{k: doc[k] for k in ("id", "period", "subreddit", "type")},
                     **scores,
                     "word_count": len(doc["text"].split())})

    df = pd.DataFrame(rows)

    # ---- export per-document CSV ----
    per_doc_path = args.output_dir / "emotions_per_doc.csv"
    df.to_csv(per_doc_path, index=False)
    print(f"  Saved: {per_doc_path}")

    # ---- aggregations ----
    agg_sub  = df.groupby("subreddit")[EMOTIONS].mean().round(4)
    agg_type = df.groupby("type")[EMOTIONS].mean().round(4)

    sub_path  = args.output_dir / "emotions_by_subreddit.csv"
    type_path = args.output_dir / "emotions_by_type.csv"
    agg_sub.to_csv(sub_path)
    agg_type.to_csv(type_path)
    print(f"  Saved: {sub_path}")
    print(f"  Saved: {type_path}")

    # ---- terminal summary ----
    print("\n--- Mean emotions — full corpus ---")
    overall = df[EMOTIONS].mean().round(4).sort_values(ascending=False)
    for emotion, val in overall.items():
        print(f"  {emotion:<15} {val:.4f}")

    print("\n--- Mean emotions by subreddit ---")
    print(agg_sub.to_string())

    print("\n--- Mean emotions by type (post/comment) ---")
    print(agg_type.to_string())

    # ---- plots ----
    if not args.no_plots:
        print("\nGenerating plots…")
        _set_style()
        plot_overall(df, args.output_dir)
        print(f"  {args.output_dir}/plot_overall.png")

        if len(agg_sub) > 1:
            plot_by_subreddit(agg_sub, args.output_dir)
            print(f"  {args.output_dir}/plot_by_subreddit.png")
            plot_heatmap(agg_sub, args.output_dir)
            print(f"  {args.output_dir}/plot_heatmap.png")

        if len(agg_type) > 1:
            plot_by_type(agg_type, args.output_dir)
            print(f"  {args.output_dir}/plot_by_type.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
