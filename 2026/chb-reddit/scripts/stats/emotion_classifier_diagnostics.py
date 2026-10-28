#!/usr/bin/env python3
"""
Emotion classifier diagnostics — empirically checks whether "disgust" and
"neutral" (DistilRoBERTa) reflect the intended construct or a classifier
artifact on critical/argumentative text (a register typical of Reddit).

Four checks, all based on already-existing data or reproducible without
re-running the transformer model:

  1. disgust×anger correlation (RoBERTa, per document) — a high correlation
     is consistent with the classifier confusing the two categories in
     complaint/critical language.
  2. Comparison against NRCLex (lexicon-based, non-neural) — mean difference
     and per-document correlation between RoBERTa and NRCLex for the shared
     emotions (uses the output of nrclex_emotions.py + compare_emotions.py).
  3. Lexical face validity for "disgust" — among the documents with the
     highest RoBERTa disgust score, how many contain a literal repulsion
     lexical marker vs. how many are merely critical/argumentative without
     that vocabulary.
  4. Face validity for "neutral" — compares length and argumentation/opinion
     markers between high- vs. low-neutral-score documents.

Outputs (in --output-dir):
  diagnostics_summary.csv        — one row per check, with the relevant number
  disgust_top_examples.csv       — the N highest-disgust documents (RoBERTa),
                                    with lexical marker flag (yes/no) and truncated text
  neutral_bins_comparison.csv    — descriptive statistics per neutral bin

Usage:
  python emotion_classifier_diagnostics.py
  python emotion_classifier_diagnostics.py --top-n 30
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr

HEADER_RE = re.compile(
    r"^\*\*\*\*\s+"
    r"\*id_(\S+)\s+"
    r"\*period_(\d+)\s+"
    r"\*r_(\S+)\s+"
    r"\*type_(\S+)"
    r"(?:\s+\*tier_(\S+))?"
)

# Literal lexical markers of repulsion/disgust (not just generic criticism).
DISGUST_MARKERS = [
    r"\bdisgust\w*\b", r"\bgross\w*\b", r"\bnasty\b", r"\bvomit\w*\b",
    r"\bsick(?:ening)?\b", r"\brepulsiv\w*\b", r"\brevolt\w*\b",
    r"\byuck\w*\b", r"\bickiest\b", r"\bicky\b", r"\bfilthy\b",
    r"\bnauseat\w*\b", r"\bcringe\w*\b",
]
_DISGUST_RE = re.compile("|".join(DISGUST_MARKERS), re.IGNORECASE)

# Argumentative/deliberative (non-affective) tone markers — a proxy for
# checking whether "neutral" is capturing an argumentative register rather
# than a genuine absence of affect.
ARGUMENTATIVE_MARKERS = [
    r"\bshould\b", r"\bmust\b", r"\bneed to\b", r"\bI think\b",
    r"\bI believe\b", r"\bin my opinion\b", r"\bargue\w*\b", r"\?",
    r"\btherefore\b", r"\bhowever\b", r"\bbecause\b",
]
_ARGUMENTATIVE_RE = re.compile("|".join(ARGUMENTATIVE_MARKERS), re.IGNORECASE)


def parse_corpus_texts(corpus_path: Path) -> dict[str, str]:
    """Reads the IRaMuTeQ corpus (with *id_) and returns {id: text}."""
    texts: dict[str, str] = {}
    current_id = None
    current_lines: list[str] = []

    def flush():
        if current_id is not None:
            texts[current_id] = " ".join(current_lines).strip()

    for line in corpus_path.read_text(encoding="utf-8").splitlines():
        m = HEADER_RE.match(line)
        if m:
            flush()
            current_id = m.group(1)
            current_lines = []
        else:
            if current_id is not None:
                current_lines.append(line)
    flush()
    return texts


def check_disgust_anger_correlation(df: pd.DataFrame) -> dict:
    r, p = pearsonr(df["disgust"], df["anger"])
    return {
        "check": "disgust_anger_correlation_roberta",
        "metric": "pearson_r",
        "value": round(float(r), 4),
        "p_value": p,
        "n": len(df),
        "note": "Per-document correlation between disgust and anger (RoBERTa). "
                "A high value is consistent with the classifier confusing the "
                "two categories in critical/complaint language.",
    }


def check_nrclex_comparison(nrclex_csv: Path, roberta_csv: Path) -> list[dict]:
    if not nrclex_csv.exists() or not roberta_csv.exists():
        print(f"[warn] Skipping NRCLex comparison: files not found "
              f"({nrclex_csv}, {roberta_csv})", file=sys.stderr)
        return []

    common = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
    df_nrc = pd.read_csv(nrclex_csv)
    df_rob = pd.read_csv(roberta_csv)
    merged = df_nrc[["id"] + common].merge(
        df_rob[["id"] + common], on="id", suffixes=("_nrc", "_rob"),
    )

    rows = []
    for e in common:
        nrc_mean = df_nrc[e].mean()
        rob_mean = df_rob[e].mean()
        r, p = pearsonr(merged[f"{e}_nrc"], merged[f"{e}_rob"])
        rows.append({
            "check": f"nrclex_vs_roberta_{e}",
            "metric": "mean_diff_rob_minus_nrc",
            "value": round(float(rob_mean - nrc_mean), 4),
            "p_value": None,
            "n": len(merged),
            "note": f"NRCLex mean={nrc_mean:.4f}, RoBERTa mean={rob_mean:.4f}, "
                    f"ratio={rob_mean / nrc_mean:.2f}x, pearson_r={r:.4f} "
                    f"(p={p:.2e}). A large difference combined with low "
                    f"correlation suggests RoBERTa is capturing something "
                    f"beyond the lexicon of {e}.",
        })
    return rows


def check_disgust_face_validity(df: pd.DataFrame, texts: dict[str, str],
                                 top_n: int, output_dir: Path) -> dict:
    top = df.nlargest(top_n, "disgust").copy()
    top["text"] = top["id"].map(texts).fillna("")
    top["has_disgust_marker"] = top["text"].apply(
        lambda t: bool(_DISGUST_RE.search(t))
    )
    n_with_marker = int(top["has_disgust_marker"].sum())
    pct = 100 * n_with_marker / len(top) if len(top) else 0.0

    out = top[["id", "subreddit", "type", "disgust", "anger",
               "has_disgust_marker", "text"]].copy()
    out["text"] = out["text"].str.slice(0, 300)
    out.to_csv(output_dir / "disgust_top_examples.csv", index=False)

    return {
        "check": "disgust_lexical_face_validity",
        "metric": "pct_top_docs_with_literal_marker",
        "value": round(pct, 1),
        "p_value": None,
        "n": len(top),
        "note": f"Of the {len(top)} documents with the highest disgust score "
                f"(RoBERTa), {n_with_marker} ({pct:.1f}%) contain a literal "
                f"repulsion/disgust lexical marker; the remaining "
                f"({len(top) - n_with_marker}, {100 - pct:.1f}%) are critical/"
                f"argumentative without that vocabulary.",
    }


def check_neutral_bins(df: pd.DataFrame, texts: dict[str, str],
                        output_dir: Path) -> dict:
    df = df.copy()
    df["text"] = df["id"].map(texts).fillna("")
    q_low, q_high = df["neutral"].quantile([0.25, 0.75])
    low_bin = df[df["neutral"] <= q_low]
    high_bin = df[df["neutral"] >= q_high]

    def _argumentative_rate(bin_df: pd.DataFrame) -> float:
        hits = bin_df["text"].apply(lambda t: bool(_ARGUMENTATIVE_RE.search(t)))
        return 100 * hits.mean() if len(bin_df) else 0.0

    low_rate = _argumentative_rate(low_bin)
    high_rate = _argumentative_rate(high_bin)

    summary = pd.DataFrame({
        "bin": ["low_neutral (Q1)", "high_neutral (Q4)"],
        "n": [len(low_bin), len(high_bin)],
        "mean_neutral": [low_bin["neutral"].mean(), high_bin["neutral"].mean()],
        "mean_word_count": [low_bin["word_count"].mean(), high_bin["word_count"].mean()],
        "pct_with_argumentative_marker": [low_rate, high_rate],
    })
    summary.to_csv(output_dir / "neutral_bins_comparison.csv", index=False)

    return {
        "check": "neutral_argumentative_marker_rate",
        "metric": "pct_high_neutral_docs_with_argumentative_marker",
        "value": round(high_rate, 1),
        "p_value": None,
        "n": len(high_bin),
        "note": f"{high_rate:.1f}% of the high-neutral documents (top "
                f"quartile, n={len(high_bin)}) contain argumentative/"
                f"deliberative tone markers (should, I think, ?, however...), "
                f"vs. {low_rate:.1f}% in the low-neutral ones (n={len(low_bin)}). "
                f"This is consistent with 'neutral' partly capturing Reddit's "
                f"argumentative register rather than a genuine absence of affect.",
    }


def build_args() -> argparse.Namespace:
    script_dir = Path(__file__).parent
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--roberta-per-doc", type=Path,
                   default=script_dir / "corpus" / "roberta_output" / "emotions_per_doc.csv")
    p.add_argument("--nrclex-per-doc", type=Path,
                   default=script_dir / "corpus" / "nrclex_output" / "emotions_per_doc.csv")
    p.add_argument("--corpus", type=Path,
                   default=script_dir / "corpus" / "year_2025_2026_iramutec_with_id.txt")
    p.add_argument("--output-dir", type=Path,
                   default=script_dir / "corpus" / "diagnostics_output")
    p.add_argument("--top-n", type=int, default=30,
                   help="Number of highest-disgust documents to inspect (default: 30)")
    return p.parse_args()


def main() -> None:
    args = build_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.roberta_per_doc.exists():
        print(f"[error] RoBERTa CSV not found: {args.roberta_per_doc}", file=sys.stderr)
        sys.exit(1)
    if not args.corpus.exists():
        print(f"[error] Corpus not found: {args.corpus}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading RoBERTa scores: {args.roberta_per_doc}")
    df_rob = pd.read_csv(args.roberta_per_doc)
    df_rob["id"] = df_rob["id"].astype(str)

    print(f"Reading corpus texts: {args.corpus}")
    texts = parse_corpus_texts(args.corpus)
    print(f"  {len(texts)} texts loaded.")

    rows: list[dict] = []
    rows.append(check_disgust_anger_correlation(df_rob))
    rows.extend(check_nrclex_comparison(args.nrclex_per_doc, args.roberta_per_doc))
    rows.append(check_disgust_face_validity(df_rob, texts, args.top_n, args.output_dir))
    rows.append(check_neutral_bins(df_rob, texts, args.output_dir))

    summary = pd.DataFrame(rows)
    summary_path = args.output_dir / "diagnostics_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"\n--- Diagnostics summary ---")
    for row in rows:
        print(f"\n[{row['check']}]")
        print(f"  {row['metric']} = {row['value']}"
              + (f" (p={row['p_value']:.2e})" if isinstance(row.get("p_value"), float) else ""))
        print(f"  n={row['n']}")
        print(f"  {row['note']}")

    print(f"\nSaved: {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
