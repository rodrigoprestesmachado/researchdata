#!/usr/bin/env python3
"""
512-token truncation count — quantifies the content loss caused by
DistilRoBERTa's (roberta_emotions.py) 512-token limit, both at the
DOCUMENT level (how many documents are affected) and at the WORD level
(how much content is actually lost in the affected documents).

Uses only the model's *fast* tokenizer (no model weights loaded, no
GPU/torch) and the token→word alignment (``encoding.word_ids()``) to
determine, per document, which words fall entirely beyond the 512th token.

Metrics (see the ``aggregate`` docstring):
  - Document level: N and % of documents whose total token count exceeds 512.
  - Word level: % of the corpus's total words lost to truncation, and the
    mean % of words lost within the affected documents.

Outputs (in --output-dir):
  truncation_per_doc.csv   — one row per document (tokens, words, lost words)
  truncation_summary.csv   — the aggregated metrics (doc-level and word-level)

Usage:
  python count_truncated_docs.py
  python count_truncated_docs.py --corpus corpus/year_2025_2026_iramutec_with_id.txt \
      --max-length 512 --output-dir corpus/truncation_output
"""

import argparse
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcache")

import pandas as pd
from transformers import AutoTokenizer

MODEL_ID = "j-hartmann/emotion-english-distilroberta-base"

HEADER_RE = re.compile(
    r"^\*\*\*\*\s+"
    r"\*id_(\S+)\s+"
    r"\*period_(\d+)\s+"
    r"\*r_(\S+)\s+"
    r"\*type_(\S+)"
    r"(?:\s+\*tier_(\S+))?"
)


def parse_corpus(corpus_path: Path) -> list[dict]:
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
                "id": m.group(1), "period": m.group(2),
                "subreddit": m.group(3), "type": m.group(4),
            }
            current_lines = []
        else:
            if current_meta is not None:
                current_lines.append(line)
    flush()
    return docs


def analyze_doc(tokenizer, text: str, max_length: int) -> dict:
    """Tokenizes without truncating and computes total/lost tokens and words."""
    if not text.strip():
        text = "."
    enc = tokenizer(text, add_special_tokens=True, truncation=False)
    input_ids = enc["input_ids"]
    word_ids = enc.word_ids()
    total_tokens = len(input_ids)

    # lowest token index per word (word_id None = special tokens)
    first_token_idx: dict[int, int] = {}
    for tok_idx, wid in enumerate(word_ids):
        if wid is None:
            continue
        if wid not in first_token_idx:
            first_token_idx[wid] = tok_idx

    total_words = len(first_token_idx)
    is_truncated = total_tokens > max_length
    if is_truncated:
        words_lost = sum(1 for idx in first_token_idx.values() if idx >= max_length)
    else:
        words_lost = 0

    return {
        "total_tokens": total_tokens,
        "total_words": total_words,
        "is_truncated": is_truncated,
        "words_lost": words_lost,
        "words_lost_pct": round(100 * words_lost / total_words, 2) if total_words else 0.0,
    }


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    n_docs = len(df)
    n_truncated = int(df["is_truncated"].sum())
    pct_docs_truncated = 100 * n_truncated / n_docs if n_docs else 0.0

    total_words_corpus = int(df["total_words"].sum())
    total_words_lost = int(df["words_lost"].sum())
    pct_words_lost_corpus = (100 * total_words_lost / total_words_corpus
                              if total_words_corpus else 0.0)

    truncated = df[df["is_truncated"]]
    mean_pct_words_lost_among_truncated = (
        truncated["words_lost_pct"].mean() if not truncated.empty else 0.0
    )

    rows = [
        {"metric": "n_docs", "value": n_docs,
         "note": "Total number of documents in the analyzed corpus."},
        {"metric": "n_docs_truncated", "value": n_truncated,
         "note": "Documents whose total token count (RoBERTa) exceeds 512 — document level."},
        {"metric": "pct_docs_truncated", "value": round(pct_docs_truncated, 2),
         "note": "% of documents affected by truncation — document level."},
        {"metric": "total_words_corpus", "value": total_words_corpus,
         "note": "Total words in the corpus (tokenizer's token->word alignment)."},
        {"metric": "total_words_lost", "value": total_words_lost,
         "note": "Total words falling beyond the 512-token cutoff, summed across all documents."},
        {"metric": "pct_words_lost_corpus", "value": round(pct_words_lost_corpus, 2),
         "note": "% of the CORPUS's total words lost to truncation — word level (the real magnitude of the loss)."},
        {"metric": "mean_pct_words_lost_among_truncated", "value": round(float(mean_pct_words_lost_among_truncated), 2),
         "note": "Mean % of words lost WITHIN the truncated documents (not the whole corpus)."},
    ]
    return pd.DataFrame(rows)


def build_args() -> argparse.Namespace:
    script_dir = Path(__file__).parent
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", type=Path,
                   default=script_dir / "corpus" / "year_2025_2026_iramutec_with_id.txt")
    p.add_argument("--output-dir", type=Path,
                   default=script_dir / "corpus" / "truncation_output")
    p.add_argument("--max-length", type=int, default=512,
                   help="Model's token limit (default: 512)")
    return p.parse_args()


def main() -> None:
    args = build_args()
    if not args.corpus.exists():
        print(f"[error] Corpus not found: {args.corpus}", file=sys.stderr)
        sys.exit(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {MODEL_ID} (no model weights)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

    print(f"Reading corpus: {args.corpus}")
    docs = parse_corpus(args.corpus)
    print(f"  {len(docs)} documents found.")

    print("Tokenizing (without truncating) to measure total tokens/words…")
    rows = []
    for doc in docs:
        stats = analyze_doc(tokenizer, doc["text"], args.max_length)
        rows.append({
            "id": doc["id"], "subreddit": doc["subreddit"], "type": doc["type"],
            **stats,
        })
    df = pd.DataFrame(rows)

    per_doc_path = args.output_dir / "truncation_per_doc.csv"
    df.to_csv(per_doc_path, index=False)
    print(f"  Saved: {per_doc_path}")

    summary = aggregate(df)
    summary_path = args.output_dir / "truncation_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n--- 512-token truncation summary ---")
    for _, row in summary.iterrows():
        print(f"  {row['metric']:<40} {row['value']}")
        print(f"    {row['note']}")

    print(f"\nSaved: {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
