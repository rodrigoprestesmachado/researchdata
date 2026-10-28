#!/usr/bin/env python3
"""
add_corpus_ids.py — Inserts *id_<reddit_id> into the header of the IRaMuTeQ corpus.

The corpus exported by the scraper (reddit_playwright_scraper.py) runs with
--include-id disabled by default, so as not to inflate IRaMuTeQ's Factorial
Correspondence Analysis with one modality per document. This leaves
corpus_iramutec.txt without *id_ in the header, which prevents scripts such
as roberta_emotions.py and nrclex_emotions.py (whose parser requires *id_)
from reading the file.

This script does not re-scrape anything: it recovers the real id of each
document from the final JSONL (same export order) and writes a copy of the
IRaMuTeQ corpus with *id_ inserted, preserving the original text and order —
the same order IRaMuTeQ uses to generate uce.csv (document→HDC class map),
which keeps the position-based pairing used by split_by_class.py and by the
per-class sentiment analysis valid.

The original file (used by IRaMuTeQ's Corpus.cira project) is not modified;
the output is a new file.

Usage:
  python add_corpus_ids.py
  python add_corpus_ids.py --corpus corpus/year_2025_2026_iramutec.txt \
      --jsonl corpus/year_2025_2026.jsonl \
      --output corpus/year_2025_2026_iramutec_with_id.txt
"""

import argparse
import json
import sys
from pathlib import Path


def load_jsonl_ids(path: Path) -> list[tuple[str, str, str]]:
    """Reads the JSONL and returns a list of (id, subreddit, type) in file order."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append((str(d.get("id", "")), str(d.get("subreddit", "")), str(d.get("type", ""))))
    return out


def process(corpus_path: Path, jsonl_path: Path, output_path: Path) -> None:
    ids = load_jsonl_ids(jsonl_path)
    lines = corpus_path.read_text(encoding="utf-8").splitlines()

    out_lines: list[str] = []
    doc_idx = 0
    mismatches = 0

    for line in lines:
        if line.startswith("****"):
            if doc_idx >= len(ids):
                print(
                    f"[error] Corpus has more documents ({doc_idx + 1}) than the JSONL ({len(ids)}).",
                    file=sys.stderr,
                )
                sys.exit(1)

            reddit_id, sub, typ = ids[doc_idx]

            # sanity check: confirms the JSONL row corresponds to the same
            # document (same subreddit/type), at the same position.
            if f"*r_{sub}" not in line or f"*type_{typ}" not in line:
                mismatches += 1
                print(
                    f"[warn] Mismatch at document {doc_idx + 1}: "
                    f"header={line!r} vs jsonl=({reddit_id}, {sub}, {typ})",
                    file=sys.stderr,
                )

            # insert *id_<reddit_id> right after "****", before *period_/*r_/...
            rest = line[len("****"):].lstrip()
            out_lines.append(f"**** *id_{reddit_id} {rest}")
            doc_idx += 1
        else:
            out_lines.append(line)

    if doc_idx != len(ids):
        print(
            f"[warn] {doc_idx} documents in the corpus vs {len(ids)} in the JSONL "
            "— counts differ.",
            file=sys.stderr,
        )

    if mismatches:
        print(f"[error] {mismatches} alignment mismatch(es) found. Aborting.", file=sys.stderr)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    print(f"[ok] {doc_idx} documents processed, 0 mismatches.")
    print(f"[ok] Corpus with *id_ saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", type=Path, default=script_dir / "corpus" / "year_2025_2026_iramutec.txt",
                    help="Original IRaMuTeQ corpus (without *id_)")
    p.add_argument("--jsonl", type=Path, default=script_dir / "corpus" / "year_2025_2026.jsonl",
                    help="Final JSONL, in the same export order as the IRaMuTeQ corpus")
    p.add_argument("--output", type=Path,
                    default=script_dir / "corpus" / "year_2025_2026_iramutec_with_id.txt",
                    help="Output path for the new corpus (with *id_)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.corpus.exists():
        print(f"[error] Corpus not found: {args.corpus}", file=sys.stderr)
        sys.exit(1)
    if not args.jsonl.exists():
        print(f"[error] JSONL not found: {args.jsonl}", file=sys.stderr)
        sys.exit(1)
    process(args.corpus, args.jsonl, args.output)
