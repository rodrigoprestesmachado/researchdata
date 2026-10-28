# Teachers' Spontaneous Discourse on Generative AI — Data and Code Release

This repository contains the corpus, analysis scripts, and results supporting:

> Machado, R. P., Arredondo Gonzalez, E. D. I., Bengochea, J., & Reategui, E. B.
> *Teachers' Spontaneous Discourse on Generative AI: A Lexicometric and
> Sentiment Analysis of Reddit Communities.* Federal Institute of Education,
> Science and Technology of Rio Grande do Sul (IFRS).

The study analyses 10,985 posts and comments from r/Teachers and
r/Professors mentioning generative AI, collected between mid-2025 and
mid-2026, using Reinert's Hierarchical Descending Classification (HDC/CHD)
and Factorial Correspondence Analysis (FCA) in IRaMuTeQ, together with
transformer-based (DistilRoBERTa) and lexicon-based (NRCLex) emotion
classification and associated statistical robustness checks.

This release lets you: (1) inspect the exact corpus used in the paper,
(2) re-run every statistical/emotion-analysis step starting from that
corpus, and (3) re-run the scraper to collect a fresh corpus from Reddit.
IRaMuTeQ itself (the CHD/FCA step) is external, free, GUI software and is
not re-implemented here; the corpus is provided in the format IRaMuTeQ
expects, and its outputs for the run used in the paper are included as-is.

## Contents

```
release/
├── README.md
├── requirements.txt
├── scripts/
│   ├── scraper/
│   │   └── reddit_playwright_scraper.py
│   ├── iramutec/
│   │   ├── add_corpus_ids.py
│   │   └── split_by_class.py
│   ├── sentiment/
│   │   ├── roberta_emotions.py
│   │   ├── nrclex_emotions.py
│   │   └── compare_emotions.py
│   └── stats/
│       ├── compute_effect_sizes.py
│       ├── count_truncated_docs.py
│       ├── emotion_classifier_diagnostics.py
│       └── temporal_class_dynamics.py
├── corpus/
│   ├── year_2025_2026.jsonl
│   ├── year_2025_2026.csv
│   ├── year_2025_2026_iramutec.txt
│   ├── year_2025_2026_iramutec_with_id.txt
│   └── iramutec/
│       ├── year_2025_2026_iramutec_alceste_3/   (HDC/CHD output used in the paper)
│       └── year_2025_2026_iramutec_stat_1/      (corpus-level lexical statistics)
└── results/
    ├── roberta_output/       (DistilRoBERTa emotion scores, ANOVA/Tukey, plots)
    ├── nrclex_output/        (NRCLex lexicon-based emotion scores, plots)
    ├── compare_output/       (RoBERTa vs. NRCLex comparison)
    ├── diagnostics_output/   (disgust/neutral validity checks)
    ├── truncation_output/    (512-token truncation impact, per document and corpus-wide)
    └── temporal_output/      (class size and within-class valence over the 12 monthly periods)
```

### Scripts

| Script | Purpose |
|---|---|
| `scripts/scraper/reddit_playwright_scraper.py` | Collects posts and comments from specified subreddits via Reddit's public JSON endpoints, using Playwright to solve the anti-bot JS challenge. In one run, produces the raw scrape, the filtered corpus (`.jsonl`/`.csv`), and an IRaMuTeQ-formatted `.txt` corpus. |
| `scripts/iramutec/add_corpus_ids.py` | Adds a `*id_<post_or_comment_id>` starred variable to each document header of the IRaMuTeQ corpus, so IRaMuTeQ output tables (e.g., `uce.csv`) can be joined back to the original documents. |
| `scripts/iramutec/split_by_class.py` | Splits the IRaMuTeQ corpus into one sub-corpus per HDC class, using the class assignment exported by IRaMuTeQ (`uce.csv`). Useful for class-specific inspection or re-analysis. |
| `scripts/sentiment/roberta_emotions.py` | Runs the `j-hartmann/emotion-english-distilroberta-base` transformer classifier over every document, aggregates mean scores by HDC class/subreddit/type, and computes the one-way ANOVA + effect sizes ($\eta^2$), Levene's test, Kruskal-Wallis + $\varepsilon^2$, and Tukey HSD post-hoc comparisons by class. |
| `scripts/sentiment/nrclex_emotions.py` | Runs the lexicon-based NRCLex classifier over the same corpus, as a non-neural point of comparison for `roberta_emotions.py`. |
| `scripts/sentiment/compare_emotions.py` | Compares the RoBERTa and NRCLex per-document scores (mean differences, Pearson correlations) for the emotion categories shared by both methods. |
| `scripts/stats/compute_effect_sizes.py` | Recomputes the ANOVA/Levene/Kruskal-Wallis/effect-size table from an existing `emotions_per_doc.csv`, without re-running the transformer model. |
| `scripts/stats/count_truncated_docs.py` | Quantifies the impact of DistilRoBERTa's 512-token input limit, at both the document level (% of documents truncated) and the word level (% of words actually lost), using the classifier's own tokenizer (no model weights needed). |
| `scripts/stats/emotion_classifier_diagnostics.py` | Runs the validity checks for the `disgust` and `neutral` DistilRoBERTa scores: disgust–anger correlation, comparison against NRCLex, lexical face-validity check on the highest-scoring `disgust` documents, and an analogous check for `neutral` against argumentative/deliberative markers. |
| `scripts/stats/temporal_class_dynamics.py` | Reuses `emotions_per_doc.csv` from `roberta_emotions.py` and `nrclex_emotions.py` (no model re-run) to test whether (1) the relative size of each HDC class changes across the corpus's 12 monthly periods (chi-square test of independence + Cramér's V, plus a per-class Spearman trend), and (2) emotional valence *within* each class changes over those same periods (per-class one-way ANOVA/Kruskal-Wallis of valence ~ period, a per-class Spearman trend at document level, and a directed September/2025 vs.\ May/2026 comparison via Welch's *t*-test and Mann-Whitney U). Valence is computed two ways: `roberta_valence = joy − (anger+disgust+fear+sadness)` and `nrclex_valence = positive − negative`. |

## Data dictionary

### `corpus/year_2025_2026.jsonl` / `corpus/year_2025_2026.csv`

One record per Reddit post or comment, after language/relevance filtering
(the "analysis corpus" used throughout the paper). Both files contain the
same records; the CSV is provided for convenience.

| Field | Description |
|---|---|
| `source` | Always `"reddit"`. |
| `type` | `"post"` or `"comment"`. |
| `id` | Reddit post/comment ID (public, matches the URL/permalink). |
| `post_id` | For comments, the ID of the parent post (empty for posts). |
| `subreddit` | `Teachers` or `Professors`. |
| `author` | Reddit username, as publicly displayed on Reddit at the time of collection. Kept as-is: Reddit usernames and their public posts/comments are public data, and no attempt is made to link them to any other identity. |
| `title` | Post title (empty for comments). |
| `text` | Post body or comment body. |
| `score` | Reddit score (upvotes − downvotes) at collection time. |
| `num_comments` | Number of comments on the post (posts only). |
| `permalink` | Relative Reddit permalink. |
| `created_date` | UTC creation date/time. |
| `relevance_tier` | Filtering tier applied during collection (e.g., `core`) — see `reddit_playwright_scraper.py` for the exact keyword/relevance rule. |
| `word_count` | Whitespace-delimited word count of `text`. |
| `has_url` | Whether `text` contains a URL. |
| `is_bot` | Heuristic flag for likely bot-authored content. |

### `corpus/year_2025_2026_iramutec.txt` / `corpus/year_2025_2026_iramutec_with_id.txt`

The same documents, reformatted as an IRaMuTeQ-compatible corpus (see
[IRaMuTeQ format](#iramutec-corpus-format) below).
`_with_id.txt` additionally stars the document ID as an `*id_` variable,
produced by `add_corpus_ids.py`, so that IRaMuTeQ's own outputs
(`uce.csv`, etc.) can be joined back to `year_2025_2026.jsonl`/`.csv` and
to the emotion-analysis CSVs in `results/`.

### `results/roberta_output/emotions_per_doc.csv` (and the equivalent NRCLex file)

One row per document, with the seven DistilRoBERTa emotion scores
(`anger, disgust, fear, joy, neutral, sadness, surprise`, softmax
probabilities summing to 1) plus `id`, `period`, `subreddit`, `type`,
`word_count`, and the HDC `classe`/`classe_label` it was assigned to
(`classe = 0` for documents IRaMuTeQ left unclassified). The NRCLex
version reports the analogous frequencies for its 10 categories (the
NRC lexicon's 8 basic emotions plus `positive`/`negative` polarity).

### `results/roberta_output/emotions_anova.csv`

One row per emotion, with the one-way ANOVA ($F$, $p$, $\eta^2$),
Levene's test (`levene_stat`, `levene_p`), and the Kruskal-Wallis test
(`kruskal_H`, `kruskal_p`, `epsilon_squared`) comparing the three HDC
classes — see `scripts/stats/compute_effect_sizes.py`.

### `results/diagnostics_output/`, `results/truncation_output/`, `results/compare_output/`

Outputs of the three robustness/validity scripts in `scripts/stats/` and
`scripts/sentiment/compare_emotions.py`, respectively; the numbers
reported here are the ones cited in the paper's "Robustness and validity
checks" subsections (Methods and Results) and in Table
"Diagnostic checks for the DistilRoBERTa disgust and neutral scores, and
for truncation at 512 tokens".

### `results/temporal_output/`

Output of `scripts/stats/temporal_class_dynamics.py`. `class_size_by_period.csv`,
`class_size_chi2.csv`, and `class_size_trend.csv` address whether the
relative size of each HDC class is stable across the corpus's 12 monthly
periods (`period`, `YYYYMM`, July/2025–June/2026). `valence_by_class_period.csv`,
`valence_anova_by_class.csv`, `valence_trend_by_class.csv`, and
`valence_sept_vs_may.csv` address whether emotional valence *within* each
class (`roberta_valence = joy − (anger+disgust+fear+sadness)`;
`nrclex_valence = positive − negative`) changes over those same periods,
including a directed comparison between September/2025 (`202509`) and
May/2026 (`202605`).

## IRaMuTeQ corpus format

[IRaMuTeQ](http://www.iramuteq.org/) expects a plain-text corpus where
each document is preceded by a header line starting with four asterisks,
followed by one or more starred ("`*variable_value`") metadata variables,
e.g.:

```
**** *id_1ujyapy *period_202606 *r_Professors *type_post *tier_core
Should we stop giving students written assignments in the age of AI? During a
conference today, I heard a strong opinion that students should no longer...
**** *id_ouscum8 *period_202606 *r_Professors *type_comment *tier_core
I completely agree with the concern about over-reliance on AI tools...
```

The starred variables used here are:

- `*id_` — the Reddit post/comment ID (added by `add_corpus_ids.py`; not
  present in `year_2025_2026_iramutec.txt`, only in `_with_id.txt`).
- `*period_` — year/month of creation (`YYYYMM`).
- `*r_` — subreddit (`Professors` or `Teachers`).
- `*type_` — `post` or `comment`.
- `*tier_` — relevance tier from the collection/filtering step.

`corpus/iramutec/year_2025_2026_iramutec_alceste_3/` contains the
IRaMuTeQ project outputs for the HDC (Reinert method) run used in the
paper (3 classes): `uce.csv` (per-segment class assignment, joinable to
`*id_`), `chisqtable.csv` (word-by-class $\chi^2$ associations),
`profiles.csv`/`antiprofiles.csv` (characteristic/anti-characteristic
words per class), `afc_row.csv`/`afc_col.csv`/`afc_facteur.csv` (FCA
coordinates), and the dendrogram/FCA plots.
`corpus/iramutec/year_2025_2026_iramutec_stat_1/` contains corpus-level
lexical statistics (Zipf plot, hapax list, active-form frequencies,
segment sizes). Older/exploratory IRaMuTeQ runs, its proprietary
project/database files (`Corpus.cira`, `*.db`, `*.RData`), and the raw,
pre-filter corpus are not included in this release (see
[Exclusions](#exclusions-from-this-release)).

## Reproducing the analysis

All commands below assume they are run from the `release/` directory,
with paths adjusted if scripts are moved.

### 0. Install

```bash
python3 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium       # only needed for the scraper
```

The emotion classifier (`roberta_emotions.py`) downloads
`j-hartmann/emotion-english-distilroberta-base` (~330 MB) from Hugging
Face on first run and requires `torch`; the other scripts do not need a
GPU and run in well under a minute on the full corpus.

### 1. Collect data (optional — a ready-made corpus is already included)

`reddit_playwright_scraper.py` scrapes Reddit and, from `--output
corpus/year_2025_2026`, writes four files in one run: the raw scrape
(`_raw.jsonl`), the filtered corpus (`.jsonl` and `.csv`), and an
IRaMuTeQ-formatted `.txt` corpus (`_iramutec.txt`):

```bash
python scripts/scraper/reddit_playwright_scraper.py \
    --start-date 2025-07-01 --end-date 2026-06-30 \
    --subreddits Teachers Professors \
    --core-only \
    --output corpus/year_2025_2026
```

This produces `corpus/year_2025_2026_raw.jsonl`,
`corpus/year_2025_2026.jsonl`, `corpus/year_2025_2026.csv`, and
`corpus/year_2025_2026_iramutec.txt`. See `--help` for the full set of
options (date range, subreddits, minimum word counts, bot/English
filters, comment-collection mode, etc.). To re-apply filtering to an
already-scraped raw file without re-scraping (e.g., after tweaking a
filter), use `--refilter corpus/year_2025_2026_raw.jsonl` in place of
`--start-date`/`--end-date`.

Passing `--iramutec-include-id` at scrape time makes the script embed
the `*id_` variable directly in `_iramutec.txt`, producing the
`_with_id.txt` file in a single step (skipping step 2 below).

### 2. Add document IDs to the IRaMuTeQ corpus (if not already included)

If the IRaMuTeQ corpus was generated without `--iramutec-include-id`
(as `corpus/year_2025_2026_iramutec.txt` was), add the `*id_` variable
used to join IRaMuTeQ's outputs back to the original documents:

```bash
python scripts/iramutec/add_corpus_ids.py \
    --corpus corpus/year_2025_2026_iramutec.txt \
    --jsonl corpus/year_2025_2026.jsonl \
    --output corpus/year_2025_2026_iramutec_with_id.txt
```

### 3. Run IRaMuTeQ (external GUI software, not scripted here)

Open `corpus/year_2025_2026_iramutec_with_id.txt` in
[IRaMuTeQ](http://www.iramuteq.org/) (free, requires R):

1. **Text statistics** (`Statistics > Text statistics`) → reproduces
   `corpus/iramutec/year_2025_2026_iramutec_stat_1/`.
2. **Descending Hierarchical Classification** (`Analysis > Text >
   Descending Hierarchical Classification`, Reinert method) with 3
   classes → reproduces `corpus/iramutec/year_2025_2026_iramutec_alceste_3/`
   (results depend on IRaMuTeQ/R versions and minor parameter choices;
   the outputs included here are the exact ones used in the paper).
3. Export `uce.csv` from the HDC results (already included).

### 4. Split the corpus by HDC class (optional)

```bash
python scripts/iramutec/split_by_class.py \
    --corpus corpus/year_2025_2026_iramutec_with_id.txt \
    --uce corpus/iramutec/year_2025_2026_iramutec_alceste_3/uce.csv \
    --jsonl corpus/year_2025_2026.jsonl \
    --output corpus/by_class/
```

### 5. Emotion analysis

```bash
python scripts/sentiment/roberta_emotions.py \
    --corpus corpus/year_2025_2026_iramutec_with_id.txt \
    --uce-csv corpus/iramutec/year_2025_2026_iramutec_alceste_3/uce.csv \
    --output-dir results/roberta_output/

python scripts/sentiment/nrclex_emotions.py \
    --corpus corpus/year_2025_2026_iramutec_with_id.txt \
    --output-dir results/nrclex_output/

python scripts/sentiment/compare_emotions.py \
    --nrclex results/nrclex_output/emotions_per_doc.csv \
    --roberta results/roberta_output/emotions_per_doc.csv \
    --output-dir results/compare_output/
```

(Run each script with `--help` to see the exact flags; defaults point to
the paths used above.)

### 6. Robustness and validity checks

```bash
# ANOVA + eta^2, Levene, Kruskal-Wallis + epsilon^2 (reuses emotions_per_doc.csv, no model re-run)
python scripts/stats/compute_effect_sizes.py \
    --per-doc results/roberta_output/emotions_per_doc.csv \
    --output results/roberta_output/emotions_anova.csv

# 512-token truncation impact (document- and word-level)
python scripts/stats/count_truncated_docs.py \
    --corpus corpus/year_2025_2026_iramutec_with_id.txt \
    --output-dir results/truncation_output/

# disgust/neutral validity diagnostics
python scripts/stats/emotion_classifier_diagnostics.py \
    --roberta-per-doc results/roberta_output/emotions_per_doc.csv \
    --nrclex-per-doc results/nrclex_output/emotions_per_doc.csv \
    --corpus corpus/year_2025_2026_iramutec_with_id.txt \
    --output-dir results/diagnostics_output/
```

The numbers these three commands produce are exactly the ones reported
in the paper's "Robustness and validity checks" subsections (Methods
and Results) and its corresponding tables.

### 7. Temporal dynamics within discourse classes

```bash
python scripts/stats/temporal_class_dynamics.py \
    --roberta-per-doc results/roberta_output/emotions_per_doc.csv \
    --nrclex-per-doc  results/nrclex_output/emotions_per_doc.csv \
    --output-dir results/temporal_output/
```

Reuses the two `emotions_per_doc.csv` files (no model re-run) to test
whether HDC class size and within-class emotional valence are stable
across the corpus's 12 monthly periods, addressing RQ3 (temporal
stability) as reported in the paper's Method subsection "Temporal
dynamics within discourse classes (RQ3)" and Results subsection "RQ3:
Temporal dynamics within discourse classes".

## License and ethics

All data was collected from publicly accessible Reddit posts and
comments (r/Teachers, r/Professors), with no authentication or access to
non-public content. Reddit usernames are kept in the `author` field
as-is, since they are already public alongside the corresponding posts
and comments on Reddit; no attempt is made in this release, or in the
underlying study, to de-anonymise, profile, or link authors to any
identity beyond what Reddit itself displays publicly. If you are the
author of a post or comment included in this corpus and would like it
removed from this release, please contact the corresponding author.

This release is made available for research and educational purposes,
in support of the reproducibility of the associated paper. Please cite
the paper (see the top of this file) if you use this data or code.

## Limitations acknowledged in this release

These are quantified in `results/diagnostics_output/` and
`results/truncation_output/`, and discussed at length in the paper's
Methods, Results, and Discussion sections:

1. **Recency bias in comment sampling.** Threads with more than 50
   comments are represented only by their 50 most recent comments,
   under-sampling early, high-engagement reactions in favour of later,
   more settled replies.
2. **512-token truncation.** DistilRoBERTa only scores the first ~512
   tokens of each document. This affects 2.85% of documents, but
   accounts for 5.20% of the corpus's total words, with an average
   27.31% of words lost within the affected documents (see
   `results/truncation_output/truncation_summary.csv`).
3. **Emotion classifier validity.** The DistilRoBERTa `disgust` score
   correlates only weakly with a lexicon-based classifier (NRCLex) and
   is on average 4.59× higher than it; 70% of the 30 highest-`disgust`
   documents contain no literal disgust vocabulary, suggesting the
   score partly captures generic critical/dismissive language rather
   than literal repulsion (see
   `results/diagnostics_output/diagnostics_summary.csv` and
   `disgust_top_examples.csv`). A parallel concern about `neutral`
   conflating affective neutrality with an argumentative register was
   tested and *not* corroborated by the lexical-marker check used here.

## Exclusions from this release

The following are intentionally not included, and are documented here
for transparency:

- `corpusOld/` and superseded/exploratory IRaMuTeQ runs (`alceste_1`,
  `alceste_2`, `spec_1`) — earlier iterations of the same analysis,
  superseded by `alceste_3`, kept out to avoid ambiguity about which
  run the paper reports.
- IRaMuTeQ's proprietary project/database files (`Corpus.cira`,
  `corpus.db`, `formes.db`, `uces.db`, `*.RData`) — internal to the
  IRaMuTeQ/R session and not needed to inspect or reproduce the
  reported results from the exported CSVs/plots included here.
- `year_2025_2026_raw.jsonl` — the unfiltered, pre-relevance-filter
  corpus.
- `classify_students_ai.py` — an LLM-based classification script used
  in an unrelated study by the same authors, not part of this paper's
  pipeline.
- `.env`, `.venv/`, `__pycache__/`, `.DS_Store` — local environment
  files and caches, not part of the code or data.
