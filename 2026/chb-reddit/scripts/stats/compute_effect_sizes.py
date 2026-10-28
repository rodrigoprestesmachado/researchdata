#!/usr/bin/env python3
"""
Effect sizes and statistical robustness for the comparison of emotions
across HDC classes — complements (and replaces) the "plain" ANOVA from
roberta_emotions.py with effect-size measures and an alternative
non-parametric test, motivated by the fact that n is large (10,943) — which
makes almost any difference "statistically significant" — and that the
scores are proportions bounded to [0, 1] (a plausible violation of the
normality/homoscedasticity assumed by ANOVA).

Reads the already-existing ``emotions_per_doc.csv`` CSV (does not re-run the
transformer model) and computes, per emotion:

  - One-way ANOVA (F, p) + eta^2 (eta^2 = SS_between/SS_total)
  - Levene's test (variance homogeneity across the 3 classes)
  - Kruskal-Wallis H (p) + epsilon^2 (non-parametric effect size analogous
    to eta^2, robust to skew/heteroscedasticity)

Output: emotions_anova.csv (same file produced by roberta_emotions.py, now
with the effect-size and non-parametric-test columns).

Usage:
  python compute_effect_sizes.py
  python compute_effect_sizes.py --per-doc corpus/roberta_output/emotions_per_doc.csv \
      --output corpus/roberta_output/emotions_anova.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import f_oneway, kruskal, levene

EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]


def eta_squared_from_f(f_stat: float, df_between: int, df_within: int) -> float:
    """eta^2 = SS_between/SS_total, equivalent to df1*F / (df1*F + df2)."""
    return (df_between * f_stat) / (df_between * f_stat + df_within)


def epsilon_squared_from_h(h_stat: float, n: int, k: int) -> float:
    """epsilon^2 (Tomczak & Tomczak, 2014) = (H - k + 1) / (n - k)."""
    if n - k <= 0:
        return float("nan")
    return (h_stat - k + 1) / (n - k)


def compute_effect_sizes(df: pd.DataFrame) -> pd.DataFrame:
    classes = sorted(c for c in df["classe"].unique() if c > 0)
    k = len(classes)
    n = len(df)
    df_between = k - 1
    df_within = n - k

    rows = []
    for emotion in EMOTIONS:
        groups = [df.loc[df["classe"] == c, emotion] for c in classes]

        f_stat, p_anova = f_oneway(*groups)
        eta_sq = eta_squared_from_f(float(f_stat), df_between, df_within)

        lev_stat, lev_p = levene(*groups)

        h_stat, p_kw = kruskal(*groups)
        eps_sq = epsilon_squared_from_h(float(h_stat), n, k)

        rows.append({
            "emotion": emotion,
            "F": round(float(f_stat), 4),
            "p_value": p_anova,
            "eta_squared": round(float(eta_sq), 4),
            "significant_p<0.05": bool(p_anova < 0.05),
            "levene_stat": round(float(lev_stat), 4),
            "levene_p": lev_p,
            "kruskal_H": round(float(h_stat), 4),
            "kruskal_p": p_kw,
            "epsilon_squared": round(float(eps_sq), 4),
        })
    return pd.DataFrame(rows)


def build_args() -> argparse.Namespace:
    script_dir = Path(__file__).parent
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per-doc", type=Path,
                   default=script_dir / "corpus" / "roberta_output" / "emotions_per_doc.csv",
                   help="Existing emotions_per_doc.csv CSV (does not re-run the model)")
    p.add_argument("--output", type=Path,
                   default=script_dir / "corpus" / "roberta_output" / "emotions_anova.csv",
                   help="Output path (default: overwrites emotions_anova.csv)")
    return p.parse_args()


def main() -> None:
    args = build_args()
    if not args.per_doc.exists():
        print(f"[error] CSV not found: {args.per_doc}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading: {args.per_doc}")
    df = pd.read_csv(args.per_doc)
    df_classified = df[df["classe"] > 0].copy()
    print(f"  {len(df_classified)} classified documents (excluding class 0).")

    result = compute_effect_sizes(df_classified)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    print(f"\n--- ANOVA + effect sizes (eta^2) and Kruskal-Wallis + epsilon^2 per emotion ---")
    print(result.to_string(index=False))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
