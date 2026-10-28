#!/usr/bin/env python3
"""
Temporal dynamics by HDC class — class size and emotional valence over the
corpus's 12 periods (Jul/2025 to Jun/2026).

Answers two questions, reusing the already-existing CSVs (does not re-run
any model/classifier):

  1. Does the size of each HDC class (% of the period's documents falling
     into each class) change over the year, or is the per-class composition
     stable month to month?
  2. Within each class, does emotional valence change over the year (e.g.,
     Class 1 becomes more negative in September/2025 than in May/2026),
     even if the class's size remains stable?

Emotional valence (two metrics, in parallel, same spirit as
compare_emotions.py):

  - roberta_valence = joy - (anger + disgust + fear + sadness)
    (surprise and neutral excluded as valence-ambiguous)
  - nrclex_valence   = positive - negative
    (comes directly from NRCLex, no need to derive it)

Both metrics range over [-1, 1]: positive = more joy/positive than
anger+disgust+fear+sadness/negative; negative = the reverse.

Sources (does not re-run RoBERTa/NRCLex, only reads the already-generated
per_doc.csv files):
  --roberta-per-doc  results/roberta_output/emotions_per_doc.csv
                      (id, period, classe, classe_label, 7 RoBERTa scores)
  --nrclex-per-doc   results/nrclex_output/emotions_per_doc.csv
                      (id, period, 10 NRCLex scores incl. positive/negative;
                      no class — joined to the RoBERTa file via id)

Documents not classified by the HDC (classe == 0) are excluded from the
whole analysis, as in the rest of the release.

Outputs (in --output-dir):
  class_size_by_period.csv        — count and % of each class, per period
  class_size_chi2.csv             — chi-square (period × class) + Cramér's V
  class_size_trend.csv            — trend (Spearman) of each class's % over the periods
  valence_by_class_period.csv     — mean valence (RoBERTa and NRCLex) by class × period
  valence_anova_by_class.csv      — ANOVA/Levene/Kruskal-Wallis (valence ~ period) within each class
  valence_trend_by_class.csv      — trend (Spearman, document level) of valence over the year, per class
  valence_sept_vs_may.csv         — directed comparison Sept/2025 (202509) vs May/2026 (202605), per class
  plot_class_size_share.png       — stacked area of each class's % per period
  plot_class_size_counts.png      — lines with document count per class and period
  plot_valence_by_class_period_roberta.png  — mean RoBERTa valence per class over the periods
  plot_valence_by_class_period_nrclex.png   — same, NRCLex valence
  plot_valence_heatmap_roberta.png          — class × period heatmap (RoBERTa valence)
  plot_valence_heatmap_nrclex.png           — class × period heatmap (NRCLex valence)

Usage:
  python temporal_class_dynamics.py
  python temporal_class_dynamics.py \
      --roberta-per-doc results/roberta_output/emotions_per_doc.csv \
      --nrclex-per-doc  results/nrclex_output/emotions_per_doc.csv \
      --output-dir results/temporal_output/
  python temporal_class_dynamics.py --no-plots
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
from scipy.stats import chi2_contingency, f_oneway, kruskal, levene, mannwhitneyu, spearmanr, ttest_ind

ROBERTA_NEG_EMOTIONS = ["anger", "disgust", "fear", "sadness"]
SEPT_PERIOD = 202509   # September/2025
MAY_PERIOD  = 202605   # May/2026


# ---------------------------------------------------------------------------
# Loading and preparation
# ---------------------------------------------------------------------------

def period_label(period: int) -> str:
    """202509 -> '2025-09'."""
    s = str(int(period))
    return f"{s[:4]}-{s[4:]}"


def load_data(roberta_path: Path, nrclex_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reads both per_doc.csv files, derives the valence columns, and joins
    the HDC class into the NRCLex DataFrame (which lacks it) via id.

    Returns (rob_df, nrc_df), both already filtered to classe > 0 and with a
    "period_order" column (chronological position 1..N of the period).
    """
    rob_df = pd.read_csv(roberta_path)
    nrc_df = pd.read_csv(nrclex_path)

    rob_df["roberta_valence"] = (
        rob_df["joy"] - rob_df[ROBERTA_NEG_EMOTIONS].sum(axis=1)
    )

    nrc_df = nrc_df.merge(
        rob_df[["id", "classe", "classe_label"]], on="id", how="inner"
    )
    nrc_df["nrclex_valence"] = nrc_df["positive"] - nrc_df["negative"]

    rob_df = rob_df[rob_df["classe"] > 0].copy()
    nrc_df = nrc_df[nrc_df["classe"] > 0].copy()

    periods_sorted = sorted(rob_df["period"].unique())
    order_map = {p: i + 1 for i, p in enumerate(periods_sorted)}
    rob_df["period_order"] = rob_df["period"].map(order_map)
    nrc_df["period_order"] = nrc_df["period"].map(order_map)

    return rob_df, nrc_df


def class_order(df: pd.DataFrame) -> list[str]:
    """Class labels ordered by class number (1, 2, 3, ...)."""
    pairs = df[["classe", "classe_label"]].drop_duplicates().sort_values("classe")
    return pairs["classe_label"].tolist()


def _short_class_label(label: str) -> str:
    """'Class 1 \u2013 Teaching Profession & Educational Context' -> 'Class 1'."""
    return label.split("\u2013")[0].strip()


# ---------------------------------------------------------------------------
# Question 1 — class size over the year
# ---------------------------------------------------------------------------

def cramers_v(chi2: float, n: int, r: int, c: int) -> float:
    return float(np.sqrt(chi2 / (n * (min(r, c) - 1))))


def compute_class_size(rob_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (counts_wide, pct_wide, chi2_df, trend_df)."""
    labels = class_order(rob_df)
    counts_wide = pd.crosstab(rob_df["period"], rob_df["classe_label"])[labels]
    pct_wide = counts_wide.div(counts_wide.sum(axis=1), axis=0) * 100

    chi2, p, dof, _ = chi2_contingency(counts_wide.to_numpy())
    v = cramers_v(float(chi2), int(counts_wide.to_numpy().sum()), *counts_wide.shape)
    chi2_df = pd.DataFrame([{
        "n_periods": counts_wide.shape[0],
        "n_classes": counts_wide.shape[1],
        "n_docs": int(counts_wide.to_numpy().sum()),
        "chi2": round(float(chi2), 4),
        "dof": int(dof),
        "p_value": float(p),
        "significant_p<0.05": bool(p < 0.05),
        "cramers_v": round(v, 4),
    }])

    period_order = list(range(1, len(counts_wide.index) + 1))
    trend_rows = []
    for label in labels:
        rho, rho_p = spearmanr(period_order, pct_wide[label].to_numpy())
        trend_rows.append({
            "classe_label": label,
            "pct_min": round(float(pct_wide[label].min()), 2),
            "pct_max": round(float(pct_wide[label].max()), 2),
            "pct_range": round(float(pct_wide[label].max() - pct_wide[label].min()), 2),
            "pct_first_period": round(float(pct_wide[label].iloc[0]), 2),
            "pct_last_period": round(float(pct_wide[label].iloc[-1]), 2),
            "spearman_rho": round(float(rho), 4),
            "spearman_p": float(rho_p),
            "significant_p<0.05": bool(rho_p < 0.05),
        })
    trend_df = pd.DataFrame(trend_rows)

    return counts_wide, pct_wide, chi2_df, trend_df


def build_class_size_long(counts_wide: pd.DataFrame, pct_wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period in counts_wide.index:
        for label in counts_wide.columns:
            rows.append({
                "period": int(period),
                "period_label": period_label(period),
                "classe_label": label,
                "n": int(counts_wide.loc[period, label]),
                "pct_of_period": round(float(pct_wide.loc[period, label]), 4),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Question 2 — valence within each class over the year
# ---------------------------------------------------------------------------

def _eta_squared_from_f(f_stat: float, df_between: int, df_within: int) -> float:
    return (df_between * f_stat) / (df_between * f_stat + df_within)


def _epsilon_squared_from_h(h_stat: float, n: int, k: int) -> float:
    if n - k <= 0:
        return float("nan")
    return (h_stat - k + 1) / (n - k)


def compute_valence_by_class_period(rob_df: pd.DataFrame, nrc_df: pd.DataFrame) -> pd.DataFrame:
    rob_agg = (rob_df.groupby(["period", "classe_label"])["roberta_valence"]
               .agg(n_roberta="count", mean_roberta_valence="mean").reset_index())
    nrc_agg = (nrc_df.groupby(["period", "classe_label"])["nrclex_valence"]
               .agg(n_nrclex="count", mean_nrclex_valence="mean").reset_index())
    merged = rob_agg.merge(nrc_agg, on=["period", "classe_label"], how="outer")
    merged["period_label"] = merged["period"].apply(period_label)
    merged["mean_roberta_valence"] = merged["mean_roberta_valence"].round(4)
    merged["mean_nrclex_valence"] = merged["mean_nrclex_valence"].round(4)
    cols = ["period", "period_label", "classe_label", "n_roberta",
            "mean_roberta_valence", "n_nrclex", "mean_nrclex_valence"]
    return merged[cols].sort_values(["classe_label", "period"]).reset_index(drop=True)


def compute_valence_anova_by_class(df: pd.DataFrame, value_col: str, metric: str) -> pd.DataFrame:
    """Per class: ANOVA/Levene/Kruskal-Wallis of (value_col ~ period), i.e.
    tests whether valence varies significantly across the 12 periods within
    that class."""
    rows = []
    for label in class_order(df):
        sub = df[df["classe_label"] == label]
        periods = sorted(sub["period"].unique())
        k = len(periods)
        n = len(sub)
        groups = [sub.loc[sub["period"] == p, value_col] for p in periods]

        f_stat, p_anova = f_oneway(*groups)
        eta_sq = _eta_squared_from_f(float(f_stat), k - 1, n - k)

        lev_stat, lev_p = levene(*groups)

        h_stat, p_kw = kruskal(*groups)
        eps_sq = _epsilon_squared_from_h(float(h_stat), n, k)

        rows.append({
            "classe_label": label,
            "metric": metric,
            "n": n,
            "n_periods": k,
            "F": round(float(f_stat), 4),
            "p_value": float(p_anova),
            "eta_squared": round(float(eta_sq), 4),
            "significant_p<0.05": bool(p_anova < 0.05),
            "levene_stat": round(float(lev_stat), 4),
            "levene_p": float(lev_p),
            "kruskal_H": round(float(h_stat), 4),
            "kruskal_p": float(p_kw),
            "epsilon_squared": round(float(eps_sq), 4),
        })
    return pd.DataFrame(rows)


def compute_valence_trend_by_class(df: pd.DataFrame, value_col: str, metric: str) -> pd.DataFrame:
    """Per class: Spearman correlation (document level) between the period's
    chronological order and valence — a monotonic trend over the year
    (independent of the pointwise Sept-vs-May comparison)."""
    rows = []
    for label in class_order(df):
        sub = df[df["classe_label"] == label]
        rho, rho_p = spearmanr(sub["period_order"], sub[value_col])
        rows.append({
            "classe_label": label,
            "metric": metric,
            "n": len(sub),
            "spearman_rho": round(float(rho), 4),
            "spearman_p": float(rho_p),
            "significant_p<0.05": bool(rho_p < 0.05),
        })
    return pd.DataFrame(rows)


def compute_sept_vs_may(df: pd.DataFrame, value_col: str, metric: str) -> pd.DataFrame:
    """Per class: directed comparison Sept/2025 (202509) vs May/2026
    (202605) — Welch's t-test (parametric) and Mann-Whitney U
    (non-parametric), over the per-document valence in the two periods."""
    rows = []
    for label in class_order(df):
        sub = df[df["classe_label"] == label]
        sept = sub.loc[sub["period"] == SEPT_PERIOD, value_col]
        may = sub.loc[sub["period"] == MAY_PERIOD, value_col]
        if len(sept) < 2 or len(may) < 2:
            continue

        t_result = ttest_ind(may, sept, equal_var=False)
        t_stat, t_p, t_df = float(t_result.statistic), float(t_result.pvalue), float(t_result.df)
        u_stat, u_p = mannwhitneyu(may, sept, alternative="two-sided")

        mean_sept = float(sept.mean())
        mean_may = float(may.mean())
        more_negative = "September/2025" if mean_sept < mean_may else (
            "May/2026" if mean_may < mean_sept else "tie")

        rows.append({
            "classe_label": label,
            "metric": metric,
            "n_sept_2025": len(sept),
            "n_may_2026": len(may),
            "mean_sept_2025": round(mean_sept, 4),
            "mean_may_2026": round(mean_may, 4),
            "mean_diff_may_minus_sept": round(mean_may - mean_sept, 4),
            "more_negative_valence_in": more_negative,
            "t_stat": round(t_stat, 4),
            "t_df": round(t_df, 2),
            "t_p_value": t_p,
            "u_stat": round(float(u_stat), 4),
            "u_p_value": float(u_p),
            "significant_p<0.05_either_test": bool(t_p < 0.05 or u_p < 0.05),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def _set_style():
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({"figure.dpi": 150})


def plot_class_size_share(pct_wide: pd.DataFrame, out_dir: Path) -> None:
    labels = [period_label(p) for p in pct_wide.index]
    short_labels = [_short_class_label(c) for c in pct_wide.columns]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.stackplot(labels, [pct_wide[c].to_numpy() for c in pct_wide.columns],
                 labels=short_labels, alpha=0.85,
                 colors=sns.color_palette("muted", len(pct_wide.columns)))
    ax.set_title("Composition by HDC class over the year (% of documents per period)",
                 fontsize=13, pad=12)
    ax.set_xlabel("Period")
    ax.set_ylabel("% of documents in period")
    ax.set_ylim(0, 100)
    ax.set_xlim(labels[0], labels[-1])
    ax.legend(title="Class", loc="upper center", bbox_to_anchor=(0.5, -0.22),
               ncol=len(short_labels), frameon=False)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(out_dir / "plot_class_size_share.png", bbox_inches="tight")
    plt.close(fig)


def plot_class_size_counts(counts_wide: pd.DataFrame, out_dir: Path) -> None:
    labels = [period_label(p) for p in counts_wide.index]
    short_labels = [_short_class_label(c) for c in counts_wide.columns]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, c in enumerate(counts_wide.columns):
        ax.plot(labels, counts_wide[c].to_numpy(), marker="o", label=short_labels[i],
                 color=sns.color_palette("muted", len(counts_wide.columns))[i])
    ax.set_title("Document count by HDC class and period", fontsize=13, pad=12)
    ax.set_xlabel("Period")
    ax.set_ylabel("No. of documents")
    ax.legend(title="Class", loc="upper center", bbox_to_anchor=(0.5, -0.22),
               ncol=len(short_labels), frameon=False)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(out_dir / "plot_class_size_counts.png", bbox_inches="tight")
    plt.close(fig)


def plot_valence_by_class_period(df: pd.DataFrame, value_col: str, title: str,
                                  filename: str, out_dir: Path) -> None:
    labels = class_order(df)
    short_labels = [_short_class_label(lbl) for lbl in labels]
    periods = sorted(df["period"].unique())
    period_labels = [period_label(p) for p in periods]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, label in enumerate(labels):
        sub = df[df["classe_label"] == label]
        means = sub.groupby("period")[value_col].mean().reindex(periods)
        ax.plot(period_labels, means.to_numpy(), marker="o", label=short_labels[i],
                 color=sns.color_palette("muted", len(labels))[i])
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("Period")
    ax.set_ylabel("Mean valence (positive = more positive)")
    ax.legend(title="Class", loc="upper center", bbox_to_anchor=(0.5, -0.28),
               ncol=len(short_labels), frameon=False)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(out_dir / filename, bbox_inches="tight")
    plt.close(fig)


def plot_valence_heatmap(df: pd.DataFrame, value_col: str, title: str,
                          filename: str, out_dir: Path) -> None:
    labels = class_order(df)
    periods = sorted(df["period"].unique())
    period_labels = [period_label(p) for p in periods]

    data = (df.groupby(["classe_label", "period"])[value_col].mean()
            .unstack("period").reindex(index=labels, columns=periods))
    data.columns = period_labels

    fig, ax = plt.subplots(figsize=(11, max(3, len(labels) * 1.1 + 1)))
    sns.heatmap(data, annot=True, fmt=".3f", cmap="coolwarm", center=0,
                linewidths=0.5, ax=ax, cbar_kws={"label": "Mean valence"})
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("Period")
    ax.set_ylabel("Class")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(out_dir / filename)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent.parent.parent
    default_rob = script_dir / "results" / "roberta_output" / "emotions_per_doc.csv"
    default_nrc = script_dir / "results" / "nrclex_output" / "emotions_per_doc.csv"
    default_out = script_dir / "results" / "temporal_output"

    p = argparse.ArgumentParser(
        description="Temporal dynamics (size and valence) of the HDC classes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--roberta-per-doc", type=Path, default=default_rob,
                   help="RoBERTa emotions_per_doc.csv CSV (with classe/classe_label)")
    p.add_argument("--nrclex-per-doc", type=Path, default=default_nrc,
                   help="NRCLex emotions_per_doc.csv CSV (no class; joined via id)")
    p.add_argument("--output-dir", type=Path, default=default_out,
                   help="Output directory for CSVs and plots")
    p.add_argument("--no-plots", action="store_true", help="Skips plot generation")
    return p.parse_args()


def main() -> None:
    args = build_args()

    for path, label in [(args.roberta_per_doc, "RoBERTa"), (args.nrclex_per_doc, "NRCLex")]:
        if not path.exists():
            print(f"[error] {label} CSV not found: {path}", file=sys.stderr)
            sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading RoBERTa: {args.roberta_per_doc}")
    print(f"Reading NRCLex:  {args.nrclex_per_doc}")
    rob_df, nrc_df = load_data(args.roberta_per_doc, args.nrclex_per_doc)
    print(f"  {len(rob_df)} classified documents (RoBERTa), "
          f"{len(nrc_df)} classified documents (NRCLex), "
          f"{rob_df['period'].nunique()} periods ({period_label(rob_df['period'].min())} to "
          f"{period_label(rob_df['period'].max())}).")

    # ---- Question 1: class size over the year ----
    print("\n=== Question 1: class size over the year ===")
    counts_wide, pct_wide, chi2_df, trend_size_df = compute_class_size(rob_df)
    class_size_long = build_class_size_long(counts_wide, pct_wide)

    class_size_path = args.output_dir / "class_size_by_period.csv"
    chi2_path = args.output_dir / "class_size_chi2.csv"
    trend_size_path = args.output_dir / "class_size_trend.csv"
    class_size_long.to_csv(class_size_path, index=False)
    chi2_df.to_csv(chi2_path, index=False)
    trend_size_df.to_csv(trend_size_path, index=False)
    print(f"  Saved: {class_size_path}")
    print(f"  Saved: {chi2_path}")
    print(f"  Saved: {trend_size_path}")

    print("\n--- % of documents by class and period ---")
    print(pct_wide.round(1).to_string())
    print("\n--- Chi-square (period × class) ---")
    print(chi2_df.to_string(index=False))
    print("\n--- Trend (Spearman) of each class's % over the periods ---")
    print(trend_size_df.to_string(index=False))

    # ---- Question 2: valence within each class over the year ----
    print("\n=== Question 2: valence within each class over the year ===")
    valence_by_class_period = compute_valence_by_class_period(rob_df, nrc_df)
    valence_path = args.output_dir / "valence_by_class_period.csv"
    valence_by_class_period.to_csv(valence_path, index=False)
    print(f"  Saved: {valence_path}")

    anova_rob = compute_valence_anova_by_class(rob_df, "roberta_valence", "roberta")
    anova_nrc = compute_valence_anova_by_class(nrc_df, "nrclex_valence", "nrclex")
    anova_df = pd.concat([anova_rob, anova_nrc], ignore_index=True)
    anova_path = args.output_dir / "valence_anova_by_class.csv"
    anova_df.to_csv(anova_path, index=False)
    print(f"  Saved: {anova_path}")

    trend_rob = compute_valence_trend_by_class(rob_df, "roberta_valence", "roberta")
    trend_nrc = compute_valence_trend_by_class(nrc_df, "nrclex_valence", "nrclex")
    trend_df = pd.concat([trend_rob, trend_nrc], ignore_index=True)
    trend_path = args.output_dir / "valence_trend_by_class.csv"
    trend_df.to_csv(trend_path, index=False)
    print(f"  Saved: {trend_path}")

    sept_may_rob = compute_sept_vs_may(rob_df, "roberta_valence", "roberta")
    sept_may_nrc = compute_sept_vs_may(nrc_df, "nrclex_valence", "nrclex")
    sept_may_df = pd.concat([sept_may_rob, sept_may_nrc], ignore_index=True)
    sept_may_path = args.output_dir / "valence_sept_vs_may.csv"
    sept_may_df.to_csv(sept_may_path, index=False)
    print(f"  Saved: {sept_may_path}")

    print("\n--- ANOVA/Kruskal-Wallis (valence ~ period) by class ---")
    print(anova_df.to_string(index=False))
    print("\n--- Trend (Spearman, document level) of valence over the year ---")
    print(trend_df.to_string(index=False))
    print("\n--- Directed comparison: September/2025 vs May/2026 ---")
    print(sept_may_df.to_string(index=False))

    # ---- plots ----
    if not args.no_plots:
        print("\nGenerating plots…")
        _set_style()

        plot_class_size_share(pct_wide, args.output_dir)
        print(f"  {args.output_dir}/plot_class_size_share.png")
        plot_class_size_counts(counts_wide, args.output_dir)
        print(f"  {args.output_dir}/plot_class_size_counts.png")

        plot_valence_by_class_period(
            rob_df, "roberta_valence",
            "Mean RoBERTa valence by HDC class over the year",
            "plot_valence_by_class_period_roberta.png", args.output_dir)
        print(f"  {args.output_dir}/plot_valence_by_class_period_roberta.png")

        plot_valence_by_class_period(
            nrc_df, "nrclex_valence",
            "Mean NRCLex valence by HDC class over the year",
            "plot_valence_by_class_period_nrclex.png", args.output_dir)
        print(f"  {args.output_dir}/plot_valence_by_class_period_nrclex.png")

        plot_valence_heatmap(
            rob_df, "roberta_valence",
            "Heatmap: class × period (RoBERTa valence)",
            "plot_valence_heatmap_roberta.png", args.output_dir)
        print(f"  {args.output_dir}/plot_valence_heatmap_roberta.png")

        plot_valence_heatmap(
            nrc_df, "nrclex_valence",
            "Heatmap: class × period (NRCLex valence)",
            "plot_valence_heatmap_nrclex.png", args.output_dir)
        print(f"  {args.output_dir}/plot_valence_heatmap_nrclex.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
