from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
PLOT_DIR = OUTPUT_DIR / "plots"
INPUT_FILE = OUTPUT_DIR / "graph_metrics_per_token.csv"

METRICS = [
    "n_nodes",
    "n_edges",
    "density",
    "largest_wcc_ratio",
    "reciprocity",
    "top1_degree_share",
    "top5_degree_share",
    "max_in_degree_centrality",
    "max_out_degree_centrality",
    "max_betweenness_centrality",
    "max_pagerank",
]

PLOT_METRICS = [
    "n_nodes",
    "density",
    "max_in_degree_centrality",
    "max_betweenness_centrality",
    "max_pagerank",
    "reciprocity",
]

Q_VALUES = ("q1", "q2", "q3")


def rank_biserial_from_u(u: float, n_rug: int, n_nonrug: int) -> float:
    if n_rug == 0 or n_nonrug == 0:
        return np.nan
    return float((2.0 * u) / (n_rug * n_nonrug) - 1.0)


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = ranked * n / np.arange(1, n + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)
    adjusted = np.empty(n, dtype=float)
    adjusted[order] = adjusted_ranked
    return adjusted


def compare_groups(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for q in Q_VALUES:
        qdf = df[df["q"] == q].copy()
        for metric in METRICS:
            rug = pd.to_numeric(
                qdf.loc[qdf["rug_pull_label"] == 1, metric], errors="coerce"
            ).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
            nonrug = pd.to_numeric(
                qdf.loc[qdf["rug_pull_label"] == 0, metric], errors="coerce"
            ).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()

            if len(rug) and len(nonrug):
                test = mannwhitneyu(rug, nonrug, alternative="two-sided", method="auto")
                u = float(test.statistic)
                p = float(test.pvalue)
                rbc = rank_biserial_from_u(u, len(rug), len(nonrug))
            else:
                u = p = rbc = np.nan

            rows.append({
                "q": q,
                "metric": metric,
                "n_rug": len(rug),
                "n_nonrug": len(nonrug),
                "rug_mean": float(np.mean(rug)) if len(rug) else np.nan,
                "rug_median": float(np.median(rug)) if len(rug) else np.nan,
                "nonrug_mean": float(np.mean(nonrug)) if len(nonrug) else np.nan,
                "nonrug_median": float(np.median(nonrug)) if len(nonrug) else np.nan,
                "mannwhitney_u": u,
                "p_value": p,
                "rank_biserial": rbc,
            })

    out = pd.DataFrame(rows)
    valid = out["p_value"].notna()
    out["p_fdr_bh"] = np.nan
    out.loc[valid, "p_fdr_bh"] = benjamini_hochberg(out.loc[valid, "p_value"].to_numpy())
    out["significant_fdr_0_05"] = out["p_fdr_bh"] < 0.05
    return out


def make_boxplot(df: pd.DataFrame, metric: str) -> None:
    labels, data = [], []
    for q in Q_VALUES:
        qdf = df[df["q"] == q]
        for label_value, label_name in ((0, "Non-rug"), (1, "Rug")):
            vals = pd.to_numeric(
                qdf.loc[qdf["rug_pull_label"] == label_value, metric], errors="coerce"
            ).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
            labels.append(f"{q}\n{label_name}")
            data.append(vals)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set_title(f"{metric} - training split only")
    ax.set_ylabel(metric)
    ax.set_xlabel("Observation window / class")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"{metric}.png", dpi=200)
    plt.close(fig)


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing {INPUT_FILE}. Run analyze_graphs.py first.")

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_FILE)

    required = {"q", "rug_pull_label", *METRICS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    print("=" * 60)
    print("RUG VS NON-RUG COMPARISON - TRAIN ONLY")
    print("=" * 60)
    print(df.groupby(["q", "rug_pull_label"]).size())

    comparison = compare_groups(df)
    comparison.to_csv(OUTPUT_DIR / "statistical_comparison.csv", index=False)

    for metric in PLOT_METRICS:
        make_boxplot(df, metric)

    print(f"\nSaved: {OUTPUT_DIR / 'statistical_comparison.csv'}")
    print(f"Saved plots to: {PLOT_DIR}")
    print()
    print(comparison[[
        "q", "metric", "n_rug", "n_nonrug", "rug_median", "nonrug_median",
        "rank_biserial", "p_value", "p_fdr_bh", "significant_fdr_0_05"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
