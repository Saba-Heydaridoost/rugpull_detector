from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu


ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / "output" / "graph_metrics_per_token.csv"
OUTPUT_DIR = ROOT / "output"
PLOTS_DIR = OUTPUT_DIR / "plots"

Q_ORDER = ["q1", "q2", "q3"]

METRICS = [
    "n_nodes",
    "n_edges",
    "density",
    "max_in_degree_centrality",
    "max_out_degree_centrality",
    "max_betweenness_centrality",
    "max_pagerank",
    "reciprocity",
    "largest_wcc_ratio",
    "top1_degree_share",
    "top5_degree_share",
]

PLOT_METRICS = [
    "n_nodes",
    "density",
    "max_in_degree_centrality",
    "max_betweenness_centrality",
    "max_pagerank",
    "reciprocity",
]


def _bh_adjust(p_values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg FDR correction."""
    p = p_values.astype(float).to_numpy()
    n = len(p)

    order = np.argsort(p)
    ranked = p[order]

    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    result = np.empty(n, dtype=float)
    result[order] = adjusted
    return pd.Series(result, index=p_values.index)


def _iqr_string(values: pd.Series) -> str:
    q25 = values.quantile(0.25)
    q75 = values.quantile(0.75)
    return f"[{q25:.6g}, {q75:.6g}]"


def _rank_biserial_from_u(
    u_stat: float,
    n_rug: int,
    n_nonrug: int,
) -> float:
    """
    Rank-biserial correlation using U for the Rug sample.

    Positive -> Rug values tend to be larger.
    Negative -> Non-rug values tend to be larger.
    """
    return (2.0 * u_stat / (n_rug * n_nonrug)) - 1.0


def compare_groups(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for q in Q_ORDER:
        q_df = df[df["q"] == q].copy()

        for metric in METRICS:
            rug = pd.to_numeric(
                q_df.loc[q_df["rug_pull_label"] == 1, metric],
                errors="coerce",
            ).dropna()

            nonrug = pd.to_numeric(
                q_df.loc[q_df["rug_pull_label"] == 0, metric],
                errors="coerce",
            ).dropna()

            if len(rug) == 0 or len(nonrug) == 0:
                raise ValueError(
                    f"Missing Rug or Non-rug observations for {q}, {metric}"
                )

            test = mannwhitneyu(
                rug,
                nonrug,
                alternative="two-sided",
                method="auto",
            )

            effect = _rank_biserial_from_u(
                u_stat=float(test.statistic),
                n_rug=len(rug),
                n_nonrug=len(nonrug),
            )

            rows.append(
                {
                    "q": q,
                    "metric": metric,
                    "n_nonrug": len(nonrug),
                    "nonrug_median": nonrug.median(),
                    "nonrug_q25": nonrug.quantile(0.25),
                    "nonrug_q75": nonrug.quantile(0.75),
                    "nonrug_iqr": _iqr_string(nonrug),
                    "n_rug": len(rug),
                    "rug_median": rug.median(),
                    "rug_q25": rug.quantile(0.25),
                    "rug_q75": rug.quantile(0.75),
                    "rug_iqr": _iqr_string(rug),
                    "mann_whitney_u": float(test.statistic),
                    "p_value": float(test.pvalue),
                    "rank_biserial": effect,
                    "effect_direction": (
                        "Rug higher"
                        if effect > 0
                        else "Non-rug higher"
                        if effect < 0
                        else "No direction"
                    ),
                }
            )

    result = pd.DataFrame(rows)
    result["p_fdr_bh"] = _bh_adjust(result["p_value"])
    return result


def plot_metric(df: pd.DataFrame, metric: str) -> None:
    """
    One figure per metric.
    For each q: Non-rug and Rug side-by-side boxplots.
    """
    data = []
    positions = []
    labels = []

    pos = 1
    gap = 1

    for q in Q_ORDER:
        q_df = df[df["q"] == q]

        nonrug = pd.to_numeric(
            q_df.loc[q_df["rug_pull_label"] == 0, metric],
            errors="coerce",
        ).dropna()

        rug = pd.to_numeric(
            q_df.loc[q_df["rug_pull_label"] == 1, metric],
            errors="coerce",
        ).dropna()

        data.extend([nonrug.to_numpy(), rug.to_numpy()])
        positions.extend([pos, pos + 1])
        labels.extend([f"{q}\nNon-rug", f"{q}\nRug"])
        pos += 2 + gap

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.boxplot(
        data,
        positions=positions,
        widths=0.7,
        showfliers=False,
        medianprops={"linewidth": 2},
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric}: Rug vs Non-rug across q1-q3")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{metric}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    required = {"q", "rug_pull_label", *METRICS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df["rug_pull_label"] = pd.to_numeric(
        df["rug_pull_label"], errors="raise"
    ).astype(int)

    labels = set(df["rug_pull_label"].unique())
    if not labels.issubset({0, 1}):
        raise ValueError(
            f"Expected binary rug_pull_label in {{0,1}}, found: {sorted(labels)}"
        )

    comparisons = compare_groups(df)

    stats_path = OUTPUT_DIR / "statistical_comparison.csv"
    comparisons.to_csv(stats_path, index=False)

    for metric in PLOT_METRICS:
        plot_metric(df, metric)

    print("\nDone.")
    print(f"Statistical comparison: {stats_path}")
    print(f"Plots directory:        {PLOTS_DIR}")
    print("\nEffect-size sign:")
    print("  rank_biserial > 0  => Rug values tend to be larger")
    print("  rank_biserial < 0  => Non-rug values tend to be larger")


if __name__ == "__main__":
    main()