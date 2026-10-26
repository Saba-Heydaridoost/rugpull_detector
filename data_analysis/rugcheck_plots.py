"""Heatmaps relating RugCheck risk level, graph node features and rule-based labels.

Figures (data_analysis/output/plots/rugcheck/):
  1_level_vs_label.png           level x label crosstab + Spearman of level features with label
  2_level_vs_node_features.png   Spearman: RugCheck level features x node features (q1)
  3_node_features_vs_label.png   Spearman of node features with label, per experiment
  4_node_feature_distributions.png  node feature distributions, negative vs positive label (q1)

All correlations are Spearman (node features are heavy-tailed). Cells whose
Benjamini-Hochberg adjusted p >= 0.05 are printed in muted ink.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
RC = ROOT / "dataset" / "rugcheck"
OUT = Path(__file__).resolve().parent / "output" / "plots" / "rugcheck"

INK, INK_2, MUTED, GRID, SURFACE = "#1f1f1e", "#52524f", "#a3a29d", "#e4e3df", "#fcfcfb"
NEG, POS = "#2a78d6", "#eb6834"  # categorical slots 1, 2
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_red", ["#184f95", "#2a78d6", "#f0efec", "#e34948", "#9e2b2b"])
SEQUENTIAL = LinearSegmentedColormap.from_list("blue_seq", ["#f0f5fd", "#9ec5f4", "#2a78d6", "#104281"])

LEVEL_FEATURES = {
    "max_risk_level_ord": "Max risk level (none/warn/danger)",
    "any_danger": "Has danger risk",
    "n_danger": "# danger risks",
    "n_warn": "# warn risks",
    "report.score_normalised": "RugCheck score (normalised)",
}
NODE_FEATURES = {
    "volume_sell_change_percent": "Sell volume change %",
    "volume_buy_change_percent": "Buy volume change %",
    "unique_wallet_count": "Unique wallets",
    "unique_wallet_change_percent": "Unique wallet change %",
    "liquidity_at_creation": "Liquidity at creation",
    "liquidity_at_observation_end": "Liquidity at obs. end",
    "has_freeze_authority": "Freeze authority set",
    "n_creators": "# creators",
}

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.titleweight": "bold",
    "axes.titlesize": 10, "axes.titlelocation": "left",
})


def spearman_table(df, xs, ys):
    r = pd.DataFrame(index=xs, columns=ys, dtype=float)
    p, n = r.copy(), r.copy()
    for x in xs:
        for y in ys:
            d = df[[x, y]].dropna()
            n.loc[x, y] = len(d)
            if len(d) > 10 and d[x].nunique() > 1 and d[y].nunique() > 1:
                r.loc[x, y], p.loc[x, y] = stats.spearmanr(d[x], d[y])
    flat = p.values.ravel()
    ok = ~np.isnan(flat)
    adj = np.full_like(flat, np.nan)
    adj[ok] = multipletests(flat[ok], method="fdr_bh")[1]
    return r, pd.DataFrame(adj.reshape(p.shape), index=xs, columns=ys), n


def heatmap(ax, r, padj, row_labels, col_labels, vmax=0.5, cbar=True):
    im = ax.imshow(r.values.astype(float), cmap=DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            v = r.iat[i, j]
            if np.isnan(v):
                ax.text(j, i, "n/a", ha="center", va="center", color=MUTED, fontsize=8)
                continue
            sig = padj.iat[i, j] < 0.05
            color = "white" if abs(v) > 0.6 * vmax else (INK if sig else MUTED)
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", color=color, fontsize=8,
                    fontweight="bold" if sig else "normal")
    ax.set_xticks(range(len(col_labels)), col_labels, rotation=35, ha="right")
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, r.shape[1]), minor=True)
    ax.set_yticks(np.arange(-.5, r.shape[0]), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    if cbar:
        cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.outline.set_visible(False)
        cb.set_label("Spearman ρ", color=INK_2)
    return im


def load():
    rc = pd.read_csv(RC / "rugcheck_reports.csv", low_memory=False,
                     usecols=["mint", "ok", "max_risk_level", "max_risk_level_ord", "n_danger",
                              "n_warn", "report.score_normalised"])
    rc = rc[rc.ok].copy()
    rc["any_danger"] = (rc.max_risk_level_ord == 2).astype(int)

    nodes = pd.read_csv(ROOT / "dataset" / "graph_nodes.csv")
    nodes["has_freeze_authority"] = nodes.freeze_authority.notna().astype(int)
    nodes["n_creators"] = nodes.creators.map(lambda s: len(json.loads(s)) if isinstance(s, str) else np.nan)

    lab = pd.read_csv(ROOT / "dataset" / "rule_based_labels.csv",
                      usecols=["token_mint", "experiment_id", "label_status", "rug_pull_label"])
    lab = lab[lab.label_status == "VALID"].drop(columns="label_status")
    return rc, nodes, lab


def fig_level_vs_label(rc, lab):
    tok = lab.groupby("token_mint", as_index=False).rug_pull_label.first()
    d = tok.merge(rc, left_on="token_mint", right_on="mint")
    ct = pd.crosstab(d.max_risk_level, d.rug_pull_label).reindex(["none", "warn", "danger"])
    share = ct.div(ct.sum(axis=1), axis=0)

    exps = sorted(lab.experiment_id.unique())
    r = pd.DataFrame(index=list(LEVEL_FEATURES), columns=exps, dtype=float)
    p = r.copy()
    for q in exps:
        dq = lab[lab.experiment_id == q].merge(rc, left_on="token_mint", right_on="mint")
        rq, pq, _ = spearman_table(dq, list(LEVEL_FEATURES), ["rug_pull_label"])
        r[q], p[q] = rq.iloc[:, 0], pq.iloc[:, 0]

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1.25]})
    im = a.imshow(share.values, cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    for i in range(share.shape[0]):
        for j in range(share.shape[1]):
            v = share.iat[i, j]
            a.text(j, i, f"{v:.0%}\n(n={ct.iat[i, j]})", ha="center", va="center", fontsize=9,
                   color="white" if v > 0.55 else INK)
    a.set_xticks([0, 1], ["Negative (not rug)", "Positive (rug)"])
    a.set_yticks(range(3), [f"{lvl}  (n={ct.loc[lvl].sum()})" for lvl in share.index])
    a.set_ylabel("RugCheck max risk level")
    a.tick_params(length=0)
    for s in a.spines.values():
        s.set_visible(False)
    a.set_title("Rule-based label split within each risk level")
    cb = plt.colorbar(im, ax=a, fraction=0.04, pad=0.02, format=lambda x, _: f"{x:.0%}")
    cb.outline.set_visible(False)
    cb.set_label("Row share", color=INK_2)

    padj = pd.DataFrame(multipletests(p.values.ravel(), method="fdr_bh")[1].reshape(p.shape),
                        index=p.index, columns=p.columns)
    heatmap(b, r, padj, list(LEVEL_FEATURES.values()),
            [f"{q} (n={lab[lab.experiment_id == q].token_mint.isin(rc.mint).sum()})" for q in exps])
    b.set_title("Spearman ρ with rug_pull_label, per experiment")
    fig.tight_layout()
    fig.savefig(OUT / "1_level_vs_label.png", dpi=200)
    plt.close(fig)


def fig_level_vs_nodes(rc, nodes, q="q1"):
    d = nodes[nodes.experiment_id == q].merge(rc, left_on="token_mint", right_on="mint")
    r, padj, n = spearman_table(d, list(LEVEL_FEATURES), list(NODE_FEATURES))
    fig, ax = plt.subplots(figsize=(10, 4))
    heatmap(ax, r, padj, list(LEVEL_FEATURES.values()),
            [f"{lbl}\n(n={int(n[c].max())})" for c, lbl in NODE_FEATURES.items()])
    ax.set_title(f"RugCheck level features vs graph node features ({q}, {len(d)} tokens) — Spearman ρ")
    fig.tight_layout()
    fig.savefig(OUT / "2_level_vs_node_features.png", dpi=200)
    plt.close(fig)


def fig_nodes_vs_label(nodes):
    exps = sorted(nodes.experiment_id.unique())
    r = pd.DataFrame(index=list(NODE_FEATURES), columns=exps, dtype=float)
    p, n = r.copy(), r.copy()
    for q in exps:
        rq, pq, nq = spearman_table(nodes[nodes.experiment_id == q], list(NODE_FEATURES), ["rug_pull_label"])
        r[q], p[q], n[q] = rq.iloc[:, 0], pq.iloc[:, 0], nq.iloc[:, 0]
    ok = ~np.isnan(p.values.ravel())
    adj = np.full(p.size, np.nan)
    adj[ok] = multipletests(p.values.ravel()[ok], method="fdr_bh")[1]
    padj = pd.DataFrame(adj.reshape(p.shape), index=p.index, columns=p.columns)

    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    heatmap(ax, r, padj, [f"{lbl}" for lbl in NODE_FEATURES.values()],
            [f"{q} (n={(nodes.experiment_id == q).sum()})" for q in exps], vmax=0.4)
    ax.set_title("Node features vs rug_pull_label — Spearman ρ")
    fig.tight_layout()
    fig.savefig(OUT / "3_node_features_vs_label.png", dpi=200)
    plt.close(fig)


def fig_node_distributions(nodes, q="q1"):
    d = nodes[nodes.experiment_id == q]
    cont = [c for c in NODE_FEATURES if d[c].nunique() > 2]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, c in zip(axes.ravel(), cont):
        groups = [d.loc[d.rug_pull_label == k, c].dropna() for k in (0, 1)]
        bp = ax.boxplot(groups, vert=False, widths=0.55, showfliers=False, patch_artist=True,
                        medianprops={"color": INK, "linewidth": 2}, whiskerprops={"color": INK_2},
                        capprops={"color": INK_2})
        for patch, col in zip(bp["boxes"], (NEG, POS)):
            patch.set(facecolor=col, edgecolor=SURFACE, linewidth=2, alpha=0.85)
        u = stats.mannwhitneyu(*groups) if all(len(g) for g in groups) else None
        ax.set_xscale("symlog", linthresh=100)
        ax.set_yticks([1, 2], [f"Negative\n(n={len(groups[0])})", f"Positive\n(n={len(groups[1])})"])
        ax.set_title(NODE_FEATURES[c])
        if u is not None:
            ax.text(0.99, 0.02, f"Mann-Whitney p={u.pvalue:.1g}", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8, color=INK_2)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
    for ax in axes.ravel()[len(cont):]:
        ax.set_visible(False)
    fig.suptitle(f"Node feature distributions by rule-based label ({q}; boxes = IQR, whiskers = 1.5·IQR, "
                 "outliers hidden; symlog x)", x=0.01, ha="left", fontweight="bold", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "4_node_feature_distributions.png", dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rc, nodes, lab = load()
    fig_level_vs_label(rc, lab)
    fig_level_vs_nodes(rc, nodes)
    fig_nodes_vs_label(nodes)
    fig_node_distributions(nodes)
    print(f"wrote figures to {OUT}")


if __name__ == "__main__":
    main()
