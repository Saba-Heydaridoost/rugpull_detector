"""Precision / recall / F1 table: all baselines x {node, node_matched, graph, node_graph} plus GNNs.

Test partition, original cohort, no_mask, Panel and Full scopes. Within each scope and window every model is
scored on the intersection of token IDs shared by all rows (cross-family alignment, paper Sec. 4.7).
Thresholded metrics use each model's saved y_pred (Validation-F1-optimal threshold). Seed mean and SD.
Ranks: best = bold, second = italic, per scope x window x metric, excluding the dummy reference.

usage: python gnn_baseline_prf_table.py [FULL_BASELINE_DIR]
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

BASE = Path(__file__).resolve().parent
GNN_DIR = BASE.parent / "trainer" / "results" / "gnn"
BASELINE_DIRS = {"panel": BASE / "input" / "baseline_panel",
                 "full": Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / "input" / "baseline_full"}
OUT = BASE / "output" / "test" / "gnn_vs_baseline_ablations"

GNNS = {"gcn": "GCN", "graphsage": "GraphSAGE", "gat": "GAT"}
BASELINES = {"logistic_regression": "Logistic regression", "decision_tree": "Decision tree",
             "random_forest": "Random forest", "gradient_boosting": "Gradient boosting", "xgboost": "XGBoost",
             "mlp": "MLP", "ft_transformer": "FT-Transformer", "autoint": "AutoInt", "dummy": "Dummy (prior)"}
VIEWS = {"node": "Node", "node_matched": "Node-matched", "graph": "Graph", "node_graph": "Node–graph"}
WINDOWS = ["q1", "q2", "q3"]
METRICS = {"precision": precision_score, "recall": recall_score, "f1": f1_score}


def path_for(scope, model, view, q):
    if model in GNNS:
        return GNN_DIR / f"final_predictions_{model}_{scope}_{q}_gnn_graph_structure_timing_node_graph_no_mask_original.csv"
    return BASELINE_DIRS[scope] / f"final_predictions_{model}_{scope}_{q}_{view}_no_mask_original.csv"


def load(path):
    d = pd.read_csv(path)
    d = d[d.partition == "test"]
    pred = d.pivot(index="token_mint", columns="seed", values="y_pred")
    assert pred.shape[1] == 5 and not pred.isna().any().any(), path
    return d.groupby("token_mint").y_true.first(), pred


def rows_for(scope):
    specs = [("GNN", m, "node_graph") for m in GNNS] + [(VIEWS[v], m, v) for v in VIEWS for m in BASELINES]
    out = []
    for q in WINDOWS:
        frames = {(fam, m): load(path_for(scope, m, v, q)) for fam, m, v in specs}
        ids = sorted(set.intersection(*(set(y.index) for y, _ in frames.values())))
        for (fam, m), (y, pred) in frames.items():
            yy, pp = y.loc[ids].values, pred.loc[ids]
            row = {"scope": scope, "q": q, "family": fam, "model": m, "n": len(ids), "n_pos": int(yy.sum())}
            for k, fn in METRICS.items():
                vals = [fn(yy, pp[s].values, zero_division=0) for s in range(5)]
                row[k], row[f"{k}_sd"] = np.mean(vals), np.std(vals, ddof=1)
            out.append(row)
    return out


def rank_marks(df):
    """Dense rank on 3-decimal values per scope x window x metric, dummy excluded."""
    df = df.copy()
    for k in METRICS:
        rounded =df[df.model != "dummy"].assign(v=lambda x: x[k].round(3)).groupby(["scope", "q"]).v.rank(
            method="dense", ascending=False)
        df[f"{k}_rank"] = rounded.reindex(df.index)
    return df


def fmt(v, rank, style):
    s = f"{v:.3f}"
    if rank == 1:
        return {"md": f"**{s}**", "tex": rf"\textbf{{{s}}}"}[style]
    if rank == 2:
        return {"md": f"*{s}*", "tex": rf"\textit{{{s}}}"}[style]
    return s


def render(df, scope, style):
    sub = df[df.scope == scope]
    order = [f for f in ["GNN", MERGED] + list(VIEWS.values()) if f in set(sub.family)]
    lines = []
    head = [f"{q} {k}" for q in WINDOWS for k in ("P", "R", "F1")]
    if style == "md":
        lines.append("| Feature view | Model | " + " | ".join(head) + " |")
        lines.append("|---|---|" + "---|" * len(head))
    for fam in order:
        models = GNNS if fam == "GNN" else BASELINES
        for m, label in models.items():
            if m == "dummy":
                continue
            cells = []
            for q in WINDOWS:
                r = sub[(sub.family == fam) & (sub.model == m) & (sub.q == q)].iloc[0]
                cells += [fmt(r[k], r[f"{k}_rank"], style) for k in METRICS]
            if style == "md":
                lines.append(f"| {fam} | {label} | " + " | ".join(cells) + " |")
            else:
                lines.append(f"{fam.replace('–', '--')} & {label} & " + " & ".join(cells) + r" \\")
        if style == "tex":
            lines.append(r"\midrule")
    return "\n".join(lines)


def collapse_identical_node_views(df):
    """On a scope where node and node_matched rows coincide (Panel), keep one block labelled with both."""
    keep = []
    for scope, sub in df.groupby("scope"):
        node = sub[sub.family == VIEWS["node"]].set_index(["q", "model"])[list(METRICS)]
        matched = sub[sub.family == VIEWS["node_matched"]].set_index(["q", "model"])[list(METRICS)]
        if np.allclose(node.sort_index().values, matched.sort_index().values):
            sub = sub[sub.family != VIEWS["node_matched"]].replace({"family": {VIEWS["node"]: MERGED}})
        keep.append(sub)
    return pd.concat(keep, ignore_index=True)


MERGED = "Node / Node-matched"


def main():
    df = pd.DataFrame(rows_for("panel") + rows_for("full"))
    df = rank_marks(collapse_identical_node_views(df[df.model != "dummy"]))
    df.to_csv(OUT / "I_prf_table.csv", index=False)
    with open(OUT / "I_prf_table.md", "w") as fh, open(OUT / "I_prf_table.tex", "w") as th:
        for scope in ("panel", "full"):
            ns = df[df.scope == scope].groupby("q")[["n", "n_pos"]].first()
            cap = ", ".join(f"{q}: n={r.n}, {r.n_pos} rug" for q, r in ns.iterrows())
            fh.write(f"### {scope.capitalize()} ({cap})\n\n{render(df, scope, 'md')}\n\n")
            th.write(f"% {scope} ({cap})\n{render(df, scope, 'tex')}\n\n")
    print(open(OUT / "I_prf_table.md").read())


if __name__ == "__main__":
    main()
