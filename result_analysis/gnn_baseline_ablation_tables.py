"""Exploratory GNN-vs-baseline ablation tables on the held-out Test partition.

Builds, for each ablation separately, (a) token-matched seed-mean AP/ROC-AUC tables
for all 9 baselines and 3 GNNs and (b) paired, label-stratified token bootstrap
deltas (shared draws across arms and seeds). All CIs are unadjusted 95% percentile
intervals; this is NOT the frozen primary analysis (see gnn_primary_inference.py).
"""
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

BASE = Path(__file__).resolve().parent
BASELINE_DIR = BASE / "input" / "baseline_panel"
GNN_DIR = BASE.parent / "trainer" / "results" / "gnn"
BATCH = BASE / "output" / "test" / "batch" / "reps_10000_seed_2026" / "comparison_results.csv"
OUT = BASE / "output" / "test" / "gnn_vs_baseline_ablations"
N_BOOT = 10000
RNG_SEED = 2026

GNNS = ["gcn", "graphsage", "gat"]
BASELINES = ["dummy", "logistic_regression", "decision_tree", "random_forest", "gradient_boosting",
             "xgboost", "mlp", "ft_transformer", "autoint"]
MODELS = GNNS + BASELINES
WINDOWS = ["q1", "q2", "q3"]
# GNN view -> analogous baseline view
GNN_VIEW = {"node_graph": "node_graph", "node_graph_freeze": "node_graph_plus_freeze_authority"}
CONFIGS = [("no_mask", "original"), ("with_mask", "original"), ("no_mask", "drop_3_or_more_missing"),
           ("with_mask", "drop_3_or_more_missing"), ("no_mask", "complete_case")]

_cache = {}


def load(model, scope, q, view, variant, cohort):
    """Return DataFrame indexed by token_mint: y_true + one score column per seed (test only)."""
    key = (model, scope, q, view, variant, cohort)
    if key in _cache:
        return _cache[key]
    if model in GNNS:
        if view not in GNN_VIEW:
            _cache[key] = None
            return None
        path = GNN_DIR / (f"final_predictions_{model}_{scope}_{q}_gnn_graph_structure_timing_"
                          f"{GNN_VIEW[view]}_{variant}_{cohort}.csv")
    else:
        path = BASELINE_DIR / f"final_predictions_{model}_{scope}_{q}_{view}_{variant}_{cohort}.csv"
    if not path.is_file():
        _cache[key] = None
        return None
    d = pd.read_csv(path)
    d = d[d.partition == "test"]
    wide = d.pivot(index="token_mint", columns="seed", values="y_score")
    y = d.groupby("token_mint").y_true.agg(["min", "max"])
    assert (y["min"] == y["max"]).all(), path
    wide.insert(0, "y_true", y["min"])
    if wide.shape[1] != 6 or wide.isna().any().any():
        print(f"INCOMPLETE (skipped): {path.name} test seeds={sorted(d.seed.unique())}")
        wide = None
    _cache[key] = wide
    return wide


def matched(frames):
    ids = sorted(set.intersection(*(set(f.index) for f in frames)))
    out = [f.loc[ids] for f in frames]
    for f in out[1:]:
        assert (f.y_true.values == out[0].y_true.values).all()
    return ids, out


def seed_mean(f):
    y = f.y_true.values
    ap = np.mean([average_precision_score(y, f[s].values) for s in range(5)])
    roc = np.mean([roc_auc_score(y, f[s].values) for s in range(5)])
    return ap, roc


def weighted_ap(y, score, W):
    """AP (sklearn step definition, ties grouped) for every row of count matrix W (reps x n)."""
    order = np.argsort(-score, kind="mergesort")
    s, yy, Ws = score[order], y[order], W[:, order]
    tp = np.cumsum(Ws * yy, axis=1)
    fp = np.cumsum(Ws * (1 - yy), axis=1)
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]  # last index of each tie group
    tp, fp = tp[:, last], fp[:, last]
    prec = tp / np.maximum(tp + fp, 1e-12)
    rec = tp / tp[:, -1:]
    drec = np.diff(np.concatenate([np.zeros((len(W), 1)), rec], axis=1), axis=1)
    return (drec * prec).sum(axis=1)


def boot_counts(y, rng):
    n = len(y)
    W = np.zeros((N_BOOT, n))
    for cls in (0, 1):
        idx = np.nonzero(y == cls)[0]
        W[:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), size=N_BOOT)
    return W


def paired(fa, fb, rng):
    """Delta AP (A-B), seed-mean, with paired stratified bootstrap 95% CI over matched tokens."""
    ids, (fa, fb) = matched([fa, fb])
    y = fa.y_true.values.astype(float)
    ap_a, roc_a = seed_mean(fa)
    ap_b, roc_b = seed_mean(fb)
    W = boot_counts(y, rng)
    da = np.mean([weighted_ap(y, fa[s].values, W) for s in range(5)], axis=0)
    db = np.mean([weighted_ap(y, fb[s].values, W) for s in range(5)], axis=0)
    lo, hi = np.percentile(da - db, [2.5, 97.5])
    return dict(n=len(ids), n_pos=int(y.sum()), ap_a=ap_a, ap_b=ap_b, delta_ap=ap_a - ap_b,
                ci_low=lo, ci_high=hi, roc_a=roc_a, roc_b=roc_b, delta_roc=roc_a - roc_b,
                ci_excludes_0=bool(lo > 0 or hi < 0))


def level_table(cells, name):
    """cells: list of (label dict, view, variant, cohort, q). AP on tokens shared by all 12 models."""
    rows = []
    for lab, view, variant, cohort, q in cells:
        frames = {m: load(m, "panel", q, view, variant, cohort) for m in MODELS}
        frames = {m: f for m, f in frames.items() if f is not None}
        ids, fs = matched(list(frames.values()))
        for m, f in zip(frames, fs):
            ap, roc = seed_mean(f)
            rows.append({**lab, "q": q, "model": m, "family": "GNN" if m in GNNS else "baseline",
                         "n": len(ids), "n_pos": int(f.y_true.sum()), "ap": ap, "roc_auc": roc})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"{name}_levels.csv", index=False)
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    # A. Temporal (Panel, original/no_mask, node_graph): levels + within-model window deltas
    level_table([({"setting": "node_graph/no_mask/original"}, "node_graph", "no_mask", "original", q)
                 for q in WINDOWS], "A_temporal")
    rows = []
    for m in MODELS:
        for qa, qb in [("q2", "q1"), ("q3", "q1"), ("q3", "q2")]:
            r = paired(load(m, "panel", qa, "node_graph", "no_mask", "original"),
                       load(m, "panel", qb, "node_graph", "no_mask", "original"), rng)
            rows.append({"model": m, "contrast": f"{qa}-{qb}", **r})
    pd.DataFrame(rows).to_csv(OUT / "A_temporal_deltas.csv", index=False)

    # B. GNN vs each baseline, same window, every view/variant/cohort config
    rows = []
    for view in GNN_VIEW:
        for variant, cohort in CONFIGS:
            for q in WINDOWS:
                for g in GNNS:
                    for b in BASELINES:
                        fa, fb = load(g, "panel", q, view, variant, cohort), load(b, "panel", q, view, variant, cohort)
                        rows.append({"view": view, "variant": variant, "cohort": cohort, "q": q,
                                     "gnn": g, "baseline": b, **paired(fa, fb, rng)})
    pd.DataFrame(rows).to_csv(OUT / "B_gnn_minus_baseline_all_configs.csv", index=False)

    # C. Representation: baseline views (node, node_matched, graph, node_graph, node_graph_freeze); GNN two views
    level_table([({"view": v}, v, "no_mask", "original", q)
                 for v in ["node", "node_matched", "graph", "node_graph", "node_graph_freeze"] for q in WINDOWS],
                "C_representation")
    rows = []
    for m in MODELS:
        for q in WINDOWS:
            r = paired(load(m, "panel", q, "node_graph_freeze", "no_mask", "original"),
                       load(m, "panel", q, "node_graph", "no_mask", "original"), rng)
            rows.append({"model": m, "q": q, "contrast": "freeze-node_graph", **r})
    pd.DataFrame(rows).to_csv(OUT / "C_freeze_deltas.csv", index=False)

    # D. Missingness mask (with - no mask), same cohort, node_graph
    level_table([({"variant": v, "cohort": c}, "node_graph", v, c, q)
                 for v, c in CONFIGS[:4] for q in WINDOWS], "D_mask")
    rows = []
    for m in MODELS:
        for cohort in ["original", "drop_3_or_more_missing"]:
            for q in WINDOWS:
                r = paired(load(m, "panel", q, "node_graph", "with_mask", cohort),
                           load(m, "panel", q, "node_graph", "no_mask", cohort), rng)
                rows.append({"model": m, "cohort": cohort, "q": q, "contrast": "with_mask-no_mask", **r})
    pd.DataFrame(rows).to_csv(OUT / "D_mask_deltas.csv", index=False)

    # E. Cohort/cleaning (no_mask, node_graph): cleaned-cohort model minus original-cohort model on shared tokens
    level_table([({"cohort": c}, "node_graph", "no_mask", c, q)
                 for c in ["original", "drop_3_or_more_missing", "complete_case"] for q in WINDOWS], "E_cohort")
    rows = []
    for m in MODELS:
        for cohort in ["drop_3_or_more_missing", "complete_case"]:
            for q in WINDOWS:
                r = paired(load(m, "panel", q, "node_graph", "no_mask", cohort),
                           load(m, "panel", q, "node_graph", "no_mask", "original"), rng)
                rows.append({"model": m, "cohort": cohort, "q": q, "contrast": f"{cohort}-original", **r})
    pd.DataFrame(rows).to_csv(OUT / "E_cohort_deltas.csv", index=False)

    full_scope()


def full_scope():
    """F. Full scope: GNN from predictions; baselines only from batch summary (no token IDs -> no paired CI)."""
    rows = []
    for g in GNNS:
        for view, gv in GNN_VIEW.items():
            for variant, cohort in CONFIGS:
                for q in WINDOWS:
                    f = load(g, "full", q, view, variant, cohort)
                    if f is None:
                        continue
                    ap, roc = seed_mean(f)
                    rows.append({"model": g, "view": view, "variant": variant, "cohort": cohort, "q": q,
                                 "n": len(f), "n_pos": int(f.y_true.sum()), "ap": ap, "roc_auc": roc,
                                 "source": "final_predictions"})
    batch = pd.read_csv(BATCH)
    batch = batch[(batch.scope == "full") & (batch.family == "cleaning") & (batch.comparison == "mask_effect")
                  & (batch.n_paired == batch.a_n_tokens) & (batch.n_paired == batch.b_n_tokens)]
    for _, r in batch.iterrows():
        for arm, variant in (("a", "with_mask"), ("b", "no_mask")):
            rows.append({"model": r["model"], "view": r["view"], "variant": variant, "cohort": r["cohort"], "q": r["q"],
                         "n": r.n_paired, "n_pos": r.n_positive, "ap": r[f"ap_{arm}"], "roc_auc": r[f"roc_{arm}"],
                         "source": "batch_summary"})
    full = pd.DataFrame(rows).drop_duplicates(["model", "view", "variant", "cohort", "q"])
    full.to_csv(OUT / "F_full_scope_levels.csv", index=False)


if __name__ == "__main__":
    main()
