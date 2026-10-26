"""Exploratory GNN-vs-baseline comparison of thresholded metrics (precision, recall, F1, flag rate).

Test, Panel, node_graph view. Uses the saved y_pred, i.e. each model's Validation-F1-optimal threshold
applied unchanged to Test. Seed-mean metrics; paired label-stratified token bootstrap with one shared
draw across both arms and all five seeds. Unadjusted 95% percentile CIs.
"""
import numpy as np
import pandas as pd

from gnn_baseline_ablation_tables import BASELINES, CONFIGS, GNNS, OUT, RNG_SEED, boot_counts
from gnn_baseline_early_f1 import WINDOW_PAIRS, load

METRICS = ["precision", "recall", "f1", "flag_rate"]


def metrics(y, pred, W):
    """Thresholded metrics for each row of count matrix W (reps x n); zero_division -> 0 like sklearn."""
    tp = W @ (y * pred)
    fp = W @ ((1 - y) * pred)
    fn = W @ (y * (1 - pred))
    n = W.sum(axis=1)
    return {
        "precision": np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-12), 0.0),
        "recall": tp / np.maximum(tp + fn, 1e-12),
        "f1": 2 * tp / np.maximum(2 * tp + fp + fn, 1e-12),
        "flag_rate": (tp + fp) / n,
    }


def seed_mean(y, pred, W):
    per_seed = [metrics(y, pred[s].values.astype(float), W) for s in range(5)]
    return {k: np.mean([m[k] for m in per_seed], axis=0) for k in METRICS}


def compare(a, b, rng):
    (ya, _, pa), (yb, _, pb) = a, b
    ids = sorted(set(ya.index) & set(yb.index))
    y = ya.loc[ids].values.astype(float)
    assert (y == yb.loc[ids].values).all()
    pa, pb = pa.loc[ids], pb.loc[ids]
    point = np.ones((1, len(ids)))
    W = boot_counts(y, rng)
    ma, mb = seed_mean(y, pa, point), seed_mean(y, pb, point)
    ba, bb = seed_mean(y, pa, W), seed_mean(y, pb, W)
    out = {"n": len(ids), "n_pos": int(y.sum())}
    for k in METRICS:
        lo, hi = np.percentile(ba[k] - bb[k], [2.5, 97.5])
        out.update({f"{k}_gnn": ma[k][0], f"{k}_base": mb[k][0], f"delta_{k}": ma[k][0] - mb[k][0],
                    f"{k}_ci_low": lo, f"{k}_ci_high": hi})
    return out


def main():
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for variant, cohort in CONFIGS:
        for qg, qb in WINDOW_PAIRS:
            for g in GNNS:
                for b in BASELINES:
                    if b == "dummy":
                        continue
                    r = compare(load(g, qg, variant, cohort), load(b, qb, variant, cohort), rng)
                    rows.append({"variant": variant, "cohort": cohort, "window_gnn": qg, "window_base": qb,
                                 "gnn": g, "baseline": b, **r})
    pd.DataFrame(rows).to_csv(OUT / "H_threshold_metrics_comparisons.csv", index=False)
    lv = []
    for variant, cohort in CONFIGS:
        for q in ["q1", "q2", "q3"]:
            for m in GNNS + BASELINES:
                y, _, p = load(m, q, variant, cohort)
                yy = y.values.astype(float)
                per = [metrics(yy, p[s].values.astype(float), np.ones((1, len(yy)))) for s in range(5)]
                row = {"variant": variant, "cohort": cohort, "q": q, "model": m, "n": len(y), "n_pos": int(yy.sum())}
                for k in METRICS:
                    vals = [d[k][0] for d in per]
                    row[k], row[f"{k}_sd"] = np.mean(vals), np.std(vals, ddof=1)
                lv.append(row)
    pd.DataFrame(lv).to_csv(OUT / "H_threshold_metrics_levels.csv", index=False)


if __name__ == "__main__":
    main()
