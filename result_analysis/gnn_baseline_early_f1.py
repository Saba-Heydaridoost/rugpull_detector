"""Exploratory GNN-vs-baseline early-window comparison on AP and F1 (Test, Panel, node_graph view only).

For every GNN window <= baseline window (q1-q1, q1-q2, q1-q3, q2-q2, q2-q3, q3-q3) computes seed-mean
AP and F1 deltas (GNN minus baseline) on matched token IDs, with paired label-stratified token
bootstrap CIs (one shared draw across both arms and all five seeds). F1 uses the saved y_pred, i.e.
each model's Validation-tuned threshold applied unchanged to Test. CIs are unadjusted 95% intervals.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from gnn_baseline_ablation_tables import (BASELINE_DIR, BASELINES, CONFIGS, GNN_DIR, GNNS, OUT, RNG_SEED,
                                          average_precision_score, boot_counts, weighted_ap)

WINDOW_PAIRS = [("q1", "q1"), ("q1", "q2"), ("q1", "q3"), ("q2", "q2"), ("q2", "q3"), ("q3", "q3")]
_cache = {}


def load(model, q, variant, cohort):
    """Test predictions as (y_true Series, score DataFrame, pred DataFrame), indexed by token, columns = seeds."""
    key = (model, q, variant, cohort)
    if key not in _cache:
        if model in GNNS:
            path = GNN_DIR / f"final_predictions_{model}_panel_{q}_gnn_graph_structure_timing_node_graph_{variant}_{cohort}.csv"
        else:
            path = BASELINE_DIR / f"final_predictions_{model}_panel_{q}_node_graph_{variant}_{cohort}.csv"
        d = pd.read_csv(path)
        d = d[d.partition == "test"]
        score = d.pivot(index="token_mint", columns="seed", values="y_score")
        pred = d.pivot(index="token_mint", columns="seed", values="y_pred")
        y = d.groupby("token_mint").y_true.first()
        assert score.shape[1] == 5 and not score.isna().any().any(), path
        _cache[key] = (y, score, pred)
    return _cache[key]


def weighted_f1(y, pred, W):
    tp = W @ (y * pred)
    fp = W @ ((1 - y) * pred)
    fn = W @ (y * (1 - pred))
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1e-12)


def compare(a, b, rng):
    (ya, sa, pa), (yb, sb, pb) = a, b
    ids = sorted(set(ya.index) & set(yb.index))
    y = ya.loc[ids].values.astype(float)
    assert (y == yb.loc[ids].values).all()
    sa, pa, sb, pb = (x.loc[ids] for x in (sa, pa, sb, pb))
    W = boot_counts(y, rng)
    out = dict(n=len(ids), n_pos=int(y.sum()))
    for metric, point, boot, arms in [
        ("ap", average_precision_score, weighted_ap, (sa, sb)),
        ("f1", f1_score, weighted_f1, (pa, pb)),
    ]:
        vals = [np.mean([point(y, arm[s].values) for s in range(5)]) for arm in arms]
        bs = [np.mean([boot(y, arm[s].values.astype(float), W) for s in range(5)], axis=0) for arm in arms]
        lo, hi = np.percentile(bs[0] - bs[1], [2.5, 97.5])
        out.update({f"{metric}_gnn": vals[0], f"{metric}_base": vals[1], f"delta_{metric}": vals[0] - vals[1],
                    f"{metric}_ci_low": lo, f"{metric}_ci_high": hi})
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
    pd.DataFrame(rows).to_csv(OUT / "G_early_window_ap_f1.csv", index=False)
    # Per-model level table (seed-mean AP/F1 on each config's full Test token set)
    lv = []
    for variant, cohort in CONFIGS:
        for q in ["q1", "q2", "q3"]:
            for m in GNNS + BASELINES:
                y, s, p = load(m, q, variant, cohort)
                lv.append({"variant": variant, "cohort": cohort, "q": q, "model": m, "n": len(y),
                           "ap": np.mean([average_precision_score(y, s[k]) for k in range(5)]),
                           "f1": np.mean([f1_score(y, p[k]) for k in range(5)]),
                           "f1_sd": np.std([f1_score(y, p[k]) for k in range(5)], ddof=1)})
    pd.DataFrame(lv).to_csv(OUT / "G_levels_ap_f1.csv", index=False)


if __name__ == "__main__":
    main()
