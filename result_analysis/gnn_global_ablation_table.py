"""Comparative table for the q1 global-transaction-graph ablation (trainer/run_gnn_global_ablation.py).

Rows: GCN and GraphSAGE on (a) the per-token transfer graph of run_gnn.py (stored final run),
(b) the same per-token graph re-run by the ablation runner with identical settings (paired
reference), (c) one global time-respecting transaction graph; plus matched node_graph
baselines. Columns: train / validation / test ROC-AUC, PR-AUC, F1, precision, recall,
mean ± SD over the five seeds (chronological split, original cohort, no_mask).

Paired tests, global vs per-token re-run, on identical Test tokens:
  * Delta AP and Delta ROC-AUC of seed-mean scores, label-stratified token bootstrap (10,000);
  * exact McNemar on seed-majority predictions at each seed's validation threshold.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

BASE = Path(__file__).resolve().parent
TRAINER = BASE.parent / 'trainer' / 'results'
ABL = TRAINER / 'gnn_global_ablation'
OUT = BASE / 'output' / 'test' / 'gnn_global_ablation'
SCOPES, GNNS = ['panel', 'full'], ['gcn', 'graphsage']
BASELINES = ['logistic_regression', 'random_forest', 'gradient_boosting', 'xgboost', 'mlp']
METRICS = ['roc_auc', 'pr_auc', 'f1', 'precision', 'recall']
TAG = 'q1_gnn_graph_structure_timing_node_graph_no_mask_original'
N_BOOT, RNG = 10000, 2026


def metrics_from_predictions(pred, partition):
    rows = []
    for s, g in pred[pred.partition == partition].groupby('seed'):
        y, sc, yp = g.y_true.values, g.y_score.values, g.y_pred.values
        rows.append({'seed': s, 'partition': partition, 'roc_auc': roc_auc_score(y, sc),
                     'pr_auc': average_precision_score(y, sc), 'f1': f1_score(y, yp, zero_division=0),
                     'precision': precision_score(y, yp, zero_division=0), 'recall': recall_score(y, yp, zero_division=0)})
    return pd.DataFrame(rows)


def gnn_runs(folder, model, scope):
    f = folder / f'final_runs_{model}_{scope}_{TAG}.csv'
    return pd.read_csv(f) if f.is_file() else None


def gnn_preds(folder, model, scope):
    f = folder / f'final_predictions_{model}_{scope}_{TAG}.csv'
    return pd.read_csv(f) if f.is_file() else None


def baseline_runs(model, scope):
    runs = BASE / 'input' / f'baseline_{scope}' / f'runs_{model}_{scope}_q1_node_graph_no_mask_original.csv'
    pred = TRAINER / f'final_predictions_{model}_{scope}_q1_node_graph_no_mask_original.csv'
    if not (runs.is_file() and pred.is_file()):
        return None, None
    r = pd.read_csv(runs)
    p = pd.read_csv(pred)
    return pd.concat([r[r.partition.isin(['train', 'val'])], metrics_from_predictions(p, 'test')]), p


def summarise(runs, label):
    row = dict(label)
    for part in ['train', 'val', 'test']:
        g = runs[runs.partition == part]
        for m in METRICS:
            row[f'{part} {m}'] = f'{g[m].mean():.3f} ± {g[m].std():.3f}' if len(g) else '—'
    return row


def wide(pred):
    t = pred[pred.partition == 'test']
    return (t.pivot(index='token_mint', columns='seed', values='y_score'),
            t.pivot(index='token_mint', columns='seed', values='y_pred'),
            t.groupby('token_mint').y_true.first())


def paired(pa, pb):
    sa, ya_pred, y = wide(pa)
    sb, yb_pred, yb = wide(pb)
    ids = sa.index.intersection(sb.index)
    y = y.loc[ids].values
    assert (y == yb.loc[ids].values).all()
    ma, mb = sa.loc[ids].mean(1).values, sb.loc[ids].mean(1).values
    rng = np.random.default_rng(RNG)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    d_ap, d_roc = [], []
    for _ in range(N_BOOT):
        i = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        d_ap.append(average_precision_score(y[i], ma[i]) - average_precision_score(y[i], mb[i]))
        d_roc.append(roc_auc_score(y[i], ma[i]) - roc_auc_score(y[i], mb[i]))
    ca = (ya_pred.loc[ids].sum(1).values >= 3) == y
    cb = (yb_pred.loc[ids].sum(1).values >= 3) == y
    b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p_mc = 1.0 if b + c == 0 else min(1.0, 2 * binom.cdf(min(b, c), b + c, 0.5))
    p_ap = min(1.0, 2 * min((np.array(d_ap) <= 0).mean(), (np.array(d_ap) >= 0).mean()))
    p_roc = min(1.0, 2 * min((np.array(d_roc) <= 0).mean(), (np.array(d_roc) >= 0).mean()))
    return {'n test tokens': len(ids), 'rugs': int(y.sum()),
            'Δ AP (global − per-token)': average_precision_score(y, ma) - average_precision_score(y, mb),
            'Δ AP 95% CI': f'[{np.percentile(d_ap, 2.5):+.3f}, {np.percentile(d_ap, 97.5):+.3f}]', 'p (AP)': p_ap,
            'Δ ROC-AUC': roc_auc_score(y, ma) - roc_auc_score(y, mb),
            'Δ ROC-AUC 95% CI': f'[{np.percentile(d_roc, 2.5):+.3f}, {np.percentile(d_roc, 97.5):+.3f}]', 'p (ROC)': p_roc,
            'McNemar b/c (global right / per-token right)': f'{b}/{c}', 'McNemar p': p_mc}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    table, tests = [], []
    for scope in SCOPES:
        for m in GNNS:
            for label, folder in [('per-token graph (run_gnn.py, stored final)', TRAINER / 'gnn'),
                                  ('per-token graph (ablation re-run)', ABL / 'per_token'),
                                  ('global transaction graph (ablation)', ABL / 'global')]:
                r = gnn_runs(folder, m, scope)
                if r is not None:
                    table.append(summarise(r, {'scope': scope, 'model': m, 'graph': label}))
            pa, pb = gnn_preds(ABL / 'global', m, scope), gnn_preds(ABL / 'per_token', m, scope)
            if pa is not None and pb is not None:
                tests.append({'scope': scope, 'model': m, **paired(pa, pb)})
        for b in BASELINES:
            r, _ = baseline_runs(b, scope)
            if r is not None:
                table.append(summarise(r, {'scope': scope, 'model': b, 'graph': 'no graph (node_graph features)'}))
    table, tests = pd.DataFrame(table), pd.DataFrame(tests)
    table.to_csv(OUT / 'global_vs_per_token_table.csv', index=False)
    tests.to_csv(OUT / 'global_vs_per_token_paired_tests.csv', index=False)
    print(table.to_string(index=False))
    print(tests.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
