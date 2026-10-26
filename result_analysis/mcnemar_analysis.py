"""Exact McNemar tests between GNNs and baselines on the held-out Test partition.

Inputs are the final predictions written by run_gnn.py and run_baselines.py
(trainer/results, chronological split). Each model's binary decision for a token is the
majority vote of its five seeds, each thresholded at its own validation-F1 threshold
(the `y_pred` column). Pairs are evaluated on the intersection of their Test tokens,
which must share labels.

For every pair (A, B):
  b = tokens A classifies correctly and B incorrectly
  c = tokens B classifies correctly and A incorrectly
  exact two-sided McNemar p = min(1, 2 * BinomCDF(min(b, c); b + c, 0.5))
Per-seed tests (seed s of A against seed s of B) are reported as robustness.
p-values are Holm- and BH-adjusted within each configuration (scope x variant x cohort).

Exploratory: complements, and does not replace, the frozen AP analysis in
gnn_primary_inference.py.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom

BASE = Path(__file__).resolve().parent
RESULTS = BASE.parent / 'trainer' / 'results'
OUT = BASE / 'output' / 'test' / 'mcnemar'

GNNS = ['gcn', 'graphsage', 'gat']
BASELINES = ['dummy', 'logistic_regression', 'decision_tree', 'random_forest', 'gradient_boosting',
             'xgboost', 'mlp', 'ft_transformer', 'autoint']
WINDOWS = ['q1', 'q2', 'q3']
CONFIGS = [  # (scope, variant, cohort, role)
    ('panel', 'no_mask', 'original', 'primary'),
    ('full', 'no_mask', 'original', 'secondary'),
    ('panel', 'with_mask', 'original', 'secondary'),
    ('full', 'with_mask', 'original', 'secondary'),
]
CROSS_WINDOW = [('q1', 'q2'), ('q1', 'q3')]  # predeclared: q1 GNN vs later-window XGBoost


def path_for(model, scope, q, variant, cohort):
    if model in GNNS:
        return RESULTS / 'gnn' / (f'final_predictions_{model}_{scope}_{q}_gnn_graph_structure_timing_'
                                  f'node_graph_{variant}_{cohort}.csv')
    return RESULTS / f'final_predictions_{model}_{scope}_{q}_node_graph_{variant}_{cohort}.csv'


def load(model, scope, q, variant, cohort):
    """Test predictions as a token x seed matrix of y_pred, plus y_true."""
    path = path_for(model, scope, q, variant, cohort)
    if not path.is_file():
        return None
    d = pd.read_csv(path)
    d = d[d.partition == 'test']
    pred = d.pivot(index='token_mint', columns='seed', values='y_pred')
    y = d.groupby('token_mint').y_true.agg(['min', 'max'])
    assert (y['min'] == y['max']).all(), path
    if sorted(pred.columns) != [0, 1, 2, 3, 4] or pred.isna().any().any():
        print(f'incomplete seeds, skipped: {path.name}')
        return None
    return pred.astype(int), y['min'].astype(int)


def exact_mcnemar(correct_a, correct_b):
    b = int((correct_a & ~correct_b).sum())
    c = int((~correct_a & correct_b).sum())
    p = 1.0 if b + c == 0 else min(1.0, 2 * binom.cdf(min(b, c), b + c, 0.5))
    return b, c, p


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    out = np.empty_like(p)
    out[order] = np.minimum(adj, 1)
    return out


def bh(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    out = np.empty_like(p)
    out[order] = np.minimum(adj, 1)
    return out


def compare(a, b, label):
    (pa, ya), (pb, yb) = a, b
    ids = pa.index.intersection(pb.index)
    assert (ya.loc[ids].values == yb.loc[ids].values).all()
    y = ya.loc[ids].values
    pa, pb = pa.loc[ids], pb.loc[ids]
    vote_a = (pa.sum(1) >= 3).values
    vote_b = (pb.sum(1) >= 3).values
    ca, cb = vote_a == y, vote_b == y
    b_, c_, p = exact_mcnemar(ca, cb)
    seed_p, seed_dir = [], []
    for s in range(5):
        sb, sc, sp = exact_mcnemar(pa[s].values == y, pb[s].values == y)
        seed_p.append(sp)
        seed_dir.append(np.sign(sb - sc))
    seed_p, seed_dir = np.array(seed_p), np.array(seed_dir)
    return {**label, 'n_tokens': len(ids), 'n_rug': int(y.sum()),
            'acc_a': ca.mean(), 'acc_b': cb.mean(),
            'b (A right, B wrong)': b_, 'c (A wrong, B right)': c_,
            'p_exact': p,
            'seeds with p<0.05, A better': int(((seed_p < 0.05) & (seed_dir > 0)).sum()),
            'seeds with p<0.05, B better': int(((seed_p < 0.05) & (seed_dir < 0)).sum()),
            'median per-seed p': float(np.median(seed_p))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for scope, variant, cohort, role in CONFIGS:
        cfg = dict(role=role, scope=scope, variant=variant, cohort=cohort)
        preds = {(m, q): load(m, scope, q, variant, cohort) for m in GNNS + BASELINES for q in WINDOWS}
        block = []
        for q in WINDOWS:
            for g in GNNS:
                for other in BASELINES + [x for x in GNNS if GNNS.index(x) > GNNS.index(g)]:
                    if preds[(g, q)] is None or preds[(other, q)] is None:
                        continue
                    block.append(compare(preds[(g, q)], preds[(other, q)],
                                         {**cfg, 'contrast': 'same window', 'window_a': q, 'window_b': q,
                                          'model_a': g, 'model_b': other}))
        if scope == 'panel':
            for g in GNNS:
                for qa, qb in CROSS_WINDOW:
                    if preds[(g, qa)] is None or preds[('xgboost', qb)] is None:
                        continue
                    block.append(compare(preds[(g, qa)], preds[('xgboost', qb)],
                                         {**cfg, 'contrast': 'cross window (predeclared)', 'window_a': qa,
                                          'window_b': qb, 'model_a': g, 'model_b': 'xgboost'}))
        t = pd.DataFrame(block)
        t['p_holm'] = holm(t['p_exact'])
        t['p_bh'] = bh(t['p_exact'])
        rows.append(t)
    res = pd.concat(rows, ignore_index=True)
    res['better (majority vote)'] = np.where(res['b (A right, B wrong)'] > res['c (A wrong, B right)'], 'A',
                                             np.where(res['b (A right, B wrong)'] < res['c (A wrong, B right)'], 'B', '='))
    res.to_csv(OUT / 'mcnemar_results.csv', index=False)
    print(f'saved {OUT / "mcnemar_results.csv"} ({len(res)} tests)')
    return res


if __name__ == '__main__':
    main()
