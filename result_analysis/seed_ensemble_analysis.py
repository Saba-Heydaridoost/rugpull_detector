"""Seed ensembling, applied identically to GNNs and baselines (Test partition).

The reported results average *metrics* over five seeds. A seed ensemble instead averages
the five seeds' *scores* per token and evaluates once, a standard variance-reduction step
that every model can use. Thresholds for F1/precision/recall are chosen on the ensemble's
validation scores (F1-max, as in the trainer); the reference baseline per cell is the one
with the best *validation* AP. Test is never used for any choice.

Inputs: final predictions of run_gnn.py and run_baselines.py (trainer/results, chronological).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

BASE = Path(__file__).resolve().parent
RESULTS = BASE.parent / 'trainer' / 'results'
OUT = BASE / 'output' / 'test' / 'seed_ensemble'
GNNS = ['gcn', 'graphsage', 'gat']
BASELINES = ['logistic_regression', 'decision_tree', 'random_forest', 'gradient_boosting', 'xgboost', 'mlp',
             'ft_transformer', 'autoint']
CONFIG = ('no_mask', 'original')
N_BOOT, RNG = 10000, 2026


def path(model, scope, q):
    v, c = CONFIG
    if model in GNNS:
        return RESULTS / 'gnn' / f'final_predictions_{model}_{scope}_{q}_gnn_graph_structure_timing_node_graph_{v}_{c}.csv'
    return RESULTS / f'final_predictions_{model}_{scope}_{q}_node_graph_{v}_{c}.csv'


def load(model, scope, q):
    f = path(model, scope, q)
    if not f.is_file():
        return None
    d = pd.read_csv(f)
    out = {}
    for part in ['val', 'test']:
        g = d[d.partition == part]
        out[part] = (g.pivot(index='token_mint', columns='seed', values='y_score'),
                     g.groupby('token_mint').y_true.first())
    return out


def select_threshold(y, p):
    cand = np.unique(np.round(p, 4))
    return 0.5 if len(cand) < 2 else max((f1_score(y, (p >= t).astype(int), zero_division=0), t) for t in cand)[1]


def boot_delta(y, a, b, rng):
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    d_ap, d_roc = [], []
    for _ in range(N_BOOT):
        i = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        d_ap.append(average_precision_score(y[i], a[i]) - average_precision_score(y[i], b[i]))
        d_roc.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    d_ap, d_roc = np.array(d_ap), np.array(d_roc)
    p = lambda d: min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    return np.percentile(d_ap, [2.5, 97.5]), p(d_ap), np.percentile(d_roc, [2.5, 97.5]), p(d_roc)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, tests = [], []
    for scope in ['panel', 'full']:
        for q in ['q1', 'q2', 'q3']:
            data = {m: load(m, scope, q) for m in GNNS + BASELINES}
            data = {m: d for m, d in data.items() if d is not None}
            ids = sorted(set.intersection(*(set(d['test'][0].index) for d in data.values())))
            vids = sorted(set.intersection(*(set(d['val'][0].index) for d in data.values())))
            ens = {}
            for m, d in data.items():
                sv, yv = d['val'][0].loc[vids], d['val'][1].loc[vids].values
                st, yt = d['test'][0].loc[ids], d['test'][1].loc[ids].values
                ev, et = sv.mean(1).values, st.mean(1).values
                thr = select_threshold(yv, ev)
                yp = (et >= thr).astype(int)
                ens[m] = (ev, et)
                rows.append({'scope': scope, 'window': q, 'model': m, 'family': 'GNN' if m in GNNS else 'baseline',
                             'n_test': len(ids), 'rugs': int(yt.sum()),
                             'val AP (ensemble)': average_precision_score(yv, ev),
                             'test ROC-AUC (seed-mean metric)': np.mean([roc_auc_score(yt, st[s]) for s in st]),
                             'test AP (seed-mean metric)': np.mean([average_precision_score(yt, st[s]) for s in st]),
                             'test ROC-AUC (ensemble)': roc_auc_score(yt, et),
                             'test AP (ensemble)': average_precision_score(yt, et),
                             'test F1 (ensemble)': f1_score(yt, yp, zero_division=0),
                             'test precision (ensemble)': precision_score(yt, yp, zero_division=0),
                             'test recall (ensemble)': recall_score(yt, yp, zero_division=0)})
                y_test = yt
            cell = pd.DataFrame([r for r in rows if r['scope'] == scope and r['window'] == q])
            best_base = cell[cell.family == 'baseline'].sort_values('val AP (ensemble)').model.iloc[-1]
            rng = np.random.default_rng(RNG)
            for g in [m for m in GNNS if m in ens]:
                for ref, tag in [('xgboost', 'XGBoost (frozen comparator)'), (best_base, f'best-by-validation baseline ({best_base})')]:
                    ci_ap, p_ap, ci_roc, p_roc = boot_delta(y_test, ens[g][1], ens[ref][1], rng)
                    tests.append({'scope': scope, 'window': q, 'GNN': g, 'reference': tag,
                                  'Δ AP': average_precision_score(y_test, ens[g][1]) - average_precision_score(y_test, ens[ref][1]),
                                  'Δ AP 95% CI': f'[{ci_ap[0]:+.3f}, {ci_ap[1]:+.3f}]', 'p (AP)': p_ap,
                                  'Δ ROC-AUC': roc_auc_score(y_test, ens[g][1]) - roc_auc_score(y_test, ens[ref][1]),
                                  'Δ ROC-AUC 95% CI': f'[{ci_roc[0]:+.3f}, {ci_roc[1]:+.3f}]', 'p (ROC)': p_roc})
    res, tst = pd.DataFrame(rows), pd.DataFrame(tests)
    res['ensemble gain AP'] = res['test AP (ensemble)'] - res['test AP (seed-mean metric)']
    res['ensemble gain ROC'] = res['test ROC-AUC (ensemble)'] - res['test ROC-AUC (seed-mean metric)']
    res.to_csv(OUT / 'seed_ensemble_results.csv', index=False)
    tst.to_csv(OUT / 'seed_ensemble_gnn_vs_baselines.csv', index=False)
    pd.set_option('display.width', 250)
    print(res.groupby('family')[['ensemble gain AP', 'ensemble gain ROC']].describe().round(3))
    print(res.pivot_table(index=['scope', 'model'], columns='window', values='test AP (ensemble)', sort=False).round(3))
    print(tst.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
