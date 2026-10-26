"""Final comparison of the GNN-improvement attempts against the official baselines (Test).

Model groups (chronological split, original cohort, no_mask, node_graph view):
  official GNNs      gcn, graphsage, gat                       trainer/results/gnn (final_*)
  direction-aware    dirsage, dirgcn                          trainer/results/gnn_improved (final_*)
  stacked / bagged   graphsage_bagged, gradient_boosting(+graphsage_stack), xgboost(+graphsage_stack)
                                                              trainer/results/gnn_stacked
  official baselines 8 non-trivial baselines                 trainer/results (final_*)

Every model is scored identically: per seed, the F1-max threshold is chosen on its validation
scores and applied to Test; ROC-AUC/AP are reported per seed (mean ± SD) and for the seed
ensemble (mean score). Paired contrasts use seed-ensemble scores on identical Test tokens with a
label-stratified bootstrap. The reference "best baseline" of each cell is chosen by validation
AP among the official baselines, never by Test. Holm correction across all contrasts.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

BASE = Path(__file__).resolve().parent
RES = BASE.parent / 'trainer' / 'results'
OUT = BASE / 'output' / 'test' / 'gnn_improvement'
SUFFIX = 'gnn_graph_structure_timing_node_graph_no_mask_original'
OFFICIAL_GNN = ['gcn', 'graphsage', 'gat']
DIR_GNN = ['dirsage', 'dirgcn']
STACKED = ['graphsage_bagged', 'gradient_boosting+graphsage_stack', 'xgboost+graphsage_stack']
BASELINES = ['logistic_regression', 'decision_tree', 'random_forest', 'gradient_boosting', 'xgboost', 'mlp',
             'ft_transformer', 'autoint']
N_BOOT, RNG = 2000, 2026


def source(model, scope, q):
    if model in OFFICIAL_GNN:
        return RES / 'gnn' / f'final_predictions_{model}_{scope}_{q}_{SUFFIX}.csv', 'official GNN (run_gnn.py)'
    if model in DIR_GNN:
        return RES / 'gnn_improved' / f'final_predictions_{model}_{scope}_{q}_{SUFFIX}.csv', 'direction-aware GNN (new)'
    if model in STACKED:
        return RES / 'gnn_stacked' / f'predictions_{model}_{scope}_{q}.csv', 'GNN bagging / stacking (new)'
    return RES / f'final_predictions_{model}_{scope}_{q}_node_graph_no_mask_original.csv', 'official baseline'


def select_threshold(y, p):
    cand = np.unique(np.round(p, 4))
    return 0.5 if len(cand) < 2 else max((f1_score(y, (p >= t).astype(int), zero_division=0), t) for t in cand)[1]


def load(model, scope, q):
    f, group = source(model, scope, q)
    if not f.is_file():
        return None
    d = pd.read_csv(f)
    w = {p: d[d.partition == p].pivot(index='token_mint', columns='seed', values='y_score') for p in ['val', 'test']}
    y = {p: d[d.partition == p].groupby('token_mint').y_true.first() for p in ['val', 'test']}
    return group, w, y


def pboot(y, a, b, rng):
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    da, dr = np.empty(N_BOOT), np.empty(N_BOOT)
    for r in range(N_BOOT):
        i = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        da[r] = average_precision_score(y[i], a[i]) - average_precision_score(y[i], b[i])
        dr[r] = roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i])
    p = lambda d: min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    return da, p(da), dr, p(dr)


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[o])
    out = np.empty_like(p)
    out[o] = np.minimum(adj, 1)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, tests = [], []
    rng = np.random.default_rng(RNG)
    for scope in ['panel', 'full']:
        for q in ['q1', 'q2', 'q3']:
            loaded = {m: load(m, scope, q) for m in OFFICIAL_GNN + DIR_GNN + STACKED + BASELINES}
            loaded = {m: v for m, v in loaded.items() if v is not None}
            ids = sorted(set.intersection(*(set(v[1]['test'].index) for v in loaded.values())))
            vids = sorted(set.intersection(*(set(v[1]['val'].index) for v in loaded.values())))
            ens = {}
            for m, (group, w, y) in loaded.items():
                yv, yt = y['val'].loc[vids].values, y['test'].loc[ids].values
                sv, st = w['val'].loc[vids], w['test'].loc[ids]
                per = []
                for s in st.columns:
                    thr = select_threshold(yv, sv[s].values)
                    yp = (st[s].values >= thr).astype(int)
                    per.append({'roc': roc_auc_score(yt, st[s]), 'ap': average_precision_score(yt, st[s]),
                                'f1': f1_score(yt, yp, zero_division=0), 'prec': precision_score(yt, yp, zero_division=0),
                                'rec': recall_score(yt, yp, zero_division=0)})
                per = pd.DataFrame(per)
                ev, et = sv.mean(1).values, st.mean(1).values
                ens[m] = et
                thr = select_threshold(yv, ev)
                ype = (et >= thr).astype(int)
                rows.append({'scope': scope, 'window': q, 'group': group, 'model': m, 'n_test': len(ids), 'rugs': int(yt.sum()),
                             'val AP (ens)': average_precision_score(yv, ev),
                             'test ROC-AUC': f"{per.roc.mean():.3f} ± {per.roc.std():.3f}",
                             'test AP': f"{per.ap.mean():.3f} ± {per.ap.std():.3f}",
                             'test F1': f"{per.f1.mean():.3f} ± {per.f1.std():.3f}",
                             'test precision': f"{per.prec.mean():.3f} ± {per.prec.std():.3f}",
                             'test recall': f"{per.rec.mean():.3f} ± {per.rec.std():.3f}",
                             'test ROC-AUC (ens)': roc_auc_score(yt, et), 'test AP (ens)': average_precision_score(yt, et),
                             'test F1 (ens)': f1_score(yt, ype, zero_division=0),
                             'test precision (ens)': precision_score(yt, ype, zero_division=0),
                             'test recall (ens)': recall_score(yt, ype, zero_division=0)})
                y_test = yt
            cell = pd.DataFrame([r for r in rows if r['scope'] == scope and r['window'] == q])
            best = cell[cell.group == 'official baseline'].sort_values('val AP (ens)').model.iloc[-1]
            for m in [x for x in OFFICIAL_GNN + DIR_GNN + STACKED if x in ens]:
                refs = [('xgboost', 'XGBoost'), (best, f'best baseline by val AP ({best})')]
                if m.endswith('_stack'):
                    refs.append((m.split('+')[0], 'same booster without GNN feature'))
                for ref, tag in refs:
                    if ref == m:
                        continue
                    da, pa, dr, pr = pboot(y_test, ens[m], ens[ref], rng)
                    tests.append({'scope': scope, 'window': q, 'model': m, 'reference': tag,
                                  'Δ AP': average_precision_score(y_test, ens[m]) - average_precision_score(y_test, ens[ref]),
                                  'Δ AP 95% CI': f'[{np.percentile(da, 2.5):+.3f}, {np.percentile(da, 97.5):+.3f}]', 'p (AP)': pa,
                                  'Δ ROC-AUC': roc_auc_score(y_test, ens[m]) - roc_auc_score(y_test, ens[ref]),
                                  'Δ ROC 95% CI': f'[{np.percentile(dr, 2.5):+.3f}, {np.percentile(dr, 97.5):+.3f}]', 'p (ROC)': pr})
            print(f'{scope} {q} done', flush=True)
    res, tst = pd.DataFrame(rows), pd.DataFrame(tests)
    tst['p (AP) Holm'] = holm(tst['p (AP)'])
    tst['p (ROC) Holm'] = holm(tst['p (ROC)'])
    res.to_csv(OUT / 'gnn_improvement_results.csv', index=False)
    tst.to_csv(OUT / 'gnn_improvement_paired_tests.csv', index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)
    print(res.pivot_table(index=['group', 'model'], columns=['scope', 'window'], values='test AP (ens)', sort=False).round(3))
    print(res.pivot_table(index=['group', 'model'], columns=['scope', 'window'], values='test ROC-AUC (ens)', sort=False).round(3))
    print(tst.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
