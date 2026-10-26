"""HNN4RP (adapted) vs the project's official GNNs and baselines (chronological split).

Table: train / validation / test ROC-AUC, PR-AUC, F1, precision, recall (mean ± SD over seeds),
for HNN4RP and its ablations, the official GCN / GraphSAGE / GAT (run_gnn.py final runs) and
the official node_graph baselines, per scope (panel, full) and window (q1-q3); original
cohort, no_mask.

Paired tests on identical Test tokens, HNN4RP (full model) vs XGBoost, vs the baseline with
the best validation AP, and vs the official GraphSAGE: Delta AP / Delta ROC-AUC of seed-mean
scores with a label-stratified bootstrap, and exact McNemar on seed-majority predictions.
Holm correction across all contrasts.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

BASE = Path(__file__).resolve().parent
SPLIT = sys.argv[1] if len(sys.argv) > 1 else 'chronological'          # chronological | stratified
RES = BASE.parent / 'trainer' / ('results' if SPLIT == 'chronological' else 'results_stratified')
OUT = BASE / 'output' / 'test' / ('hnn4rp' if SPLIT == 'chronological' else 'hnn4rp_stratified')
GNN_SUFFIX = 'gnn_graph_structure_timing_node_graph_no_mask_original'
HNN = ['hnn4rp', 'hnn4rp-no_token_nbrs', 'hnn4rp-no_wallet_nbrs', 'hnn4rp-no_lstm']
GNNS = ['gcn', 'graphsage', 'gat']
BASELINES = ['logistic_regression', 'decision_tree', 'random_forest', 'gradient_boosting', 'xgboost', 'mlp',
             'ft_transformer', 'autoint']
METRICS = ['roc_auc', 'pr_auc', 'f1', 'precision', 'recall']
N_BOOT, RNG = 5000, 2026


def test_metrics(pred):
    rows = []
    for s, g in pred[pred.partition == 'test'].groupby('seed'):
        y, sc, yp = g.y_true.values, g.y_score.values, g.y_pred.values
        rows.append({'seed': s, 'partition': 'test', 'roc_auc': roc_auc_score(y, sc),
                     'pr_auc': average_precision_score(y, sc), 'f1': f1_score(y, yp, zero_division=0),
                     'precision': precision_score(y, yp, zero_division=0), 'recall': recall_score(y, yp, zero_division=0)})
    return pd.DataFrame(rows)


def load(model, scope, q):
    """(runs with train/val/test rows, final predictions)."""
    if model in HNN:
        tag = f'{model}_{scope}_{q}_node_graph_no_mask_original'
        r, p = RES / 'hnn4rp' / f'runs_{tag}.csv', RES / 'hnn4rp' / f'final_predictions_{tag}.csv'
        return (pd.read_csv(r), pd.read_csv(p)) if r.is_file() and p.is_file() else (None, None)
    if model in GNNS:
        if SPLIT == 'stratified':            # stratified GNN runs score test directly (no final_ prefix)
            r = RES / 'gnn_selection_ap' / f'runs_{model}_{scope}_{q}_{GNN_SUFFIX}.csv'
            p = RES / 'gnn_selection_ap' / f'predictions_{model}_{scope}_{q}_{GNN_SUFFIX}.csv'
        else:
            r = RES / 'gnn' / f'final_runs_{model}_{scope}_{q}_{GNN_SUFFIX}.csv'
            p = RES / 'gnn' / f'final_predictions_{model}_{scope}_{q}_{GNN_SUFFIX}.csv'
        return (pd.read_csv(r), pd.read_csv(p)) if r.is_file() and p.is_file() else (None, None)
    if SPLIT == 'stratified':                # stratified baseline runs carry train/val/test rows
        r = RES / f'runs_{model}_{scope}_{q}_node_graph_no_mask_original.csv'
        p = RES / f'predictions_{model}_{scope}_{q}_node_graph_no_mask_original.csv'
        return (pd.read_csv(r), pd.read_csv(p)) if r.is_file() and p.is_file() else (None, None)
    r = BASE / 'input' / f'baseline_{scope}' / f'runs_{model}_{scope}_{q}_node_graph_no_mask_original.csv'
    p = RES / f'final_predictions_{model}_{scope}_{q}_node_graph_no_mask_original.csv'
    if not (r.is_file() and p.is_file()):
        return None, None
    r, p = pd.read_csv(r), pd.read_csv(p)
    return pd.concat([r[r.partition.isin(['train', 'val'])], test_metrics(p)]), p


def wide(pred):
    t = pred[pred.partition == 'test']
    return (t.pivot(index='token_mint', columns='seed', values='y_score'),
            t.pivot(index='token_mint', columns='seed', values='y_pred'), t.groupby('token_mint').y_true.first())


def paired(pa, pb, rng):
    sa, va, y = wide(pa)
    sb, vb, yb = wide(pb)
    ids = sa.index.intersection(sb.index)
    y = y.loc[ids].values
    assert (y == yb.loc[ids].values).all()
    ma, mb = sa.loc[ids].mean(1).values, sb.loc[ids].mean(1).values
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    da, dr = np.empty(N_BOOT), np.empty(N_BOOT)
    for r in range(N_BOOT):
        i = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        da[r] = average_precision_score(y[i], ma[i]) - average_precision_score(y[i], mb[i])
        dr[r] = roc_auc_score(y[i], ma[i]) - roc_auc_score(y[i], mb[i])
    ca, cb = (va.loc[ids].sum(1).values >= 3) == y, (vb.loc[ids].sum(1).values >= 3) == y
    b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p2 = lambda d: min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    return {'n test': len(ids), 'rugs': int(y.sum()),
            'Δ AP (seed-mean score)': average_precision_score(y, ma) - average_precision_score(y, mb),
            'Δ AP 95% CI': f'[{np.percentile(da, 2.5):+.3f}, {np.percentile(da, 97.5):+.3f}]', 'p (AP)': p2(da),
            'Δ ROC-AUC': roc_auc_score(y, ma) - roc_auc_score(y, mb),
            'Δ ROC 95% CI': f'[{np.percentile(dr, 2.5):+.3f}, {np.percentile(dr, 97.5):+.3f}]', 'p (ROC)': p2(dr),
            'McNemar b/c': f'{b}/{c}', 'McNemar p': 1.0 if b + c == 0 else min(1.0, 2 * binom.cdf(min(b, c), b + c, 0.5))}


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[o])
    out = np.empty_like(p)
    out[o] = np.minimum(adj, 1)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    table, tests = [], []
    rng = np.random.default_rng(RNG)
    for scope in ['panel', 'full']:
        for q in ['q1', 'q2', 'q3']:
            loaded = {m: load(m, scope, q) for m in HNN + GNNS + BASELINES}
            loaded = {m: v for m, v in loaded.items() if v[0] is not None}
            for m, (runs, _) in loaded.items():
                fam = 'HNN4RP (adapted)' if m in HNN else 'official GNN' if m in GNNS else 'baseline'
                row = {'scope': scope, 'window': q, 'family': fam, 'model': m}
                for part in ['train', 'val', 'test']:
                    g = runs[runs.partition == part]
                    for k in METRICS:
                        row[f'{part} {k}'] = f'{g[k].mean():.3f} ± {g[k].std():.3f}' if len(g) else '—'
                row['_val_ap'] = runs[runs.partition == 'val'].pr_auc.mean()
                table.append(row)
            if 'hnn4rp' not in loaded:
                continue
            cell = pd.DataFrame([r for r in table if r['scope'] == scope and r['window'] == q])
            bases = cell[cell.family == 'baseline']
            best = bases.sort_values('_val_ap').model.iloc[-1] if len(bases) else None
            refs = ([('xgboost', 'XGBoost'), ('graphsage', 'official GraphSAGE')]
                    + ([(best, f'best baseline by val AP ({best})')] if best not in (None, 'xgboost') else []))
            for ref, tag in refs:
                if ref in loaded:
                    tests.append({'scope': scope, 'window': q, 'reference': tag,
                                  **paired(loaded['hnn4rp'][1], loaded[ref][1], rng)})
            print(f'{scope} {q} done', flush=True)
    table = pd.DataFrame(table).drop(columns='_val_ap')
    tests = pd.DataFrame(tests)
    for c in ['p (AP)', 'p (ROC)', 'McNemar p']:
        tests[f'{c} Holm'] = holm(tests[c])
    table.to_csv(OUT / 'hnn4rp_vs_project_models.csv', index=False)
    tests.to_csv(OUT / 'hnn4rp_paired_tests.csv', index=False)
    pd.set_option('display.width', 260); pd.set_option('display.max_rows', 400)
    print(table[['scope', 'window', 'model', 'train roc_auc', 'val roc_auc', 'test roc_auc', 'test pr_auc',
                 'test f1', 'test precision', 'test recall']].to_string(index=False))
    print(tests.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
