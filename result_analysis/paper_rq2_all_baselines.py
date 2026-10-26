"""RQ2 robustness: GNNs at q1 against every non-graph baseline at q1, q2 and q3 (panel).

Same estimator as the frozen primary analysis (gnn_primary_inference.py): panel, original
cohort, no_mask, node_graph view for the baselines; endpoint = mean per-seed test AP difference;
paired label-stratified token bootstrap shared across the five seeds (paired_bootstrap.bootstrap),
10,000 replicates, RNG seed 2026. The primary analysis itself is not changed or re-run here.

Families (all exploratory):
  all_baselines   3 GNNs x 8 baselines x 3 windows = 72 contrasts; unadjusted 95% CI, bootstrap
                  p = 2 min((#{d<=0}+1)/(B+1), (#{d>=0}+1)/(B+1)), Holm across the 72.
  val_selected    3 GNNs x 3 windows = 9 contrasts against the baseline with the highest mean
                  validation ROC-AUC in each window; Bonferroni simultaneous intervals at
                  marginal confidence 1 - 0.05/9, exactly as in the primary analysis.

Outputs: output/test/paper_tables/rq2_all_baselines.csv, rq2_val_selected.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

from paired_bootstrap import bootstrap

BASE = Path(__file__).resolve().parent
RES = BASE.parent / 'trainer' / 'results'
INP = BASE / 'input'
OUT = BASE / 'output' / 'test' / 'paper_tables'
GNN_SUFFIX = 'gnn_graph_structure_timing_node_graph_no_mask_original'
GNNS = ['gcn', 'graphsage', 'gat']
BASELINES = ['logistic_regression', 'decision_tree', 'random_forest', 'gradient_boosting', 'xgboost', 'mlp',
             'ft_transformer', 'autoint']
WINDOWS = ['q1', 'q2', 'q3']
B, SEED, ALPHA = 10000, 2026, 0.05


def matrix(path):
    d = pd.read_csv(path)
    d = d[d.partition == 'test']
    S = d.pivot(index='token_mint', columns='seed', values='y_score').sort_index()
    y = d.groupby('token_mint').y_true.first().reindex(S.index).astype(int)
    assert list(S.columns) == [0, 1, 2, 3, 4], path
    return S, y


def val_roc(m, q):
    r = pd.read_csv(INP / 'baseline_panel' / f'runs_{m}_panel_{q}_node_graph_no_mask_original.csv')
    return r.loc[r.partition == 'val', 'roc_auc'].mean()


def boot_p(d):
    return min(1.0, 2 * min(((d <= 0).sum() + 1) / (len(d) + 1), ((d >= 0).sum() + 1) / (len(d) + 1)))


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[o])
    out = np.empty_like(p)
    out[o] = np.minimum(adj, 1)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gnn = {m: matrix(INP / 'gnn_panel' / f'final_predictions_{m}_panel_q1_{GNN_SUFFIX}.csv') for m in GNNS}
    base = {(m, q): matrix(RES / f'final_predictions_{m}_panel_{q}_node_graph_no_mask_original.csv')
            for m in BASELINES for q in WINDOWS}
    selected = {q: max(BASELINES, key=lambda m: val_roc(m, q)) for q in WINDOWS}
    rows, cache = [], {}
    for g in GNNS:
        Sg, yg = gnn[g]
        for q in WINDOWS:
            for m in BASELINES:
                Sb, yb = base[(m, q)]
                assert Sg.index.equals(Sb.index) and yg.equals(yb)
                point, diffs, ci, *_ = bootstrap(yg.values, Sg.values, Sb.values, B, SEED, 1 - ALPHA)
                cache[(g, q, m)] = (point, diffs[:, 0])
                rows.append({'gnn (q1)': g, 'baseline': m, 'baseline window': q,
                             'validation-selected baseline': m == selected[q],
                             'n_tokens': len(yg), 'n_positive': int(yg.sum()),
                             'gnn AP': point['ap_a'], 'baseline AP': point['ap_b'], 'ΔAP': point['delta_ap'],
                             'ΔAP 95% CI low': ci[0, 0], 'ΔAP 95% CI high': ci[1, 0], 'p': boot_p(diffs[:, 0])})
            print(f'{g} vs all baselines at {q} done', flush=True)
    t = pd.DataFrame(rows)
    t['p Holm (72)'] = holm(t['p'])
    t.to_csv(OUT / 'rq2_all_baselines.csv', index=False)

    sel = []
    lo_q, hi_q = ALPHA / (2 * 9), 1 - ALPHA / (2 * 9)
    for g in GNNS:
        for q in WINDOWS:
            point, d = cache[(g, q, selected[q])]
            lo, hi = np.quantile(d, [lo_q, hi_q])
            sel.append({'gnn (q1)': g, 'baseline window': q, 'validation-selected baseline': selected[q],
                        'gnn AP': point['ap_a'], 'baseline AP': point['ap_b'], 'ΔAP': point['delta_ap'],
                        'Bonferroni CI low': lo, 'Bonferroni CI high': hi, 'supports superiority': bool(lo > 0),
                        'marginal confidence': 1 - ALPHA / 9})
    s = pd.DataFrame(sel)
    s.to_csv(OUT / 'rq2_val_selected.csv', index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_rows', 100)
    print(s.round(3).to_string(index=False))
    print(t.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
