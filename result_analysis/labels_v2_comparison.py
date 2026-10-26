"""Model comparison on the multi-type (v2) labels, side by side with the v1 labels.

Uses the scoring and testing of gnn_improvement_table.py unchanged (seed ensembles, F1-max
thresholds on validation, best baseline chosen by validation AP, paired label-stratified bootstrap,
Holm correction across all contrasts), applied to trainer/results_labels_v2 (built by
trainer/build_labels_v2.py + run_labels_v2.py). v1 numbers are read from the existing
output/test/gnn_improvement/ tables. Note that v1 and v2 test sets differ (T3 tokens are
excluded in v2 and T2 tokens are positive), so v1-v2 differences are not paired.
"""
from pathlib import Path

import pandas as pd

import gnn_improvement_table as g

BASE = Path(__file__).resolve().parent
V2 = BASE.parent / 'trainer' / 'results_labels_v2'
OUT = BASE / 'output' / 'test' / 'labels_v2'


def source_v2(model, scope, q):
    if model in g.OFFICIAL_GNN + g.DIR_GNN:
        grp = 'official GNN (run_gnn.py)' if model in g.OFFICIAL_GNN else 'direction-aware GNN (new)'
        return V2 / 'gnn' / f'final_predictions_{model}_{scope}_{q}_{g.SUFFIX}.csv', grp
    if model in g.STACKED:
        return V2 / 'gnn_stacked' / f'predictions_{model}_{scope}_{q}.csv', 'GNN bagging / stacking (new)'
    return V2 / f'final_predictions_{model}_{scope}_{q}_node_graph_no_mask_original.csv', 'official baseline'


def main():
    g.source = source_v2
    g.OUT = OUT
    g.N_BOOT = 5000
    g.main()
    v2 = pd.read_csv(OUT / 'gnn_improvement_results.csv')
    v1 = pd.read_csv(BASE / 'output' / 'test' / 'gnn_improvement' / 'gnn_improvement_results.csv')
    rows = []
    for label, d in [('v1', v1), ('v2', v2)]:
        for metric in ['test AP (ens)', 'test ROC-AUC (ens)', 'test F1 (ens)']:
            p = d.pivot_table(index=['group', 'model'], columns=['scope', 'window'], values=metric, sort=False)
            p['mean'] = p.mean(1)
            p.columns = [f'{label} {metric} {c[0]} {c[1]}' if isinstance(c, tuple) and c[1] else f'{label} {metric} mean'
                         for c in p.columns]
            rows.append(p)
    side = pd.concat(rows, axis=1)
    side.to_csv(OUT / 'v1_vs_v2_side_by_side.csv')
    tv2 = pd.read_csv(OUT / 'gnn_improvement_paired_tests.csv')
    print('\nv2 contrasts significant after Holm:',
          int((tv2['p (AP) Holm'] < 0.05).sum()), '(AP),', int((tv2['p (ROC) Holm'] < 0.05).sum()), '(ROC)')


if __name__ == '__main__':
    main()
