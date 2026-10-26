"""Build a separate copy of the trainer artifacts with the multi-type (v2) rug-pull labels.

The original artifacts are not modified. Source: artifacts/final/missingness_experiments/original/q1-q3.
Destination: artifacts_labels_v2/final/missingness_experiments/original/q1-q3.

v2 label (dataset/rule_based_labels_v2_types.csv, derived from the stored rule-based label fields;
all evidence lies after the prediction cutoff, as in v1):
  rug (1)      T1 hard rug (v1 positive), T2 LP pull with frozen price, T4 drain with no trading
               afterwards, T5 drain after which the pool's liquidity history ends
  non-rug (0)  no liquidity drain within the horizon, or a temporary dip
  excluded     T3 drain with >= 90% price collapse but liquidity partly restored (ambiguous)
T4/T5 tokens were UNKNOWN in v1 and have no feature rows, so they cannot enter the model tables.

Also seeds results_labels_v2/ with params-only copies of the v1 validation-selected
hyperparameters, so run_labels_v2.py retrains each model with its v1 configuration.
"""
import glob
import os
import shutil

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, 'artifacts', 'final', 'missingness_experiments', 'original')
DST = os.path.join(BASE, 'artifacts_labels_v2', 'final', 'missingness_experiments', 'original')
TYPES = os.path.join(os.path.dirname(BASE), 'dataset', 'rule_based_labels_v2_types.csv')
RES_V1 = os.path.join(BASE, 'results')
RES_V2 = os.path.join(BASE, 'results_labels_v2')
POS = {'T1 hard rug (current)', 'T2 LP pull, frozen price', 'T4 drain, no trading after', 'T5 drain, pool history ends'}
NEG = {'non-rug: no drain', 'non-rug: temporary dip'}


def main():
    types = pd.read_csv(TYPES)
    types['label_v2'] = np.where(types.type.isin(POS), 1, np.where(types.type.isin(NEG), 0, -1))
    report = []
    for q in ['q1', 'q2', 'q3']:
        lab = types[types.experiment_id == q].drop_duplicates('token_mint').set_index('token_mint')
        src, dst = os.path.join(SRC, q), os.path.join(DST, q)
        os.makedirs(dst, exist_ok=True)
        node = pd.read_parquet(os.path.join(src, 'model_nodes_raw_nan.parquet'), columns=['token_mint', 'rug_pull_label'])
        t = lab.reindex(node.token_mint)
        assert t.type.notna().all(), f'{q}: model tokens without a v2 type'
        assert (t.label_v2.values[t.type.isin(['T1 hard rug (current)'])] == 1).all()
        keep = set(node.token_mint[t.label_v2.values >= 0])
        for f in sorted(glob.glob(os.path.join(src, '*.parquet'))):
            d = pd.read_parquet(f)
            d = d[d.token_mint.isin(keep)].copy()
            if 'rug_pull_label' in d.columns:
                d['rug_pull_label'] = lab.loc[d.token_mint, 'label_v2'].to_numpy().astype(d['rug_pull_label'].dtype)
            d.to_parquet(os.path.join(dst, os.path.basename(f)), index=False)
        tv = t.assign(v1=node.rug_pull_label.values)
        report.append({'window': q, 'tokens v1': len(node), 'tokens v2': len(keep),
                       'dropped (T3)': int((tv.type == 'T3 partial-LP / relist rug').sum()),
                       'flipped 0->1 (T2)': int(((tv.v1 == 0) & (tv.label_v2 == 1)).sum()),
                       'rug rate v1': node.rug_pull_label.mean(),
                       'rug rate v2': tv.loc[tv.label_v2 >= 0, 'label_v2'].mean()})
    print(pd.DataFrame(report).round(3).to_string(index=False))

    # params-only copies of v1 selections (n_train dropped: the v2 population differs by design)
    for sub, pattern in [('', 'runs_*_node_graph_no_mask_original.csv'),
                         ('gnn', 'runs_*_gnn_graph_structure_timing_node_graph_no_mask_original.csv')]:
        os.makedirs(os.path.join(RES_V2, sub), exist_ok=True)
    seeds = []
    for scope in ['panel', 'full']:
        seeds += glob.glob(os.path.join(os.path.dirname(BASE), 'result_analysis', 'input', f'baseline_{scope}',
                                        'runs_*_node_graph_no_mask_original.csv'))
    for f in seeds:
        r = pd.read_csv(f).drop(columns=['n_train'], errors='ignore')
        r.to_csv(os.path.join(RES_V2, os.path.basename(f)), index=False)
    gnn_seeds = glob.glob(os.path.join(RES_V1, 'gnn', 'runs_*_gnn_graph_structure_timing_node_graph_no_mask_original.csv'))
    gnn_seeds += glob.glob(os.path.join(RES_V1, 'gnn_improved', 'runs_*_gnn_graph_structure_timing_node_graph_no_mask_original.csv'))
    for f in gnn_seeds:
        r = pd.read_csv(f).drop(columns=['n_train'], errors='ignore')
        r.to_csv(os.path.join(RES_V2, 'gnn', os.path.basename(f)), index=False)
    print(f'seeded {len(seeds)} baseline and {len(gnn_seeds)} GNN frozen-parameter files')


if __name__ == '__main__':
    main()
