"""Retrain the compared models on the multi-type (v2) labels, separately from the v1 results.

Data: artifacts_labels_v2/ (built by build_labels_v2.py); results: results_labels_v2/.
Same protocol as the v1 final runs (chronological split, original cohort, no_mask, node_graph
view, panel and full scope): every model reuses its v1 validation-selected hyperparameters and
is retrained over the same five seeds with Test scoring, so differences reflect the labels.

Models: 9 baselines (run_baselines.py), GCN / GraphSAGE / GAT (run_gnn.py), DirSAGE / DirGCN,
and the 5-fold bagged GraphSAGE plus GraphSAGE -> GB / XGB stacks (run_gnn_stacked.py).

Usage: python run_labels_v2.py --experiments q1 [--parts baselines gnn stacked]
"""
import argparse
import os
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

from baselines import config as bconfig  # noqa: E402
from gnn import config as gconfig  # noqa: E402

BASE = Path(__file__).resolve().parent
ART = BASE / 'artifacts_labels_v2'
RES = BASE / 'results_labels_v2'
for c in (bconfig, gconfig):
    c.ARTIFACT_DIR = str(ART)
    c.COHORT_DIR = str(ART / 'final' / 'missingness_experiments')
bconfig.RESULT_DIR = str(RES)
bconfig.PLOT_DIR = str(RES / 'plots')
gconfig.RESULT_ROOT = str(RES)
gconfig.RESULT_DIR = str(RES / 'gnn')
gconfig.PLOT_DIR = str(RES / 'gnn' / 'plots')

import run_baselines  # noqa: E402
import run_gnn  # noqa: E402
import run_gnn_stacked  # noqa: E402

GNNS = ['gcn', 'graphsage', 'gat', 'dirsage', 'dirgcn']


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--experiments', nargs='+', required=True)
    p.add_argument('--parts', nargs='+', default=['baselines', 'gnn', 'stacked'])
    a = p.parse_args()
    for d in [RES / 'plots', RES / 'gnn' / 'plots', RES / 'gnn_stacked']:
        os.makedirs(d, exist_ok=True)
    for q in a.experiments:
        if 'baselines' in a.parts:
            bconfig.EVALUATE_TEST = True
            run_baselines.run(scopes=['panel', 'full'], experiments=[q], views=['node_graph'], variants=['no_mask'],
                              cohorts=['original'], reuse_best_params=True, refit=False)
        if 'gnn' in a.parts:
            gconfig.EVALUATE_TEST = True
            run_gnn.run(models=GNNS, scopes=['panel', 'full'], experiments=[q], cohorts=['original'],
                        feature_variants=['base'], variants=['no_mask'], reuse_best_params=True, refit=False,
                        skip_existing=True)
        if 'stacked' in a.parts:
            run_gnn_stacked.RES = RES
            run_gnn_stacked.OUT = RES / 'gnn_stacked'
            run_gnn_stacked.BASELINE_RUNS = RES          # seeded v1 parameter files live here
            orig = run_gnn_stacked.frozen_params

            def frozen_params(scope, q_):
                import ast
                import pandas as pd
                g = pd.read_csv(RES / 'gnn' / f'final_runs_graphsage_{scope}_{q_}_gnn_graph_structure_timing_node_graph_no_mask_original.csv')
                boost = {n: ast.literal_eval(pd.read_csv(RES / f'runs_{n}_{scope}_{q_}_node_graph_no_mask_original.csv').params.iloc[0])
                         for n in run_gnn_stacked.BOOSTERS}
                return ast.literal_eval(g.params.iloc[0]), boost
            run_gnn_stacked.frozen_params = frozen_params
            for scope in ['panel', 'full']:
                if not (RES / 'gnn_stacked' / f'predictions_graphsage_bagged_{scope}_{q}.csv').exists():
                    run_gnn_stacked.run_cell(scope, q)
            run_gnn_stacked.frozen_params = orig


if __name__ == '__main__':
    main()
