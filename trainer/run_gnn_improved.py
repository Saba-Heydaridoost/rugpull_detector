"""Train the direction-aware GNNs (dirsage, dirgcn) with the exact run_gnn.py protocol.

For each scope and window: hyperparameter search on validation with Test frozen, then the
frozen configuration is re-run over the seeds with Test scoring enabled (final_* files),
mirroring the official GNN runs. Results go to results/gnn_improved/ (chronological split).

Usage: python run_gnn_improved.py --experiments q1 [--scopes panel full] [--models dirsage dirgcn]
"""
import argparse
import os
import warnings

warnings.filterwarnings('ignore')

from gnn import config  # noqa: E402
import run_gnn  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--experiments', nargs='+', required=True)
    p.add_argument('--scopes', nargs='+', default=['panel', 'full'])
    p.add_argument('--models', nargs='+', default=['dirsage', 'dirgcn'])
    p.add_argument('--n-trials', type=int, default=None)
    p.add_argument('--n-seeds', type=int, default=None)
    p.add_argument('--result-dir', default=None)
    a = p.parse_args()
    config.RESULT_DIR = os.path.abspath(a.result_dir or os.path.join(config.RESULT_ROOT, 'gnn_improved'))
    config.PLOT_DIR = os.path.join(config.RESULT_DIR, 'plots')
    os.makedirs(config.PLOT_DIR, exist_ok=True)
    common = dict(models=a.models, cohorts=['original'], feature_variants=['base'], variants=['no_mask'],
                  n_trials=a.n_trials, n_seeds=a.n_seeds, refit=False, skip_existing=True)
    for q in a.experiments:
        for scope in a.scopes:
            config.EVALUATE_TEST = False
            run_gnn.run(scopes=[scope], experiments=[q], reuse_best_params=False, **common)
            config.EVALUATE_TEST = True
            run_gnn.run(scopes=[scope], experiments=[q], reuse_best_params=True, **common)


if __name__ == '__main__':
    main()
