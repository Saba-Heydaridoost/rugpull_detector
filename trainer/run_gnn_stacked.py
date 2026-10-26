"""Stacked GNN + gradient-boosting hybrid, and fold-bagged GraphSAGE (chronological split).

For each window and scope (original cohort, no_mask, node_graph view):

1. GraphSAGE with its official, already-selected hyperparameters (results/gnn final runs) is
   cross-fitted on the training tokens (stratified 5-fold, per seed). Each training token gets
   an out-of-fold P(rug); validation and test tokens get the mean over the five fold models.
   The fold average on its own is reported as "GraphSAGE (5-fold bagged)".
2. Gradient boosting and XGBoost with their official hyperparameters are fit on the
   node_graph features, without and with the GNN's out-of-fold logit as one extra feature.
   The pair isolates what the learned graph representation adds to the trees.

Nothing is tuned here; all hyperparameters come from earlier validation-based searches.
Thresholds (F1-max) are chosen on validation per seed. Test is scored once per model and seed.

Usage: python run_gnn_stacked.py --experiments q1 [--scopes panel full]
Outputs: results/gnn_stacked/predictions_<model>_<scope>_<q>.csv
"""
import argparse
import ast
import dataclasses
import os
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from gnn import config as gconfig, data as gdata_mod  # noqa: E402
from gnn.models import graphsage  # noqa: E402
from gnn.torch_utils import torch_train_kwargs, train_gnn_model  # noqa: E402
from baselines import data as bdata_mod  # noqa: E402
from baselines.models import gradient_boosting, xgboost_model  # noqa: E402

ROOT = Path(__file__).resolve().parent
RES = ROOT / 'results'
BASELINE_RUNS = ROOT.parent / 'result_analysis' / 'input'
OUT = RES / 'gnn_stacked'
SEEDS = range(5)
BOOSTERS = {'gradient_boosting': gradient_boosting.build, 'xgboost': xgboost_model.build}


def frozen_params(scope, q):
    g = pd.read_csv(RES / 'gnn' / f'final_runs_graphsage_{scope}_{q}_gnn_graph_structure_timing_node_graph_no_mask_original.csv')
    gnn_params = ast.literal_eval(g.params.iloc[0])
    boost = {}
    for name in BOOSTERS:
        b = pd.read_csv(BASELINE_RUNS / f'baseline_{scope}' / f'runs_{name}_{scope}_{q}_node_graph_no_mask_original.csv')
        boost[name] = ast.literal_eval(b.params.iloc[0])
    return gnn_params, boost


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def run_cell(scope, q):
    gconfig.EVALUATE_TEST = True
    g = gdata_mod.get_data(scope, q, 'original', variant='no_mask', feature_variant='base')
    b = bdata_mod.get_data(scope, q, view='node_graph', variant='no_mask', cohort='original')
    gnn_params, boost_params = frozen_params(scope, q)

    # Align the GNN token order to the baseline design matrices.
    def order(g_mints, b_mints):
        pos = {m: i for i, m in enumerate(g_mints)}
        missing = [m for m in b_mints if m not in pos]
        assert not missing, f'{len(missing)} baseline tokens without a graph'
        return np.array([pos[m] for m in b_mints])
    o_tr = order(list(g.train_token_mints), list(b.train_token_mints))
    o_va = order(list(g.val_token_mints), list(b.val_token_mints))
    o_te = order(list(g.test_token_mints), list(b.test_token_mints))
    assert (g.ytr[o_tr] == b.ytr).all() and (g.yva[o_va] == b.yva).all() and (g.yte[o_te] == b.yte).all()

    rows = []

    def record(model, seed, p_va, p_te):
        for part, mints, y, p in [('val', b.val_token_mints, b.yva, p_va), ('test', b.test_token_mints, b.yte, p_te)]:
            rows.extend({'model': model, 'seed': seed, 'partition': part, 'token_mint': m, 'y_true': int(t),
                         'y_score': float(s)} for m, t, s in zip(mints, y, p))

    for seed in SEEDS:
        oof = np.zeros(len(g.train_graphs))
        va_fold, te_fold = [], []
        folds = StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(g.ytr)), g.ytr)
        for k, (fit_idx, held_idx) in enumerate(folds):
            d_fold = dataclasses.replace(g, train_graphs=[g.train_graphs[i] for i in fit_idx])
            predict = train_gnn_model(graphsage.build(gnn_params), d_fold, seed * 10 + k, **torch_train_kwargs(gnn_params))
            oof[held_idx] = predict([g.train_graphs[i] for i in held_idx])
            va_fold.append(predict(g.val_graphs))
            te_fold.append(predict(g.test_graphs))
        gnn_va, gnn_te = np.mean(va_fold, 0), np.mean(te_fold, 0)
        record('graphsage_bagged', seed, gnn_va[o_va], gnn_te[o_te])
        z_tr, z_va, z_te = logit(oof[o_tr])[:, None], logit(gnn_va[o_va])[:, None], logit(gnn_te[o_te])[:, None]
        for name, build in BOOSTERS.items():
            base = build(boost_params[name], seed).fit(b.Xtr, b.ytr)
            record(name, seed, base.predict_proba(b.Xva)[:, 1], base.predict_proba(b.Xte)[:, 1])
            stacked = build(boost_params[name], seed).fit(np.hstack([b.Xtr, z_tr]), b.ytr)
            record(f'{name}+graphsage_stack', seed, stacked.predict_proba(np.hstack([b.Xva, z_va]))[:, 1],
                   stacked.predict_proba(np.hstack([b.Xte, z_te]))[:, 1])
        print(f'[{scope} {q}] seed {seed} done', flush=True)
    out = pd.DataFrame(rows)
    for model, d in out.groupby('model'):
        d.assign(scope=scope, experiment=q).to_csv(OUT / f'predictions_{model}_{scope}_{q}.csv', index=False)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--experiments', nargs='+', required=True)
    p.add_argument('--scopes', nargs='+', default=['panel', 'full'])
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for q in a.experiments:
        for scope in a.scopes:
            if (OUT / f'predictions_graphsage_bagged_{scope}_{q}.csv').exists():
                print(f'skip {scope} {q} (done)')
                continue
            run_cell(scope, q)


if __name__ == '__main__':
    main()
