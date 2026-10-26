"""Sensitivity analysis: XGBoost re-tuned with scale_pos_weight in its search space.

The official XGBoost baseline keeps scale_pos_weight fixed at 1.0. Here the existing Optuna
space (baselines/models/xgboost_model.SPACE) is extended with
scale_pos_weight in {1, n_neg / n_pos}, where the ratio comes from each cell's training
partition. Everything else is unchanged: 60 trials, validation ROC-AUC objective, TPE seed 0,
five seeds, F1-max threshold on validation, chronological split.

Cells: panel and full x q1-q3, original cohort, no_mask, node_graph view.
Results go to results_xgb_spw/ (the official results/ are not touched).

Usage: python run_xgb_spw_sensitivity.py
"""
import os
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

from baselines import config as bconfig, data as bdata  # noqa: E402
from baselines.models import xgboost_model  # noqa: E402

OUT = Path(__file__).resolve().parent / 'results_xgb_spw'
bconfig.RESULT_DIR = str(OUT)
bconfig.PLOT_DIR = str(OUT / 'plots')
bconfig.EVALUATE_TEST = True
bconfig.MAKE_PLOTS = False

import run_baselines  # noqa: E402


def space_with_ratio(ratio):
    def space(trial):
        params = xgboost_model.SPACE(trial)
        params['scale_pos_weight'] = trial.suggest_categorical('scale_pos_weight', [1.0, float(ratio)])
        return params
    return space


def main():
    os.makedirs(OUT, exist_ok=True)
    for scope in ['panel', 'full']:
        for q in ['q1', 'q2', 'q3']:
            d = bdata.get_data(scope=scope, q=q, view='node_graph', variant='no_mask', cohort='original')
            ratio = round(float((d.ytr == 0).sum() / (d.ytr == 1).sum()), 4)
            print(f'[{scope} {q}] n_train={len(d.ytr)} n_neg/n_pos={ratio}', flush=True)
            run_baselines.run(models=['xgboost'], scopes=[scope], experiments=[q], views=['node_graph'],
                              variants=['no_mask'], cohorts=['original'],
                              space_overrides={'xgboost': space_with_ratio(ratio)}, refit=False)


if __name__ == '__main__':
    main()
