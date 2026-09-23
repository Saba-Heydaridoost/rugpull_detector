"""Gradient-boosted trees via XGBoost."""

from .. import config
from ..evaluation import evaluate, sklearn_predictor

try:
    import xgboost as xgb

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

NAME = 'xgboost'


def SPACE(trial):
    """`subsample` and `colsample_bytree` were pinned at 0.8 by the grid; both are searched
    now, which is most of the point of moving off an exhaustive product."""
    return {
        'n_estimators': trial.suggest_int('n_estimators', 200, 800, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 2, 8),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 30),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 20.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
    }


def build(params, seed):
    if not HAS_XGB:
        raise ImportError('xgboost is not installed')

    return xgb.XGBClassifier(
        n_estimators=params.get('n_estimators', 400),
        learning_rate=params.get('learning_rate', 0.05),
        max_depth=params.get('max_depth', 4),
        min_child_weight=params.get('min_child_weight', 1),
        reg_lambda=params.get('reg_lambda', 1.0),
        reg_alpha=params.get('reg_alpha', 0.0),
        subsample=params.get('subsample', 0.8),
        colsample_bytree=params.get('colsample_bytree', 0.8),
        scale_pos_weight=params.get('scale_pos_weight', 1.0),
        eval_metric='logloss',
        tree_method='hist',
        n_jobs=config.N_JOBS,
        random_state=seed,
    )


def train(data, space=None, n_trials=None, n_seeds=None, **kw):
    if not HAS_XGB:
        raise ImportError('xgboost is not installed')

    def make_predictor(params, d, seed):
        return sklearn_predictor(build, params, d, seed)

    return evaluate(NAME, make_predictor, space or SPACE, data,
                    n_trials=n_trials, n_seeds=n_seeds, **kw)
