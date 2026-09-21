"""Gradient-boosted trees via XGBoost."""

from ..evaluation import evaluate, sklearn_predictor

try:
    import xgboost as xgb

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

NAME = 'xgboost'

GRID = [
    {'n_estimators': n, 'learning_rate': lr, 'max_depth': d,
     'min_child_weight': w, 'reg_lambda': rl, 'subsample': 0.8, 'colsample_bytree': 0.8}
    for n in [300, 600]
    for lr in [0.02, 0.05, 0.1]
    for d in [3, 4, 6]
    for w in [1, 5, 20]
    for rl in [1.0, 5.0]
]


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
        n_jobs=-1,
        random_state=seed,
    )


def train(data, grid=None, n_seeds=None, **kw):
    if not HAS_XGB:
        raise ImportError('xgboost is not installed')

    def make_predictor(params, d, seed):
        return sklearn_predictor(build, params, d, seed)

    return evaluate(NAME, make_predictor, grid or GRID, data, n_seeds=n_seeds, **kw)
