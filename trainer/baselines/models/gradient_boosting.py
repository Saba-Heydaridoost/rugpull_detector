"""Scikit-learn gradient boosting.

Note that this estimator cannot consume NaN directly; the shared preprocessor already
imputes, so the design matrix it receives is dense.
"""

from sklearn.ensemble import GradientBoostingClassifier

from ..evaluation import evaluate, sklearn_predictor

NAME = 'gradient_boosting'

GRID = [
    {'n_estimators': n, 'learning_rate': lr, 'max_depth': d,
     'subsample': sub, 'min_samples_leaf': m}
    for n in [200, 500]
    for lr in [0.02, 0.05, 0.1]
    for d in [2, 3, 5]
    for sub in [1.0, 0.8]
    for m in [1, 20]
]


def build(params, seed):
    return GradientBoostingClassifier(
        n_estimators=params.get('n_estimators', 300),
        learning_rate=params.get('learning_rate', 0.05),
        max_depth=params.get('max_depth', 3),
        subsample=params.get('subsample', 1.0),
        min_samples_leaf=params.get('min_samples_leaf', 1),
        random_state=seed,
    )


def train(data, grid=None, n_seeds=None, **kw):
    def make_predictor(params, d, seed):
        return sklearn_predictor(build, params, d, seed)

    return evaluate(NAME, make_predictor, grid or GRID, data, n_seeds=n_seeds, **kw)
