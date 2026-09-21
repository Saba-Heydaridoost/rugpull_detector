"""Regularised linear baseline."""

from sklearn.linear_model import LogisticRegression

from ..evaluation import evaluate, sklearn_predictor

NAME = 'logistic_regression'

GRID = [
    {'C': c, 'class_weight': w}
    for c in [0.001, 0.003, 0.01, 0.03, 0.05, 0.06, 0.07, 0.08, 0.1, 0.3, 1.0, 3.0, 10.0]
    for w in [None, 'balanced']
]


def build(params, seed):
    return LogisticRegression(
        C=params.get('C', 1.0),
        class_weight=params.get('class_weight'),
        solver=params.get('solver', 'lbfgs'),
        max_iter=params.get('max_iter', 5000),
        random_state=seed,
    )


def train(data, grid=None, n_seeds=None, **kw):
    def make_predictor(params, d, seed):
        return sklearn_predictor(build, params, d, seed)

    return evaluate(NAME, make_predictor, grid or GRID, data, n_seeds=n_seeds, **kw)
