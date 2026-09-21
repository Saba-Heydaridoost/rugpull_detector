"""Bagged tree ensemble."""

from sklearn.ensemble import RandomForestClassifier

from ..evaluation import evaluate, sklearn_predictor

NAME = 'random_forest'

GRID = [
    {'n_estimators': s, 'max_depth': d, 'min_samples_leaf': m,
     'max_features': f, 'class_weight': w}
    for s in [400, 500, 600]
    for d in [4, 6, 8, 10, None]
    for m in [1, 5, 20, 30]
    for f in ['sqrt', 0.5, 0.75, 1.0]
    for w in [None, 'balanced']
]


def build(params, seed):
    return RandomForestClassifier(
        n_estimators=params.get('n_estimators', 500),
        max_depth=params.get('max_depth'),
        min_samples_leaf=params.get('min_samples_leaf', 1),
        max_features=params.get('max_features', 'sqrt'),
        class_weight=params.get('class_weight'),
        n_jobs=-1,
        random_state=seed,
    )


def train(data, grid=None, n_seeds=None, **kw):
    def make_predictor(params, d, seed):
        return sklearn_predictor(build, params, d, seed)

    return evaluate(NAME, make_predictor, grid or GRID, data, n_seeds=n_seeds, **kw)
