"""Single CART tree. Included mainly as an interpretable reference point."""

from sklearn.tree import DecisionTreeClassifier

from ..evaluation import evaluate, sklearn_predictor

NAME = 'decision_tree'

GRID = [
    {'max_depth': d, 'min_samples_leaf': m, 'criterion': c, 'class_weight': w}
    for d in [2, 3, 5, 6, 7, 8, None]
    for m in [1, 10, 30, 50, 60, 70]
    for c in ['gini', 'entropy', 'log_loss']
    for w in [None, 'balanced']
]


def build(params, seed):
    return DecisionTreeClassifier(
        max_depth=params.get('max_depth'),
        min_samples_leaf=params.get('min_samples_leaf', 1),
        criterion=params.get('criterion', 'gini'),
        class_weight=params.get('class_weight'),
        ccp_alpha=params.get('ccp_alpha', 0.0),
        random_state=seed,
    )


def train(data, grid=None, n_seeds=None, **kw):
    def make_predictor(params, d, seed):
        return sklearn_predictor(build, params, d, seed)

    return evaluate(NAME, make_predictor, grid or GRID, data, n_seeds=n_seeds, **kw)
