"""Regularised linear baseline."""

from sklearn.linear_model import LogisticRegression

from ..evaluation import evaluate, sklearn_predictor

NAME = 'logistic_regression'


def SPACE(trial):
    """C spans six orders of magnitude on a log scale.

    The grid this replaced clustered its points around 0.05 after a manual narrowing pass;
    a log-uniform range lets the search re-find that region without pinning it there.
    """
    return {
        'C': trial.suggest_float('C', 1e-4, 1e2, log=True),
        'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
    }


def build(params, seed):
    return LogisticRegression(
        C=params.get('C', 1.0),
        class_weight=params.get('class_weight'),
        solver=params.get('solver', 'lbfgs'),
        max_iter=params.get('max_iter', 5000),
        random_state=seed,
    )


def train(data, space=None, n_trials=None, n_seeds=None, **kw):
    def make_predictor(params, d, seed):
        return sklearn_predictor(build, params, d, seed)

    return evaluate(NAME, make_predictor, space or SPACE, data,
                    n_trials=n_trials, n_seeds=n_seeds, **kw)
