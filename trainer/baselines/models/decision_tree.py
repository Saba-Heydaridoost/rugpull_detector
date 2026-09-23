"""Single CART tree. Included mainly as an interpretable reference point."""

from sklearn.tree import DecisionTreeClassifier

from ..evaluation import evaluate, sklearn_predictor

NAME = 'decision_tree'


def SPACE(trial):
    """Unlimited depth is drawn as its own branch rather than as a sentinel value.

    A conditional parameter is the honest encoding: `max_depth=None` is a different kind of
    choice from `max_depth=16`, and mixing them into one distribution would have the sampler
    interpolating between a number and an absence.
    """
    params = {
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 80),
        'criterion': trial.suggest_categorical(
            'criterion', ['gini', 'entropy', 'log_loss']
        ),
        'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
    }

    if trial.suggest_categorical('limit_depth', [True, False]):
        params['max_depth'] = trial.suggest_int('max_depth', 2, 16)
    else:
        params['max_depth'] = None

    return params


def build(params, seed):
    return DecisionTreeClassifier(
        max_depth=params.get('max_depth'),
        min_samples_leaf=params.get('min_samples_leaf', 1),
        criterion=params.get('criterion', 'gini'),
        class_weight=params.get('class_weight'),
        ccp_alpha=params.get('ccp_alpha', 0.0),
        random_state=seed,
    )


def train(data, space=None, n_trials=None, n_seeds=None, **kw):
    def make_predictor(params, d, seed):
        return sklearn_predictor(build, params, d, seed)

    return evaluate(NAME, make_predictor, space or SPACE, data,
                    n_trials=n_trials, n_seeds=n_seeds, **kw)
