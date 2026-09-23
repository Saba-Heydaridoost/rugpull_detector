"""Bagged tree ensemble."""

from sklearn.ensemble import RandomForestClassifier

from .. import config
from ..evaluation import evaluate, sklearn_predictor

NAME = 'random_forest'

# `max_features` mixes a rule name with a fraction, which a single categorical
# distribution cannot hold without mixing types. The choice is drawn over labels and
# resolved here.
MAX_FEATURES = {'sqrt': 'sqrt', 'log2': 'log2', '0.5': 0.5, '0.75': 0.75, '1.0': 1.0}


def SPACE(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 800, step=100),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 40),
        'max_features': MAX_FEATURES[
            trial.suggest_categorical('max_features', list(MAX_FEATURES))
        ],
        'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
    }

    if trial.suggest_categorical('limit_depth', [True, False]):
        params['max_depth'] = trial.suggest_int('max_depth', 2, 16)
    else:
        params['max_depth'] = None

    return params


def build(params, seed):
    return RandomForestClassifier(
        n_estimators=params.get('n_estimators', 500),
        max_depth=params.get('max_depth'),
        min_samples_leaf=params.get('min_samples_leaf', 1),
        max_features=params.get('max_features', 'sqrt'),
        class_weight=params.get('class_weight'),
        n_jobs=config.N_JOBS,
        random_state=seed,
    )


def train(data, space=None, n_trials=None, n_seeds=None, **kw):
    def make_predictor(params, d, seed):
        return sklearn_predictor(build, params, d, seed)

    return evaluate(NAME, make_predictor, space or SPACE, data,
                    n_trials=n_trials, n_seeds=n_seeds, **kw)
