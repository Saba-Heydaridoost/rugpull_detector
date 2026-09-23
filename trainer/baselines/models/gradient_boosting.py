"""Scikit-learn gradient boosting.

Note that this estimator cannot consume NaN directly; the shared preprocessor already
imputes, so the design matrix it receives is dense.
"""

from sklearn.ensemble import GradientBoostingClassifier

from ..evaluation import evaluate, sklearn_predictor

NAME = 'gradient_boosting'


def SPACE(trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 600, step=50),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 2, 6),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 40),
    }


def build(params, seed):
    return GradientBoostingClassifier(
        n_estimators=params.get('n_estimators', 300),
        learning_rate=params.get('learning_rate', 0.05),
        max_depth=params.get('max_depth', 3),
        subsample=params.get('subsample', 1.0),
        min_samples_leaf=params.get('min_samples_leaf', 1),
        random_state=seed,
    )


def train(data, space=None, n_trials=None, n_seeds=None, **kw):
    def make_predictor(params, d, seed):
        return sklearn_predictor(build, params, d, seed)

    return evaluate(NAME, make_predictor, space or SPACE, data,
                    n_trials=n_trials, n_seeds=n_seeds, **kw)
