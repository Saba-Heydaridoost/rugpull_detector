"""Prior-probability reference. Establishes the floor every baseline must clear."""

from sklearn.dummy import DummyClassifier

from ..evaluation import evaluate, sklearn_predictor

NAME = 'dummy'

# Nothing to tune: the prior-probability rule has no hyperparameters. One trial is spent so
# the study still produces the trial table the reporting layer expects.
N_TRIALS = 1


def SPACE(trial):
    return {'strategy': 'prior'}


def build(params, seed):
    return DummyClassifier(strategy=params.get('strategy', 'prior'))


def train(data, space=None, n_trials=None, n_seeds=1, **kw):
    def make_predictor(params, d, seed):
        return sklearn_predictor(build, params, d, seed)

    return evaluate(NAME, make_predictor, space or SPACE, data,
                    n_trials=n_trials or N_TRIALS, n_seeds=n_seeds, **kw)
