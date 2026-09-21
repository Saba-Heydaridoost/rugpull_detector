"""Prior-probability reference. Establishes the floor every baseline must clear."""

from sklearn.dummy import DummyClassifier

from ..evaluation import evaluate, sklearn_predictor

NAME = 'dummy'

GRID = [{'strategy': 'prior'}]


def build(params, seed):
    return DummyClassifier(strategy=params.get('strategy', 'prior'))


def train(data, grid=None, n_seeds=1, **kw):
    def make_predictor(params, d, seed):
        return sklearn_predictor(build, params, d, seed)

    return evaluate(NAME, make_predictor, grid or GRID, data, n_seeds=n_seeds, **kw)
