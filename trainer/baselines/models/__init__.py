"""Baseline registry.

Each model lives in its own module exposing `NAME`, `SPACE`, `build` and `train`. To add a
baseline, write the module and add it to `REGISTRY` below — nothing else needs to change.

`SPACE(trial)` is the Optuna search space: it draws from `trial` and returns the parameter
dict `build` consumes. A module may also set `N_TRIALS` to override `config.N_TRIALS` for
itself, which the torch baselines do because each of their trials trains a network.
"""

from . import (
    autoint,
    decision_tree,
    dummy,
    ft_transformer,
    gradient_boosting,
    logistic_regression,
    mlp,
    random_forest,
    xgboost_model,
)

REGISTRY = {
    dummy.NAME: dummy,
    logistic_regression.NAME: logistic_regression,
    decision_tree.NAME: decision_tree,
    random_forest.NAME: random_forest,
    gradient_boosting.NAME: gradient_boosting,
    xgboost_model.NAME: xgboost_model,
    mlp.NAME: mlp,
    ft_transformer.NAME: ft_transformer,
    autoint.NAME: autoint,
}

ALL_MODELS = list(REGISTRY)


def get(name):
    if name not in REGISTRY:
        raise KeyError(f'unknown model {name!r}; available: {ALL_MODELS}')

    return REGISTRY[name]


def is_available(name):
    """Whether the model's optional dependency is installed."""
    module = get(name)

    if name == 'xgboost':
        return module.HAS_XGB

    if name in ('mlp', 'ft_transformer', 'autoint'):
        from ..torch_utils import HAS_TORCH
        return HAS_TORCH

    return True
