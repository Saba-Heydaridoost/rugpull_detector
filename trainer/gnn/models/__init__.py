"""GNN model registry.

Each architecture exposes:

    NAME
    SPACE(trial)
    build(params)
    train(data, ...)

Model modules are imported lazily so the package itself can still be
inspected when optional PyTorch Geometric dependencies are unavailable.
"""

from importlib import import_module


MODEL_MODULES = {
    'gcn': '.gcn',
    'graphsage': '.graphsage',
    'gat': '.gat',
    'dirsage': '.dirsage',
    'dirgcn': '.dirgcn',
}

REGISTRY = {
    name: None
    for name in MODEL_MODULES
}

ALL_MODELS = list(
    MODEL_MODULES
)


def _validate_name(
    name,
):
    if name not in MODEL_MODULES:
        raise KeyError(
            f'unknown GNN model '
            f'{name!r}; '
            f'available: {ALL_MODELS}'
        )


def get(
    name,
):
    """Return one registered GNN model module."""

    _validate_name(
        name
    )

    if REGISTRY[name] is None:

        REGISTRY[name] = import_module(
            MODEL_MODULES[name],
            package=__package__,
        )

    return REGISTRY[name]


def is_available(
    name,
):
    """Whether the optional PyTorch Geometric dependency is available."""

    _validate_name(
        name
    )

    from ..torch_utils import (
        HAS_PYG,
    )

    return HAS_PYG