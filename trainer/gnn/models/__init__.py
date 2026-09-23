"""GNN model registry.

Each architecture exposes:

    NAME
    SPACE(trial)
    build(params)
    train(data, ...)
"""

from . import (
    gat,
    gcn,
    graphsage,
)


REGISTRY = {
    gcn.NAME:
        gcn,

    graphsage.NAME:
        graphsage,

    gat.NAME:
        gat,
}


ALL_MODELS = list(
    REGISTRY
)


def get(
    name,
):
    """Return one registered GNN model module."""

    if name not in REGISTRY:

        raise KeyError(
            f'unknown GNN model '
            f'{name!r}; '
            f'available: {ALL_MODELS}'
        )

    return REGISTRY[name]


def is_available(
    name,
):
    """Check whether PyTorch Geometric is installed."""

    get(name)

    from ..torch_utils import (
        HAS_PYG,
    )

    return HAS_PYG