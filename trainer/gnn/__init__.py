"""Graph neural networks for early rug-pull detection on Solana.

Typical use is through:

    python run_gnn.py

For one architecture directly:

    from gnn.data import get_data
    from gnn.models import gcn

    data = get_data(
        'full',
        'q1',
        'original',
    )

    result = gcn.train(
        data,
        n_trials=10,
    )
"""

from . import config

from .data import (
    GraphRunData,
    NodeFeaturePreprocessor,
    get_data,
    load_graphs,
)

from .evaluation import (
    evaluate,
    summarise,
)

from .models import (
    ALL_MODELS,
    REGISTRY,
    get,
    is_available,
)

from .reporting import (
    load_summary,
)


__all__ = [
    'config',
    'GraphRunData',
    'NodeFeaturePreprocessor',
    'get_data',
    'load_graphs',
    'evaluate',
    'summarise',
    'REGISTRY',
    'ALL_MODELS',
    'get',
    'is_available',
    'load_summary',
]