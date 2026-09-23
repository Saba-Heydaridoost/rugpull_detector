"""Tabular baselines for early rug-pull detection on Solana.

Typical use is through the driver script:

    python run_baselines.py

For a single model, import directly:

    from baselines.data import get_data
    from baselines.models import random_forest

    data = get_data('full', 'q1', 'node', variant='with_mask')
    result = random_forest.train(data, n_trials=30)

`train` tunes with Optuna and then re-runs the winner over several seeds. Pass `space=` a
`SPACE(trial)` callable to search something other than the model's own space.
"""

from . import config
from .data import Preprocessor, RunData, get_data, load_view, prepare
from .evaluation import evaluate, summarise
from .models import ALL_MODELS, REGISTRY, get, is_available
from .reporting import load_summary

__all__ = [
    'config',
    'load_view', 'prepare', 'get_data', 'Preprocessor', 'RunData',
    'evaluate', 'summarise',
    'REGISTRY', 'ALL_MODELS', 'get', 'is_available',
    'load_summary',
]
