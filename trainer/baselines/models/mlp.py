"""Fully connected network over the transformed design matrix."""

from ..evaluation import evaluate
from ..torch_utils import HAS_TORCH, require_torch, torch_train_kwargs, train_torch_model

NAME = 'mlp'

# The torch baselines are the expensive ones, so they get a smaller budget than the
# default. Each trial trains a network to early stopping.
N_TRIALS = 40


def SPACE(trial):
    """Depth and width are drawn separately and composed into a halving stack.

    The grid enumerated four hand-written shapes; drawing (depth, width) reaches the same
    family — (64,), (128, 64), (256, 128, 64) — plus the widths between them, without
    needing a categorical over tuples.
    """
    n_layers = trial.suggest_int('n_layers', 1, 3)
    width = trial.suggest_categorical('width', [64, 128, 256])

    return {
        'hidden': tuple(max(16, width // 2 ** i) for i in range(n_layers)),
        'dropout': trial.suggest_float('dropout', 0.0, 0.5),
        'lr': trial.suggest_float('lr', 1e-4, 3e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256]),
    }


if HAS_TORCH:
    import torch.nn as nn

    class MLPNet(nn.Module):
        def __init__(self, n_features, hidden=(128, 64), dropout=0.2):
            super().__init__()

            layers, d_in = [], n_features

            for h in hidden:
                layers += [nn.Linear(d_in, h), nn.ReLU(), nn.Dropout(dropout)]
                d_in = h

            layers.append(nn.Linear(d_in, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x).squeeze(-1)


def build(params):
    """Return a factory mapping n_features to an initialised network."""
    require_torch(NAME)

    return lambda n_features: MLPNet(
        n_features,
        params.get('hidden', (128, 64)),
        params.get('dropout', 0.2),
    )


def train(data, space=None, n_trials=None, n_seeds=None, **kw):
    require_torch(NAME)

    def make_predictor(params, d, seed):
        return train_torch_model(build(params), d, seed, **torch_train_kwargs(params))

    return evaluate(NAME, make_predictor, space or SPACE, data,
                    n_trials=n_trials or N_TRIALS, n_seeds=n_seeds, **kw)
