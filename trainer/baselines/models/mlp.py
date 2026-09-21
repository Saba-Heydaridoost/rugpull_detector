"""Fully connected network over the transformed design matrix."""

from ..evaluation import evaluate
from ..torch_utils import HAS_TORCH, require_torch, torch_train_kwargs, train_torch_model

NAME = 'mlp'

GRID = [
    {'hidden': h, 'dropout': dr, 'lr': lr, 'weight_decay': wd, 'batch_size': 128}
    for h in [(64,), (128, 64), (256, 128), (256, 128, 64)]
    for dr in [0.1, 0.3, 0.5]
    for lr in [1e-3, 3e-4]
    for wd in [1e-5, 1e-3]
]

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


def train(data, grid=None, n_seeds=None, **kw):
    require_torch(NAME)

    def make_predictor(params, d, seed):
        return train_torch_model(build(params), d, seed, **torch_train_kwargs(params))

    return evaluate(NAME, make_predictor, grid or GRID, data, n_seeds=n_seeds, **kw)
