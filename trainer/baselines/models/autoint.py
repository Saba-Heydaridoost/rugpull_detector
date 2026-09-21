"""AutoInt.

Interacting layers: multi-head self-attention over the feature embeddings with a residual
projection, then a linear head over the flattened embeddings.

AutoInt was designed for high-cardinality categorical interaction modelling. With a small
set of purely numerical predictors it has little structure to exploit, so it is included as
a reference point rather than an expected winner.
"""

from ..evaluation import evaluate
from ..torch_utils import (
    HAS_TORCH,
    require_torch,
    torch_train_kwargs,
    train_torch_model,
)

NAME = 'autoint'

GRID = [
    {'d_token': d, 'n_layers': n, 'n_heads': 4, 'dropout': dr,
     'lr': lr, 'weight_decay': 1e-5, 'batch_size': 128}
    for d in [16, 32, 64]
    for n in [1, 2, 3]
    for dr in [0.1, 0.3]
    for lr in [1e-3, 3e-4]
]

if HAS_TORCH:
    import torch
    import torch.nn as nn

    from ..torch_utils import NumericalTokenizer

    class AutoIntNet(nn.Module):
        def __init__(self, n_features, d_token=32, n_layers=3, n_heads=4, dropout=0.1):
            super().__init__()

            self.tokenizer = NumericalTokenizer(n_features, d_token)

            self.attn = nn.ModuleList([
                nn.MultiheadAttention(d_token, n_heads, dropout=dropout, batch_first=True)
                for _ in range(n_layers)
            ])
            self.res = nn.ModuleList([
                nn.Linear(d_token, d_token, bias=False) for _ in range(n_layers)
            ])

            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(n_features * d_token, 1)

        def forward(self, x):
            t = self.tokenizer(x)

            for attn, res in zip(self.attn, self.res):
                a, _ = attn(t, t, t, need_weights=False)
                t = torch.relu(self.drop(a) + res(t))

            return self.head(t.flatten(1)).squeeze(-1)


def build(params):
    require_torch(NAME)

    return lambda n_features: AutoIntNet(
        n_features,
        params.get('d_token', 32),
        params.get('n_layers', 3),
        params.get('n_heads', 4),
        params.get('dropout', 0.1),
    )


def train(data, grid=None, n_seeds=None, **kw):
    require_torch(NAME)

    def make_predictor(params, d, seed):
        return train_torch_model(build(params), d, seed, **torch_train_kwargs(params))

    return evaluate(NAME, make_predictor, grid or GRID, data, n_seeds=n_seeds, **kw)
