"""FT-Transformer.

Feature tokenizer, a prepended [CLS] token, and pre-norm transformer blocks. Since every
predictor here is numerical, the tokenizer reduces to one linear embedding per feature and
no categorical embedding table is required. Classification reads out the [CLS] position.
"""

from ..evaluation import evaluate
from ..torch_utils import (
    HAS_TORCH,
    require_torch,
    torch_train_kwargs,
    train_torch_model,
)

NAME = 'ft_transformer'

GRID = [
    {'d_token': d, 'n_blocks': b, 'n_heads': 8, 'dropout': dr,
     'lr': lr, 'weight_decay': 1e-5, 'batch_size': 128}
    for d in [32, 64]
    for b in [1, 2, 3]
    for dr in [0.1, 0.2, 0.3]
    for lr in [1e-3, 3e-4]
]

if HAS_TORCH:
    import torch
    import torch.nn as nn

    from ..torch_utils import NumericalTokenizer

    class FTTransformerNet(nn.Module):
        def __init__(self, n_features, d_token=64, n_blocks=2, n_heads=8, dropout=0.1):
            super().__init__()

            self.tokenizer = NumericalTokenizer(n_features, d_token)

            self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
            nn.init.normal_(self.cls, std=0.02)

            self.blocks = nn.ModuleList([
                nn.ModuleDict({
                    'norm1': nn.LayerNorm(d_token),
                    'attn': nn.MultiheadAttention(
                        d_token, n_heads, dropout=dropout, batch_first=True
                    ),
                    'norm2': nn.LayerNorm(d_token),
                    'ffn': nn.Sequential(
                        nn.Linear(d_token, 2 * d_token), nn.GELU(),
                        nn.Dropout(dropout), nn.Linear(2 * d_token, d_token),
                    ),
                    'drop': nn.Dropout(dropout),
                })
                for _ in range(n_blocks)
            ])

            self.head = nn.Sequential(
                nn.LayerNorm(d_token), nn.ReLU(), nn.Linear(d_token, 1)
            )

        def forward(self, x):
            t = self.tokenizer(x)
            t = torch.cat([self.cls.expand(t.size(0), -1, -1), t], dim=1)

            for b in self.blocks:
                h = b['norm1'](t)
                a, _ = b['attn'](h, h, h, need_weights=False)
                t = t + b['drop'](a)
                t = t + b['drop'](b['ffn'](b['norm2'](t)))

            return self.head(t[:, 0]).squeeze(-1)


def build(params):
    require_torch(NAME)

    return lambda n_features: FTTransformerNet(
        n_features,
        params.get('d_token', 64),
        params.get('n_blocks', 2),
        params.get('n_heads', 8),
        params.get('dropout', 0.1),
    )


def train(data, grid=None, n_seeds=None, **kw):
    require_torch(NAME)

    def make_predictor(params, d, seed):
        return train_torch_model(build(params), d, seed, **torch_train_kwargs(params))

    return evaluate(NAME, make_predictor, grid or GRID, data, n_seeds=n_seeds, **kw)
