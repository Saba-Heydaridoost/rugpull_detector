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

N_TRIALS = 40

# Attention requires d_token to divide evenly by the head count, so the token width is drawn
# from multiples of N_HEADS rather than from a continuous range.
N_HEADS = 8


def SPACE(trial):
    return {
        'd_token': trial.suggest_categorical('d_token', [32, 64, 128]),
        'n_blocks': trial.suggest_int('n_blocks', 1, 3),
        'n_heads': N_HEADS,
        'dropout': trial.suggest_float('dropout', 0.0, 0.4),
        'lr': trial.suggest_float('lr', 1e-4, 3e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256]),
    }


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


def train(data, space=None, n_trials=None, n_seeds=None, **kw):
    require_torch(NAME)

    def make_predictor(params, d, seed):
        return train_torch_model(build(params), d, seed, **torch_train_kwargs(params))

    return evaluate(NAME, make_predictor, space or SPACE, data,
                    n_trials=n_trials or N_TRIALS, n_seeds=n_seeds, **kw)
