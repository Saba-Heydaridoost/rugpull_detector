"""Direction-aware GCN graph classifier (Dir-GCN, Rossi et al., 2023) with jumping knowledge.

Each layer combines a GCN over incoming edges and a GCN over outgoing (reversed) edges,

    h' = LayerNorm( alpha * GCN_in(h) + (1 - alpha) * GCN_out(h) ) + h,

with alpha tuned on validation. Jumping knowledge (concatenation of all layers) and
mean || max readout as in dirsage. See dirsage.py for the motivation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool

from ..evaluation import evaluate
from ..torch_utils import require_pyg, torch_train_kwargs, train_gnn_model
from .common import fuse_token_features, make_mlp_head

NAME = 'dirgcn'
N_TRIALS = 40


def SPACE(trial):
    """GCN search space without pooling (fixed mean || max), plus the direction weight alpha."""
    return {
        'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 64, 128]),
        'n_layers': trial.suggest_int('n_layers', 1, 3),
        'dropout': trial.suggest_float('dropout', 0.0, 0.5),
        'alpha': trial.suggest_categorical('alpha', [0.25, 0.5, 0.75]),
        'lr': trial.suggest_float('lr', 1e-4, 3e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [4, 8, 16]),
        'class_weighting': trial.suggest_categorical('class_weighting', ['none', 'balanced']),
    }


class DirGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, alpha):
        super().__init__()
        self.alpha = alpha
        self.inc = GCNConv(in_dim, out_dim)
        self.out = GCNConv(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.skip = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, edge_index):
        h = self.alpha * self.inc(x, edge_index) + (1 - self.alpha) * self.out(x, edge_index.flip(0))
        return F.relu(self.norm(h)) + self.skip(x)


class DirGCNNet(nn.Module):
    def __init__(self, n_node_features, n_token_features, hidden_dim=64, n_layers=2, dropout=0.2, alpha=0.5):
        super().__init__()
        self.dropout = dropout
        dims = [n_node_features] + [hidden_dim] * n_layers
        self.layers = nn.ModuleList(DirGCNLayer(a, b, alpha) for a, b in zip(dims[:-1], dims[1:]))
        jk_dim = hidden_dim * n_layers
        self.head = make_mlp_head(hidden_dim, dropout, input_dim=2 * jk_dim + n_token_features)

    def forward(self, batch):
        x, edge_index, b = batch.x, batch.edge_index, batch.batch
        layers = []
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layers.append(x)
        h = torch.cat(layers, dim=-1)
        graph_x = torch.cat([global_mean_pool(h, b), global_max_pool(h, b)], dim=-1)
        return self.head(fuse_token_features(graph_x, batch)).squeeze(-1)


def build(params):
    require_pyg(NAME)
    return (lambda n_node_features, n_edge_features, n_token_features:
            DirGCNNet(n_node_features=n_node_features, n_token_features=n_token_features,
                      hidden_dim=params.get('hidden_dim', 64), n_layers=params.get('n_layers', 2),
                      dropout=params.get('dropout', 0.2), alpha=params.get('alpha', 0.5)))


def train(data, space=None, n_trials=None, n_seeds=None, **kw):
    require_pyg(NAME)

    def make_predictor(params, d, seed, **train_kw):
        return train_gnn_model(build(params), d, seed, **torch_train_kwargs(params), **train_kw)

    return evaluate(NAME, make_predictor, space or SPACE, data,
                    n_trials=N_TRIALS if n_trials is None else n_trials, n_seeds=n_seeds, **kw)
