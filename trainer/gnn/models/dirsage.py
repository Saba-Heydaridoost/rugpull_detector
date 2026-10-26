"""Direction-aware GraphSAGE graph classifier (Dir-GNN style) with jumping knowledge.

Motivation (EDA): early rug-pull transfer graphs are star-shaped, with a few distributor
wallets sending to many receivers. The standard GraphSAGE/GCN layers pass messages only
along edge direction (sender -> receiver), so a distributor never aggregates information
from its receivers. Following Dir-GNN (Rossi et al., 2023), each layer aggregates
in-neighbours and out-neighbours with separate weights:

    h_i' = LayerNorm( W_self h_i + W_in mean_{j->i} h_j + W_out mean_{i->j} h_j ) + h_i

Node embeddings of all layers are concatenated (jumping knowledge, Xu et al., 2018) and
read out with mean || max pooling, then fused with the token-level predictors exactly as
in the other models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_max_pool, global_mean_pool

from ..evaluation import evaluate
from ..torch_utils import require_pyg, torch_train_kwargs, train_gnn_model
from .common import fuse_token_features, make_mlp_head

NAME = 'dirsage'
N_TRIALS = 40


def SPACE(trial):
    """Same search space as GraphSAGE, except pooling (fixed to mean || max)."""
    return {
        'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 64, 128]),
        'n_layers': trial.suggest_int('n_layers', 1, 3),
        'dropout': trial.suggest_float('dropout', 0.0, 0.5),
        'lr': trial.suggest_float('lr', 1e-4, 3e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [4, 8, 16]),
        'class_weighting': trial.suggest_categorical('class_weighting', ['none', 'balanced']),
    }


class DirSAGELayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.inc = SAGEConv(in_dim, out_dim, aggr='mean', root_weight=True)
        self.out = SAGEConv(in_dim, out_dim, aggr='mean', root_weight=False)
        self.norm = nn.LayerNorm(out_dim)
        self.skip = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, edge_index):
        h = self.inc(x, edge_index) + self.out(x, edge_index.flip(0))
        return F.relu(self.norm(h)) + self.skip(x)


class DirSAGENet(nn.Module):
    def __init__(self, n_node_features, n_token_features, hidden_dim=64, n_layers=2, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        dims = [n_node_features] + [hidden_dim] * n_layers
        self.layers = nn.ModuleList(DirSAGELayer(a, b) for a, b in zip(dims[:-1], dims[1:]))
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
            DirSAGENet(n_node_features=n_node_features, n_token_features=n_token_features,
                       hidden_dim=params.get('hidden_dim', 64), n_layers=params.get('n_layers', 2),
                       dropout=params.get('dropout', 0.2)))


def train(data, space=None, n_trials=None, n_seeds=None, **kw):
    require_pyg(NAME)

    def make_predictor(params, d, seed, **train_kw):
        return train_gnn_model(build(params), d, seed, **torch_train_kwargs(params), **train_kw)

    return evaluate(NAME, make_predictor, space or SPACE, data,
                    n_trials=N_TRIALS if n_trials is None else n_trials, n_seeds=n_seeds, **kw)
