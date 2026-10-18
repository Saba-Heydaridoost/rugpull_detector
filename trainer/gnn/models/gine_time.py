"""Exploratory edge-time-aware graph classifier, separate from primary GraphSAGE.

Uses GINEConv with the observed, cutoff-filtered transfer edge t_frac.
Unlike SAGEConv, GINEConv consumes edge_attr in each message.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv

from ..evaluation import evaluate
from ..torch_utils import require_pyg, torch_train_kwargs, train_gnn_model
from .common import fuse_token_features, graph_pool, make_mlp_head
from .graphsage import SPACE

NAME = 'gine_time'
N_TRIALS = 40


class GINETimeNet(nn.Module):
    def __init__(self, n_node_features, n_edge_features, n_token_features,
                 hidden_dim=64, n_layers=2, dropout=0.2, pooling='mean'):
        super().__init__()
        if n_edge_features < 1:
            raise ValueError('GINETimeNet requires edge_attr with at least one edge feature')
        if n_layers < 1:
            raise ValueError('n_layers must be positive')
        self.dropout = dropout
        self.pooling = pooling
        self.n_edge_features = n_edge_features
        self.node_projection = nn.Linear(n_node_features, hidden_dim)
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, edge_dim=n_edge_features))
        self.head = make_mlp_head(
            hidden_dim, dropout, input_dim=hidden_dim + n_token_features
        )

    def forward(self, batch):
        if not hasattr(batch, 'edge_attr') or batch.edge_attr is None:
            raise ValueError('Missing edge_attr for edge-time GNN')
        if batch.edge_attr.ndim != 2 or batch.edge_attr.shape[1] != self.n_edge_features:
            raise ValueError('Unexpected edge_attr shape')
        if batch.edge_attr.shape[0] != batch.edge_index.shape[1]:
            raise ValueError('Edge attributes are not aligned with edge_index')
        if not torch.isfinite(batch.edge_attr).all():
            raise ValueError('Non-finite edge attributes')
        x = F.relu(self.node_projection(batch.x))
        for conv in self.convs:
            x = conv(x, batch.edge_index, edge_attr=batch.edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        graph_x = graph_pool(x, batch.batch, self.pooling)
        return self.head(fuse_token_features(graph_x, batch)).squeeze(-1)


def build(params):
    require_pyg(NAME)
    return lambda n_node_features, n_edge_features, n_token_features: GINETimeNet(
        n_node_features=n_node_features,
        n_edge_features=n_edge_features,
        n_token_features=n_token_features,
        hidden_dim=params.get('hidden_dim', 64),
        n_layers=params.get('n_layers', 2),
        dropout=params.get('dropout', 0.2),
        pooling=params.get('pooling', 'mean'),
    )


def train(data, space=None, n_trials=None, n_seeds=None, **kw):
    require_pyg(NAME)
    def make_predictor(params, d, seed):
        return train_gnn_model(build(params), d, seed, **torch_train_kwargs(params))
    return evaluate(
        NAME, make_predictor, space or SPACE, data,
        n_trials=N_TRIALS if n_trials is None else n_trials,
        n_seeds=n_seeds, **kw,
    )
