"""Shared components for graph-level GNN classifiers."""

import torch

from torch_geometric.nn import (
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)


def graph_pool(
    x,
    batch,
    mode,
):
    """Pool node embeddings into one embedding per graph."""

    if mode == 'mean':

        return global_mean_pool(
            x,
            batch,
        )

    if mode == 'max':

        return global_max_pool(
            x,
            batch,
        )

    if mode == 'add':

        return global_add_pool(
            x,
            batch,
        )

    raise ValueError(
        f'unknown pooling mode '
        f'{mode!r}'
    )


def make_mlp_head(
    hidden_dim,
    dropout,
):
    """Small graph-level classification head."""

    return torch.nn.Sequential(
        torch.nn.Linear(
            hidden_dim,
            hidden_dim,
        ),
        torch.nn.ReLU(),
        torch.nn.Dropout(
            dropout
        ),
        torch.nn.Linear(
            hidden_dim,
            1,
        ),
    )