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


def fuse_token_features(
    graph_x,
    batch,
):
    """Concatenate graph embedding with graph-level token predictors."""

    token_x = batch.token_features

    if token_x.ndim == 1:
        token_x = token_x.unsqueeze(
            0
        )

    if (
        token_x.shape[0]
        != graph_x.shape[0]
    ):
        raise AssertionError(
            'one token-feature vector is required '
            'for every graph in the batch'
        )

    return torch.cat(
        [
            graph_x,
            token_x,
        ],
        dim=1,
    )


def make_mlp_head(
    hidden_dim,
    dropout,
    input_dim=None,
):
    """Small graph-level classification head."""

    input_dim = (
        hidden_dim
        if input_dim is None
        else input_dim
    )

    return torch.nn.Sequential(
        torch.nn.Linear(
            input_dim,
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