"""GraphSAGE graph classifier."""

import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv

from ..evaluation import evaluate

from ..torch_utils import (
    require_pyg,
    torch_train_kwargs,
    train_gnn_model,
)

from .common import (
    fuse_token_features,
    graph_pool,
    make_mlp_head,
)


NAME = 'graphsage'

N_TRIALS = 40


def SPACE(
    trial,
):
    """Optuna search space for GraphSAGE."""

    return {
        'hidden_dim':
            trial.suggest_categorical(
                'hidden_dim',
                [
                    32,
                    64,
                    128,
                ],
            ),

        'n_layers':
            trial.suggest_int(
                'n_layers',
                1,
                3,
            ),

        'dropout':
            trial.suggest_float(
                'dropout',
                0.0,
                0.5,
            ),

        'pooling':
            trial.suggest_categorical(
                'pooling',
                [
                    'mean',
                    'max',
                    'add',
                ],
            ),

        'lr':
            trial.suggest_float(
                'lr',
                1e-4,
                3e-3,
                log=True,
            ),

        'weight_decay':
            trial.suggest_float(
                'weight_decay',
                1e-6,
                1e-2,
                log=True,
            ),

        'batch_size':
            trial.suggest_categorical(
                'batch_size',
                [
                    4,
                    8,
                    16,
                ],
            ),
    }


class GraphSAGENet(
    nn.Module,
):
    """Stacked GraphSAGE layers with graph-level pooling."""

    def __init__(
        self,
        n_node_features,
        n_token_features,
        hidden_dim=64,
        n_layers=2,
        dropout=0.2,
        pooling='mean',
    ):
        super().__init__()

        self.dropout = dropout
        self.pooling = pooling

        self.convs = nn.ModuleList()

        self.convs.append(
            SAGEConv(
                n_node_features,
                hidden_dim,
            )
        )

        for _ in range(
            n_layers - 1
        ):

            self.convs.append(
                SAGEConv(
                    hidden_dim,
                    hidden_dim,
                )
            )

        self.head = make_mlp_head(
            hidden_dim,
            dropout,
            input_dim=(
                hidden_dim
                + n_token_features
            ),
        )

    def forward(
        self,
        batch,
    ):

        x = batch.x
        edge_index = batch.edge_index

        for conv in self.convs:

            x = conv(
                x,
                edge_index,
            )

            x = F.relu(
                x
            )

            x = F.dropout(
                x,
                p=self.dropout,
                training=self.training,
            )

        graph_x = graph_pool(
            x,
            batch.batch,
            self.pooling,
        )

        graph_x = fuse_token_features(
            graph_x,
            batch,
        )

        return (
            self.head(
                graph_x
            )
            .squeeze(-1)
        )


def build(
    params,
):

    require_pyg(
        NAME
    )

    return (
        lambda n_node_features,
            n_edge_features,
            n_token_features:
        GraphSAGENet(
            n_node_features=n_node_features,
            n_token_features=n_token_features,

            hidden_dim=params.get(
                'hidden_dim',
                64,
            ),

            n_layers=params.get(
                'n_layers',
                2,
            ),

            dropout=params.get(
                'dropout',
                0.2,
            ),

            pooling=params.get(
                'pooling',
                'mean',
            ),
        )
    )


def train(
    data,
    space=None,
    n_trials=None,
    n_seeds=None,
    **kw,
):

    require_pyg(
        NAME
    )

    def make_predictor(
        params,
        d,
        seed,
    ):

        return train_gnn_model(
            build(params),
            d,
            seed,
            **torch_train_kwargs(
                params
            ),
        )

    return evaluate(
        NAME,
        make_predictor,
        space or SPACE,
        data,
        n_trials=(
            N_TRIALS
            if n_trials is None
            else n_trials
        ),
        n_seeds=n_seeds,
        **kw,
    )