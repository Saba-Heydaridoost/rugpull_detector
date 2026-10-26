"""Graph Attention Network graph classifier."""

import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv

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


NAME = 'gat'

N_TRIALS = 40


HEAD_CHOICES = [
    2,
    4,
    8,
]


def SPACE(
    trial,
):
    """Optuna search space for GAT."""

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

        'heads':
            trial.suggest_categorical(
                'heads',
                HEAD_CHOICES,
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

        'class_weighting':
            trial.suggest_categorical(
                'class_weighting',
                [
                    'none',
                    'balanced',
                ],
            ),
    }


class GATNet(
    nn.Module,
):
    """Multi-head GAT with a fixed hidden width."""

    def __init__(
        self,
        n_node_features,
        n_token_features,
        hidden_dim=64,
        n_layers=2,
        heads=4,
        dropout=0.2,
        pooling='mean',
    ):
        super().__init__()

        if (
            hidden_dim
            % heads
            != 0
        ):

            raise ValueError(
                'hidden_dim must be '
                'divisible by heads'
            )

        self.dropout = dropout
        self.pooling = pooling

        per_head = (
            hidden_dim
            // heads
        )

        self.convs = nn.ModuleList()

        self.convs.append(
            GATConv(
                n_node_features,
                per_head,
                heads=heads,
                concat=True,
                dropout=dropout,
            )
        )

        for _ in range(
            n_layers - 1
        ):

            self.convs.append(
                GATConv(
                    hidden_dim,
                    per_head,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
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

            x = F.elu(
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
        GATNet(
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

            heads=params.get(
                'heads',
                4,
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
        **train_kw,
    ):

        return train_gnn_model(
            build(params),
            d,
            seed,
            **torch_train_kwargs(
                params
            ),
            **train_kw,
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