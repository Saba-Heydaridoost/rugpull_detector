"""Loading recovered transfer graphs and building GNN-ready samples.

One token mint is one graph-level classification sample.

Nodes
-----
Recovered transfer endpoints:
    - resolved owner address when available upstream;
    - otherwise the underlying token-account address.

Edges
-----
Each retained recovered transfer row becomes one directed edge from
`source_address` to `target_address`.

Parallel edges are preserved. This is intentional: repeated recovered
transfer rows between the same ordered endpoint pair are distinct
observations in the exported graph.

Primary features
----------------
The primary GNN experiment uses structural and temporal information.

Canonical normalized transfer amounts are available in the frozen EDA
edge artifacts, but absolute token amounts are intentionally excluded
from the primary GNN representation because decimal normalization alone
does not make transfer quantities economically comparable across token
mints.

Any future amount-aware experiment must be defined as a separate model
variant rather than silently changing the primary representation.

Node features are structural/temporal quantities derived only from the
current token's recovered transfers inside its observation window.

The stored edge attribute currently contains only `t_frac`.

Preprocessing
-------------
Node-feature transforms are fitted on nodes from TRAIN graphs only and
then applied unchanged to validation and test graphs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

from . import config


# =============================================================================
# Optional PyTorch / PyG dependency
# =============================================================================

try:
    import torch
    from torch_geometric.data import Data

    HAS_PYG = True

except ImportError:
    torch = None
    Data = None
    HAS_PYG = False


def require_pyg():
    """Raise a clear error if PyTorch Geometric is unavailable."""
    if not HAS_PYG:
        raise ImportError(
            'The GNN pipeline requires both torch and torch_geometric.'
        )


# =============================================================================
# Artifact loading helpers
# =============================================================================

def _check_artifact(path):
    """Fail with the exact missing artifact path."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} not found. '
            'Run EDA.ipynb to export the final graph artifacts.'
        )


def _normalise_split(frame):
    """Normalise exported split labels to the trainer convention."""
    return frame.assign(
        **{
            config.SPLIT_COL:
                frame[config.SPLIT_COL]
                .replace(config.SPLIT_ALIASES)
        }
    )


def cohort_folder(
    scope='full',
    q='q1',
    cohort='original',
):
    """Return the folder for one missingness cohort and observation window."""

    if scope not in config.SCOPES:
        raise ValueError(
            f'unknown scope {scope!r}; '
            f'expected one of {config.SCOPES}'
        )

    if q not in config.EXPERIMENTS:
        raise ValueError(
            f'unknown experiment {q!r}; '
            f'expected one of {config.EXPERIMENTS}'
        )

    if cohort not in config.COHORTS:
        raise ValueError(
            f'unknown cohort {cohort!r}; '
            f'expected one of {config.COHORTS}'
        )

    return os.path.join(
        config.COHORT_DIR,
        cohort,
        q,
    )


def _check_columns(frame, required, source):
    """Raise instead of silently dropping a required field."""
    missing = [
        col
        for col in required
        if col not in frame.columns
    ]

    if missing:
        raise KeyError(
            f'{source} does not contain required columns {missing}. '
            f'Available columns: {list(frame.columns)}'
        )


# =============================================================================
# Node features
# =============================================================================

def _node_feature_frame(edges, node_order):
    """Build deterministic endpoint-level structural/temporal features.

    Features use only recovered transfer rows belonging to the current
    token and observation window.

    Counts are log1p transformed before the dataset-level train-only
    quantile transformation. This prevents very high-degree endpoints
    from dominating the numerical scale.
    """

    # -----------------------------------------------------------------
    # Transfer multiplicity
    # -----------------------------------------------------------------

    out_transfers = (
        edges
        .groupby(config.SOURCE_COL)
        .size()
    )

    in_transfers = (
        edges
        .groupby(config.TARGET_COL)
        .size()
    )

    # -----------------------------------------------------------------
    # Distinct directed neighbours
    # -----------------------------------------------------------------

    out_neighbors = (
        edges
        .groupby(config.SOURCE_COL)[config.TARGET_COL]
        .nunique()
    )

    in_neighbors = (
        edges
        .groupby(config.TARGET_COL)[config.SOURCE_COL]
        .nunique()
    )

    # -----------------------------------------------------------------
    # Incident-edge timing
    # -----------------------------------------------------------------

    source_times = (
        edges[
            [
                config.SOURCE_COL,
                config.EDGE_TIME_COL,
            ]
        ]
        .rename(
            columns={
                config.SOURCE_COL: 'endpoint'
            }
        )
    )

    target_times = (
        edges[
            [
                config.TARGET_COL,
                config.EDGE_TIME_COL,
            ]
        ]
        .rename(
            columns={
                config.TARGET_COL: 'endpoint'
            }
        )
    )

    incident_times = pd.concat(
        [
            source_times,
            target_times,
        ],
        ignore_index=True,
    )

    timing = (
        incident_times
        .groupby('endpoint')[config.EDGE_TIME_COL]
        .agg(
            first_t_frac='min',
            mean_t_frac='mean',
        )
    )

    # -----------------------------------------------------------------
    # Assemble in the exact deterministic node order.
    # -----------------------------------------------------------------

    out = pd.DataFrame(
        index=pd.Index(
            node_order,
            name='endpoint',
        )
    )

    out['log1p_out_transfers'] = np.log1p(
        out.index
        .to_series()
        .map(out_transfers)
        .fillna(0)
        .astype(float)
    ).to_numpy()

    out['log1p_in_transfers'] = np.log1p(
        out.index
        .to_series()
        .map(in_transfers)
        .fillna(0)
        .astype(float)
    ).to_numpy()

    out['log1p_out_neighbors'] = np.log1p(
        out.index
        .to_series()
        .map(out_neighbors)
        .fillna(0)
        .astype(float)
    ).to_numpy()

    out['log1p_in_neighbors'] = np.log1p(
        out.index
        .to_series()
        .map(in_neighbors)
        .fillna(0)
        .astype(float)
    ).to_numpy()

    out['first_t_frac'] = (
        out.index
        .to_series()
        .map(timing['first_t_frac'])
        .astype(float)
        .to_numpy()
    )

    out['mean_t_frac'] = (
        out.index
        .to_series()
        .map(timing['mean_t_frac'])
        .astype(float)
        .to_numpy()
    )

    # Every endpoint came from at least one retained edge, so timing
    # features must exist for every row.
    if out[config.NODE_FEATURES].isna().any().any():
        raise AssertionError(
            'node feature construction produced unexpected NaNs'
        )

    return out[config.NODE_FEATURES]


# =============================================================================
# One token -> one PyG graph
# =============================================================================

def _build_graph(edges, sample_id, label):
    """Convert one token's retained recovered edges into a PyG Data object."""

    require_pyg()

    if edges.empty:
        raise ValueError(
            'cannot construct a GNN sample from an empty edge table'
        )

    # Stable ordering ensures the same artifact always creates the same
    # local node IDs and edge ordering.
    sort_cols = [
        col
        for col in [
            'timestamp',
            'tx_signature',
            config.SOURCE_COL,
            config.TARGET_COL,
        ]
        if col in edges.columns
    ]

    if sort_cols:
        edges = (
            edges
            .sort_values(
                sort_cols,
                kind='mergesort',
            )
            .reset_index(drop=True)
        )
    else:
        edges = edges.reset_index(drop=True)

    # Deterministic local node IDs.
    endpoints = pd.concat(
        [
            edges[config.SOURCE_COL],
            edges[config.TARGET_COL],
        ],
        ignore_index=True,
    )

    node_order = sorted(
        endpoints
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if len(node_order) < 2:
        raise AssertionError(
            'a recovered graph must contain at least two endpoints '
            'after self-loop removal'
        )

    node_to_index = {
        endpoint: i
        for i, endpoint in enumerate(node_order)
    }

    source_index = (
        edges[config.SOURCE_COL]
        .astype(str)
        .map(node_to_index)
        .to_numpy(dtype=np.int64)
    )

    target_index = (
        edges[config.TARGET_COL]
        .astype(str)
        .map(node_to_index)
        .to_numpy(dtype=np.int64)
    )

    edge_index = np.vstack(
        [
            source_index,
            target_index,
        ]
    )

    # Structural + temporal endpoint features.
    node_features = _node_feature_frame(
        edges,
        node_order,
    ).to_numpy(dtype=np.float32)

    # Keep temporal edge information.
    edge_attr = (
        edges[
            config.EDGE_FEATURES
        ]
        .to_numpy(dtype=np.float32)
    )

    graph = Data(
        x=torch.tensor(
            node_features,
            dtype=torch.float32,
        ),
        edge_index=torch.tensor(
            edge_index,
            dtype=torch.long,
        ),
        edge_attr=torch.tensor(
            edge_attr,
            dtype=torch.float32,
        ),
        y=torch.tensor(
            [float(label)],
            dtype=torch.float32,
        ),
        sample_id=torch.tensor(
            [int(sample_id)],
            dtype=torch.long,
        ),
    )

    # Explicit num_nodes avoids any ambiguity in PyG inference.
    graph.num_nodes = len(node_order)

    return graph


# =============================================================================
# Loaded corpus
# =============================================================================

@dataclass
class GraphCorpus:
    """Raw graph samples plus graph-level metadata."""

    graphs: list
    metadata: pd.DataFrame
    scope: str
    experiment: str
    cohort: str

    def graphs_for_split(self, split):
        ids = (
            self.metadata.loc[
                self.metadata[config.SPLIT_COL] == split,
                'sample_id',
            ]
            .astype(int)
            .tolist()
        )

        return [
            self.graphs[i]
            for i in ids
        ]


def load_graphs(
    scope='full',
    q='q1',
    cohort='original',
):
    """Load one GNN dataset before train-only numerical preprocessing."""

    require_pyg()

    folder = cohort_folder(
        scope=scope,
        q=q,
        cohort=cohort,
    )

    meta_path = os.path.join(
        folder,
        config.GRAPH_META_FILE,
    )

    edge_path = os.path.join(
        folder,
        config.EDGE_FILE,
    )

    _check_artifact(meta_path)
    _check_artifact(edge_path)

    meta_columns = [
        config.LP_COL,
        config.TOKEN_COL,
        config.CREATION_TIME_COL,
        config.LABEL_COL,
        config.SPLIT_COL,
    ]

    edge_columns = [
        config.TOKEN_COL,
        config.SOURCE_COL,
        config.TARGET_COL,
        config.EDGE_TIME_COL,
        config.SPLIT_COL,
        'timestamp',
        'tx_signature',
    ]

    meta = _normalise_split(
        pd.read_parquet(
            meta_path,
            columns=meta_columns,
        )
    )

    edges = _normalise_split(
        pd.read_parquet(
            edge_path,
            columns=edge_columns,
        )
    )

    # -----------------------------------------------------------------
    # Required schema
    # -----------------------------------------------------------------

    required_meta = [
        config.LP_COL,
        config.TOKEN_COL,
        config.CREATION_TIME_COL,
        config.LABEL_COL,
        config.SPLIT_COL,
    ]

    required_edges = [
        config.TOKEN_COL,
        config.SOURCE_COL,
        config.TARGET_COL,
        config.EDGE_TIME_COL,
        config.SPLIT_COL,
    ]

    _check_columns(
        meta,
        required_meta,
        meta_path,
    )

    _check_columns(
        edges,
        required_edges,
        edge_path,
    )

    # -----------------------------------------------------------------
    # Restrict to the official modelling partitions.
    # -----------------------------------------------------------------

    meta = (
        meta[
            meta[config.SPLIT_COL]
            .isin(config.VALID_SPLITS)
        ]
        .copy()
        .sort_values(
            [
                config.CREATION_TIME_COL,
                config.TOKEN_COL,
            ],
            kind='mergesort',
        )
        .reset_index(drop=True)
    )

    edges = (
        edges[
            edges[config.SPLIT_COL]
            .isin(config.VALID_SPLITS)
        ]
        .copy()
        .reset_index(drop=True)
    )

    # -----------------------------------------------------------------
    # Integrity checks
    # -----------------------------------------------------------------

    if meta[config.TOKEN_COL].duplicated().any():
        raise AssertionError(
            'graph metadata contains duplicate token_mint rows'
        )

    if not meta[config.LABEL_COL].isin([0, 1]).all():
        raise AssertionError(
            'graph metadata contains non-binary labels'
        )

    if edges[
        [
            config.SOURCE_COL,
            config.TARGET_COL,
        ]
    ].isna().any().any():
        raise AssertionError(
            'edge endpoints contain missing values'
        )

    if not np.isfinite(
        edges[config.EDGE_TIME_COL].to_numpy(dtype=float)
    ).all():
        raise AssertionError(
            'edge t_frac contains non-finite values'
        )

    if not (
        (
            edges[config.EDGE_TIME_COL] >= 0
        )
        &
        (
            edges[config.EDGE_TIME_COL] < 1
        )
    ).all():
        raise AssertionError(
            'edge t_frac must satisfy 0 <= t_frac < 1'
        )

    if (
        edges[config.SOURCE_COL]
        == edges[config.TARGET_COL]
    ).any():
        raise AssertionError(
            'self-loop recovered edges should already have been '
            'removed by EDA'
        )

    meta_tokens = set(
        meta[config.TOKEN_COL]
    )

    edge_tokens = set(
        edges[config.TOKEN_COL]
    )

    # The final cohort export is designed so every graph metadata row
    # has retained recovered edges and vice versa.
    if meta_tokens != edge_tokens:
        only_meta = meta_tokens - edge_tokens
        only_edges = edge_tokens - meta_tokens

        raise AssertionError(
            'graph metadata / edge token mismatch: '
            f'{len(only_meta)} token(s) only in metadata, '
            f'{len(only_edges)} token(s) only in edges'
        )

    # Each token must belong to one and only one split.
    edge_split_counts = (
        edges
        .groupby(config.TOKEN_COL)[config.SPLIT_COL]
        .nunique()
    )

    if (
        edge_split_counts > 1
    ).any():
        raise AssertionError(
            'at least one token has edge rows in multiple splits'
        )

    edge_split = (
        edges
        .groupby(config.TOKEN_COL)[config.SPLIT_COL]
        .first()
    )

    meta_split = (
        meta
        .set_index(config.TOKEN_COL)[config.SPLIT_COL]
    )

    aligned_edge_split = (
        edge_split
        .reindex(meta_split.index)
    )

    if not (
        aligned_edge_split.to_numpy()
        == meta_split.to_numpy()
    ).all():
        raise AssertionError(
            'edge and graph metadata split assignments disagree'
        )

    # -----------------------------------------------------------------
    # Build one graph per token.
    # -----------------------------------------------------------------

    edge_groups = {
        token: group.copy()
        for token, group in edges.groupby(
            config.TOKEN_COL,
            sort=False,
        )
    }

    graphs = []
    metadata_rows = []

    for sample_id, row in meta.iterrows():

        token = row[config.TOKEN_COL]

        token_edges = edge_groups[token]

        graph = _build_graph(
            token_edges,
            sample_id=sample_id,
            label=int(row[config.LABEL_COL]),
        )

        graphs.append(graph)

        metadata_rows.append(
            {
                'sample_id': int(sample_id),
                config.LP_COL:
                    row[config.LP_COL],
                config.TOKEN_COL:
                    token,
                config.CREATION_TIME_COL:
                    row[config.CREATION_TIME_COL],
                config.LABEL_COL:
                    int(row[config.LABEL_COL]),
                config.SPLIT_COL:
                    row[config.SPLIT_COL],
                'n_nodes':
                    int(graph.num_nodes),
                'n_edges':
                    int(graph.edge_index.shape[1]),
            }
        )

    metadata = pd.DataFrame(
        metadata_rows
    )

    if len(graphs) != len(meta):
        raise AssertionError(
            'graph count does not equal graph metadata row count'
        )

    if not (
        metadata['sample_id'].to_numpy()
        == np.arange(len(metadata))
    ).all():
        raise AssertionError(
            'sample IDs are not contiguous'
        )

    return GraphCorpus(
        graphs=graphs,
        metadata=metadata,
        scope=scope,
        experiment=q,
        cohort=cohort,
    )


# =============================================================================
# Train-only node-feature preprocessing
# =============================================================================

class NodeFeaturePreprocessor:
    """Train-fitted numerical transform for endpoint node features."""

    def __init__(
        self,
        transform=None,
    ):
        self.transform_name = (
            transform
            or config.TRANSFORM
        )

    def fit(self, graphs):
        """Fit using nodes from TRAIN graphs only."""

        if not graphs:
            raise ValueError(
                'cannot fit node preprocessor on an empty graph list'
            )

        X = np.vstack(
            [
                graph.x.cpu().numpy()
                for graph in graphs
            ]
        ).astype(np.float64)

        if X.shape[1] != len(config.NODE_FEATURES):
            raise AssertionError(
                'unexpected raw node-feature dimension'
            )

        if not np.isfinite(X).all():
            raise AssertionError(
                'raw training node features contain non-finite values'
            )

        if self.transform_name == 'quantile':

            self.scaler_ = QuantileTransformer(
                n_quantiles=min(
                    1000,
                    len(X),
                ),
                output_distribution='normal',
                subsample=10 ** 9,
                random_state=0,
            )

            self.scaler_.fit(X)

        elif self.transform_name == 'standard':

            self.mu_ = X.mean(axis=0)

            sd = X.std(axis=0)

            self.sd_ = np.where(
                sd < 1e-12,
                1.0,
                sd,
            )

        else:
            raise ValueError(
                f'unknown transform {self.transform_name!r}'
            )

        self.n_features_in_ = X.shape[1]

        return self

    def transform_graph(self, graph):
        """Return a cloned graph with transformed node features."""

        X = (
            graph.x
            .cpu()
            .numpy()
            .astype(np.float64)
        )

        if X.shape[1] != self.n_features_in_:
            raise AssertionError(
                'node feature dimension changed after fitting'
            )

        if not np.isfinite(X).all():
            raise AssertionError(
                'raw node features contain non-finite values '
                'before preprocessing'
            )

        if self.transform_name == 'quantile':
            Xt = self.scaler_.transform(X)

        else:
            Xt = (
                X - self.mu_
            ) / self.sd_

        if not np.isfinite(Xt).all():
            raise FloatingPointError(
                'node-feature preprocessing produced '
                'non-finite values'
            )

        out = graph.clone()

        out.x = torch.tensor(
            Xt,
            dtype=torch.float32,
        )

        return out

    def transform(self, graphs):
        return [
            self.transform_graph(graph)
            for graph in graphs
        ]


# =============================================================================
# Final run data
# =============================================================================

@dataclass
class GraphRunData:
    """Prepared graph datasets plus experiment context."""

    train_graphs: list
    val_graphs: list
    test_graphs: list

    node_feature_names: list
    edge_feature_names: list

    scope: str
    experiment: str
    cohort: str

    preprocessor: NodeFeaturePreprocessor

    @staticmethod
    def _labels(graphs):
        return np.array(
            [
                int(graph.y.item())
                for graph in graphs
            ],
            dtype=int,
        )

    @property
    def ytr(self):
        return self._labels(self.train_graphs)

    @property
    def yva(self):
        return self._labels(self.val_graphs)

    @property
    def yte(self):
        if not self.test_graphs:
            return None

        return self._labels(self.test_graphs)

    @property
    def view(self):
        return config.VIEW

    @property
    def variant(self):
        return config.VARIANT

    @property
    def n_node_features(self):
        return len(self.node_feature_names)

    @property
    def n_edge_features(self):
        return len(self.edge_feature_names)

    @property
    def usable(self):
        """Whether the training partition is usable for binary classification."""

        if (
            len(self.train_graphs)
            < config.MIN_TRAIN_GRAPHS
        ):
            return False

        y = np.array(
            [
                int(graph.y.item())
                for graph in self.train_graphs
            ]
        )

        return (
            len(np.unique(y)) > 1
        )

    def describe(self):
        """Compact human-readable dataset summary."""

        def rug_rate(graphs):
            if not graphs:
                return float('nan')

            y = np.array(
                [
                    float(graph.y.item())
                    for graph in graphs
                ]
            )

            return float(y.mean())

        test_text = (
            f'test={len(self.test_graphs)} '
            '(frozen)'
        )

        if config.EVALUATE_TEST:
            test_text = (
                f'test={len(self.test_graphs)} '
                f'(rug {rug_rate(self.test_graphs):.3f})'
            )

        return (
            f'[{self.experiment} | '
            f'{self.scope} | '
            f'{self.view} | '
            f'{self.cohort}] '
            f'{len(self.node_feature_names)} node features | '
            f'{len(self.edge_feature_names)} stored edge features | '
            f'train={len(self.train_graphs)} '
            f'(rug {rug_rate(self.train_graphs):.3f}) | '
            f'val={len(self.val_graphs)} '
            f'(rug {rug_rate(self.val_graphs):.3f}) | '
            f'{test_text}'
        )


def get_data(
    scope='full',
    q='q1',
    cohort='original',
):
    """Load graphs and fit preprocessing on the training partition only."""

    corpus = load_graphs(
        scope=scope,
        q=q,
        cohort=cohort,
    )

    train_raw = corpus.graphs_for_split(
        'train'
    )

    val_raw = corpus.graphs_for_split(
        'val'
    )

    test_raw = corpus.graphs_for_split(
        'test'
    )

    if not train_raw:
        raise ValueError(
            f'{q}/{cohort}: empty training graph partition'
        )

    y_train_raw = np.array(
        [
            int(graph.y.item())
            for graph in train_raw
        ],
        dtype=int,
    )

    if len(np.unique(y_train_raw)) < 2:
        raise ValueError(
            f'{q}/{cohort}: training partition '
            'does not contain both classes'
        )

    if not val_raw:
        raise ValueError(
            f'{q}/{cohort}: empty validation graph partition'
        )

    y_val_raw = np.array(
        [
            int(graph.y.item())
            for graph in val_raw
        ],
        dtype=int,
    )

    if len(np.unique(y_val_raw)) < 2:
        raise ValueError(
            f'{q}/{cohort}: validation partition '
            'does not contain both classes; '
            'validation ROC-AUC is undefined'
        )

    if config.EVALUATE_TEST:

        if not test_raw:
            raise ValueError(
                f'{q}/{cohort}: empty test graph partition '
                'with EVALUATE_TEST=True'
            )

        y_test_raw = np.array(
            [
                int(graph.y.item())
                for graph in test_raw
            ],
            dtype=int,
        )

        if len(np.unique(y_test_raw)) < 2:
            raise ValueError(
                f'{q}/{cohort}: test partition '
                'does not contain both classes; '
                'test ROC-AUC is undefined'
            )

    preprocessor = (
        NodeFeaturePreprocessor()
        .fit(train_raw)
    )

    train_graphs = preprocessor.transform(
        train_raw
    )

    val_graphs = preprocessor.transform(
        val_raw
    )

    test_graphs = preprocessor.transform(
        test_raw
    )

    return GraphRunData(
        train_graphs=train_graphs,
        val_graphs=val_graphs,
        test_graphs=test_graphs,

        node_feature_names=list(
            config.NODE_FEATURES
        ),

        edge_feature_names=list(
            config.EDGE_FEATURES
        ),

        scope=scope,
        experiment=q,
        cohort=cohort,

        preprocessor=preprocessor,
    )