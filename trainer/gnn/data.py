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
The graph encoder uses structural and temporal endpoint features derived
only from recovered transfers inside the observation window.

The primary graph-level classifier additionally receives the exact
tabular predictors used by the matched `node_graph` baseline:

    - five token-level predictors;
    - nine recovered-graph summary predictors.

These auxiliary predictors are concatenated with the pooled graph
embedding before classification.

Thus the tabular baseline and the GNN share the same graph-level
predictor set, while the GNN additionally learns from graph topology
and endpoint-level representations.

Canonical normalized transfer amounts remain excluded from the primary
representation.

Preprocessing
-------------
Node-feature transforms are fitted on nodes from TRAIN graphs only and
then applied unchanged to validation and test graphs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

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


@lru_cache(maxsize=None)
def _panel_tokens(cohort):
    """Return graph-eligible token mints shared by q1/q2/q3 for one cohort."""

    if cohort not in config.COHORTS:
        raise ValueError(
            f'unknown cohort {cohort!r}; '
            f'expected one of {config.COHORTS}'
        )

    frames = {}
    token_sets = []

    for q in config.EXPERIMENTS:
        folder = os.path.join(
            config.COHORT_DIR,
            cohort,
            q,
        )

        meta_path = os.path.join(
            folder,
            config.GRAPH_META_FILE,
        )

        _check_artifact(meta_path)

        frame = _normalise_split(
            pd.read_parquet(
                meta_path,
                columns=[
                    config.TOKEN_COL,
                    config.LABEL_COL,
                    config.SPLIT_COL,
                ],
            )
        )

        frame = (
            frame[
                frame[config.SPLIT_COL]
                .isin(config.VALID_SPLITS)
            ]
            .copy()
            .reset_index(drop=True)
        )

        if frame[config.TOKEN_COL].duplicated().any():
            raise AssertionError(
                f'{q}/{cohort}: graph metadata contains '
                'duplicate token_mint rows'
            )

        frames[q] = frame

        token_sets.append(
            set(frame[config.TOKEN_COL])
        )

    common_tokens = frozenset(
        set.intersection(*token_sets)
    )

    if not common_tokens:
        raise ValueError(
            f'{cohort}: no graph-eligible tokens are shared '
            'across q1/q2/q3'
        )

    audit = pd.concat(
        [
            frames[q][
                frames[q][config.TOKEN_COL]
                .isin(common_tokens)
            ]
            .assign(experiment_id=q)
            for q in config.EXPERIMENTS
        ],
        ignore_index=True,
    )

    if (
        audit
        .groupby(config.TOKEN_COL)[config.LABEL_COL]
        .nunique()
        .gt(1)
        .any()
    ):
        raise AssertionError(
            f'{cohort}: panel contains inconsistent labels '
            'across q1/q2/q3'
        )

    if (
        audit
        .groupby(config.TOKEN_COL)[config.SPLIT_COL]
        .nunique()
        .gt(1)
        .any()
    ):
        raise AssertionError(
            f'{cohort}: panel contains inconsistent split assignments '
            'across q1/q2/q3'
        )

    return common_tokens


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

def _build_graph(
    edges,
    sample_id,
    label,
    token_features,
):
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
        token_features=torch.tensor(
            np.asarray(
                token_features,
                dtype=np.float64,
            ).reshape(1, -1),
            dtype=torch.float64,
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
    feature_variant='base',
):
    """Load one GNN dataset before train-only numerical preprocessing."""

    require_pyg()

    if feature_variant not in config.FEATURE_VARIANTS:
        raise ValueError(
            f'unknown feature variant {feature_variant!r}; '
            f'expected one of {config.FEATURE_VARIANTS}'
        )

    token_features = list(
        config.TOKEN_FEATURE_SETS[feature_variant]
    )

    global_features = list(
        config.GLOBAL_FEATURE_SETS[feature_variant]
    )

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

    node_path = os.path.join(
        folder,
        config.COHORT_NODE_FILES[cohort],
    )

    _check_artifact(meta_path)
    _check_artifact(edge_path)
    _check_artifact(node_path)

    meta_columns = [
        config.LP_COL,
        config.TOKEN_COL,
        config.CREATION_TIME_COL,
        config.LABEL_COL,
        config.SPLIT_COL,
    ] + list(
        config.GRAPH_SUMMARY_FEATURES
    )

    edge_columns = [
        config.TOKEN_COL,
        config.SOURCE_COL,
        config.TARGET_COL,
        config.EDGE_TIME_COL,
        config.SPLIT_COL,
        'timestamp',
        'tx_signature',
    ]

    node_columns = [
        config.LP_COL,
        config.TOKEN_COL,
        config.LABEL_COL,
        config.SPLIT_COL,
    ] + list(
        token_features
    )

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

    token_frame = _normalise_split(
        pd.read_parquet(
            node_path,
            columns=node_columns,
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
    ] + list(
        config.GRAPH_SUMMARY_FEATURES
    )

    required_edges = [
        config.TOKEN_COL,
        config.SOURCE_COL,
        config.TARGET_COL,
        config.EDGE_TIME_COL,
        config.SPLIT_COL,
    ]

    required_token_columns = [
        config.LP_COL,
        config.TOKEN_COL,
        config.LABEL_COL,
        config.SPLIT_COL,
    ] + list(
        token_features
    )

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

    _check_columns(
        token_frame,
        required_token_columns,
        node_path,
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

    token_frame = (
        token_frame[
            token_frame[config.SPLIT_COL]
            .isin(config.VALID_SPLITS)
        ]
        .copy()
        .reset_index(drop=True)
    )

    # -----------------------------------------------------------------
    # Controlled common-token panel.
    #
    # The full path above is unchanged. For panel runs only, retain
    # exactly the graph-eligible token mints that are available in all
    # three observation windows for the selected missingness cohort.
    # -----------------------------------------------------------------

    if scope == 'panel':
        panel_tokens = _panel_tokens(cohort)

        meta = (
            meta[
                meta[config.TOKEN_COL]
                .isin(panel_tokens)
            ]
            .copy()
            .reset_index(drop=True)
        )

        edges = (
            edges[
                edges[config.TOKEN_COL]
                .isin(panel_tokens)
            ]
            .copy()
            .reset_index(drop=True)
        )

        token_frame = (
            token_frame[
                token_frame[config.TOKEN_COL]
                .isin(panel_tokens)
            ]
            .copy()
            .reset_index(drop=True)
        )

        if set(meta[config.TOKEN_COL]) != set(panel_tokens):
            raise AssertionError(
                f'{q}/{cohort}: graph metadata does not exactly '
                'match the common panel token set'
            )

        if set(edges[config.TOKEN_COL]) != set(panel_tokens):
            raise AssertionError(
                f'{q}/{cohort}: recovered edges do not exactly '
                'match the common panel token set'
            )

        if set(token_frame[config.TOKEN_COL]) != set(panel_tokens):
            raise AssertionError(
                f'{q}/{cohort}: token-level features do not exactly '
                'match the common panel token set'
            )

    # -----------------------------------------------------------------
    # Integrity checks
    # -----------------------------------------------------------------

    if meta[config.TOKEN_COL].duplicated().any():
        raise AssertionError(
            'graph metadata contains duplicate token_mint rows'
        )

    if token_frame[config.TOKEN_COL].duplicated().any():
        raise AssertionError(
            'token-level feature table contains duplicate token_mint rows'
        )

    token_frame_tokens = set(
        token_frame[config.TOKEN_COL]
    )

    missing_token_features = (
        set(meta[config.TOKEN_COL])
        - token_frame_tokens
    )

    if missing_token_features:
        raise AssertionError(
            'token-level features are missing for '
            f'{len(missing_token_features)} graph token(s)'
        )

    token_frame_indexed = (
        token_frame
        .set_index(config.TOKEN_COL)
    )

    aligned_token_frame = (
        token_frame_indexed
        .reindex(
            meta[config.TOKEN_COL]
            .tolist()
        )
    )

    if not (
        aligned_token_frame[
            config.SPLIT_COL
        ].to_numpy()
        ==
        meta[
            config.SPLIT_COL
        ].to_numpy()
    ).all():
        raise AssertionError(
            'token-level and graph split assignments disagree'
        )

    if not (
        aligned_token_frame[
            config.LABEL_COL
        ].astype(int).to_numpy()
        ==
        meta[
            config.LABEL_COL
        ].astype(int).to_numpy()
    ).all():
        raise AssertionError(
            'token-level and graph labels disagree'
        )

    if not (
        aligned_token_frame[
            config.LP_COL
        ].astype(str).to_numpy()
        ==
        meta[
            config.LP_COL
        ].astype(str).to_numpy()
    ).all():
        raise AssertionError(
            'token-level and graph lp_address metadata disagree'
        )

    token_values = (
        aligned_token_frame[
            token_features
        ]
        .to_numpy(dtype=float)
    )

    if np.isinf(token_values).any():
        raise AssertionError(
            'token-level predictors contain infinite values'
        )

    if (
        cohort == 'complete_case'
        and np.isnan(token_values).any()
    ):
        raise AssertionError(
            'complete_case contains missing token-level predictors'
        )

    for feature in token_features:
        meta[feature] = (
            aligned_token_frame[
                feature
            ].to_numpy()
        )

    graph_summary_values = (
        meta[
            config.GRAPH_SUMMARY_FEATURES
        ]
        .to_numpy(dtype=float)
    )

    if (
        graph_summary_values.shape[1]
        != len(config.GRAPH_SUMMARY_FEATURES)
    ):
        raise AssertionError(
            'unexpected graph-summary feature dimension'
        )

    if np.isinf(
        graph_summary_values
    ).any():
        raise AssertionError(
            'graph-summary predictors contain infinite values'
        )

    if np.isnan(
        graph_summary_values
    ).any():
        raise AssertionError(
            'graph-summary predictors contain missing values'
        )

    global_values = (
        meta[
            global_features
        ]
        .to_numpy(dtype=float)
    )

    if (
        global_values.shape[1]
        != len(global_features)
    ):
        raise AssertionError(
            'unexpected global feature dimension'
        )

    if np.isinf(
        global_values
    ).any():
        raise AssertionError(
            'global predictors contain infinite values'
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
            token_features=(
                row[
                    global_features
                ]
                .astype(float)
                .to_numpy()
            ),
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


class TokenFeaturePreprocessor:
    """Train-fitted preprocessing for graph-level auxiliary predictors.

    This includes the token-level and graph-summary predictors used by
    the matched `node_graph` tabular baseline.
    """

    def __init__(
        self,
        add_mask=True,
        transform=None,
        columns=None,
    ):
        self.add_mask = add_mask

        self.transform_name = (
            transform
            or config.TRANSFORM
        )

        self.requested_columns_ = (
            list(config.GLOBAL_FEATURES)
            if columns is None
            else list(columns)
        )

    def fit(
        self,
        graphs,
    ):
        if not graphs:
            raise ValueError(
                'cannot fit token-feature preprocessor '
                'on an empty graph list'
            )

        X = np.vstack(
            [
                graph.token_features
                .cpu()
                .numpy()
                .reshape(1, -1)
                for graph in graphs
            ]
        ).astype(np.float64)

        if X.shape[1] != len(
            self.requested_columns_
        ):
            raise AssertionError(
                'unexpected raw token-feature dimension'
            )

        self.columns_ = list(
            self.requested_columns_
        )

        frame = pd.DataFrame(
            X,
            columns=self.columns_,
        )

        self.median_ = (
            frame
            .median(numeric_only=True)
            .reindex(self.columns_)
            .fillna(0.0)
        )

        Xi = (
            frame
            .fillna(self.median_)
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )

        if self.transform_name == 'quantile':

            self.scaler_ = QuantileTransformer(
                n_quantiles=max(
                    10,
                    min(
                        1000,
                        Xi.shape[0],
                    ),
                ),
                output_distribution='normal',
                subsample=10 ** 9,
                random_state=0,
            )

            self.scaler_.fit(
                Xi
            )

        elif self.transform_name == 'standard':

            self.mu_ = Xi.mean(
                axis=0
            )

            sd = Xi.std(
                axis=0
            )

            self.sd_ = np.where(
                sd < 1e-12,
                1.0,
                sd,
            )

        else:
            raise ValueError(
                f'unknown transform '
                f'{self.transform_name!r}'
            )

        return self

    def transform_graph(
        self,
        graph,
    ):
        X = (
            graph.token_features
            .cpu()
            .numpy()
            .reshape(1, -1)
            .astype(np.float64)
        )

        frame = pd.DataFrame(
            X,
            columns=self.columns_,
        )

        mask = (
            frame
            .isna()
            .to_numpy(dtype=np.float64)
        )

        Xi = (
            frame
            .fillna(self.median_)
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )

        if self.transform_name == 'quantile':

            Xt = self.scaler_.transform(
                Xi
            )

        else:

            Xt = (
                Xi
                - self.mu_
            ) / self.sd_

        if self.add_mask:
            Xt = np.hstack(
                [
                    Xt,
                    mask,
                ]
            )

        Xt = np.nan_to_num(
            Xt,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        if not np.isfinite(
            Xt
        ).all():
            raise FloatingPointError(
                'token-feature preprocessing produced '
                'non-finite values'
            )

        out = graph.clone()

        out.token_features = torch.tensor(
            Xt,
            dtype=torch.float32,
        )

        return out

    def transform(
        self,
        graphs,
    ):
        return [
            self.transform_graph(
                graph
            )
            for graph in graphs
        ]

    def output_names(
        self,
    ):
        if not self.add_mask:
            return list(
                self.columns_
            )

        return (
            list(
                self.columns_
            )
            + [
                f'{column}__missing'
                for column in self.columns_
            ]
        )
    

# =============================================================================
# Final run data
# =============================================================================

@dataclass
class GraphRefitData:
    """Graphs for the final refit on train + validation.

    Node and token preprocessors are re-fitted on the combined partition. Validation is
    in-sample here; only the test partition is scored.
    """

    train_graphs: list
    test_graphs: list

    train_token_mints: np.ndarray
    test_token_mints: np.ndarray

    n_node_features: int
    n_edge_features: int
    n_token_features: int

    @property
    def ytr(self):
        return GraphRunData._labels(
            self.train_graphs
        )

    @property
    def yte(self):
        if not self.test_graphs:
            return None

        return GraphRunData._labels(
            self.test_graphs
        )


@dataclass
class GraphRunData:
    """Prepared graph datasets plus experiment context."""

    train_graphs: list
    val_graphs: list
    test_graphs: list

    train_token_mints: np.ndarray
    val_token_mints: np.ndarray
    test_token_mints: np.ndarray

    node_feature_names: list
    edge_feature_names: list
    token_feature_names: list

    scope: str
    experiment: str
    cohort: str
    feature_variant: str
    mask_variant: str

    node_preprocessor: NodeFeaturePreprocessor
    token_preprocessor: TokenFeaturePreprocessor

    # Train + validation graphs for the refit protocol; None unless requested.
    refit: GraphRefitData | None = None

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
        return self._labels(
            self.train_graphs
        )

    @property
    def yva(self):
        return self._labels(
            self.val_graphs
        )

    @property
    def yte(self):
        if not self.test_graphs:
            return None

        return self._labels(
            self.test_graphs
        )

    @property
    def view(self):
        return config.VIEW

    @property
    def variant(self):
        prefix = (
            config.GRAPH_VARIANT
            if config.USE_GLOBAL_FEATURES
            else config.GRAPH_ONLY_VARIANT
        )

        # Keep historical primary result names unchanged.
        if self.feature_variant != 'base':
            prefix = (
                f'{prefix}_'
                f'{self.feature_variant}'
            )

        return (
            f'{prefix}_'
            f'{self.mask_variant}'
        )

    @property
    def n_node_features(self):
        return len(
            self.node_feature_names
        )

    @property
    def n_edge_features(self):
        return len(
            self.edge_feature_names
        )

    @property
    def n_token_features(self):
        # Width of the auxiliary block entering the head; zero for graph-only.
        if not config.USE_GLOBAL_FEATURES:
            return 0

        return len(
            self.token_feature_names
        )

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

            return float(
                y.mean()
            )

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
            f'{self.variant} | '
            f'{self.cohort}] '
            f'{len(self.node_feature_names)} node features | '
            f'{self.n_token_features} token features | '
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
    variant='with_mask',
    feature_variant='base',
    refit=False,
):
    """Load graphs and fit all numerical preprocessing on training only.

    With `refit`, the returned data also carries train + validation graphs, preprocessed
    by preprocessors fitted on that combined partition, for the final refit (`data.refit`).
    """

    if feature_variant not in config.FEATURE_VARIANTS:
        raise ValueError(
            f'unknown feature variant {feature_variant!r}; '
            f'expected one of {config.FEATURE_VARIANTS}'
        )

    if variant not in config.TOKEN_VARIANTS:
        raise ValueError(
            f'unknown token-feature variant '
            f'{variant!r}; expected one of '
            f'{config.TOKEN_VARIANTS}'
        )

    if (
        cohort == 'complete_case'
        and variant not in config.COMPLETE_CASE_VARIANTS
    ):
        raise ValueError(
            'complete_case must use no_mask because '
            'its token-level predictors contain no missing values'
        )

    corpus = load_graphs(
        scope=scope,
        q=q,
        cohort=cohort,
        feature_variant=feature_variant,
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

    train_token_mints = (
        corpus.metadata.loc[
            corpus.metadata[
                config.SPLIT_COL
            ] == 'train',
            config.TOKEN_COL,
        ]
        .astype(str)
        .to_numpy()
    )

    val_token_mints = (
        corpus.metadata.loc[
            corpus.metadata[
                config.SPLIT_COL
            ] == 'val',
            config.TOKEN_COL,
        ]
        .astype(str)
        .to_numpy()
    )

    test_token_mints = (
        corpus.metadata.loc[
            corpus.metadata[
                config.SPLIT_COL
            ] == 'test',
            config.TOKEN_COL,
        ]
        .astype(str)
        .to_numpy()
    )

    if (
        len(train_token_mints)
        != len(train_raw)
    ):
        raise AssertionError(
            'train token IDs do not align with graphs'
        )

    if (
        len(val_token_mints)
        != len(val_raw)
    ):
        raise AssertionError(
            'validation token IDs do not align with graphs'
        )

    if (
        len(test_token_mints)
        != len(test_raw)
    ):
        raise AssertionError(
            'test token IDs do not align with graphs'
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

    def fit_preprocessors(
        graphs,
    ):
        node = (
            NodeFeaturePreprocessor()
            .fit(
                graphs
            )
        )

        token = (
            TokenFeaturePreprocessor(
                add_mask=(
                    variant
                    == 'with_mask'
                ),
                columns=(
                    config.GLOBAL_FEATURE_SETS[
                        feature_variant
                    ]
                ),
            )
            .fit(
                graphs
            )
        )

        return node, token

    node_preprocessor, token_preprocessor = fit_preprocessors(
        train_raw
    )

    train_graphs = node_preprocessor.transform(
        train_raw
    )

    val_graphs = node_preprocessor.transform(
        val_raw
    )

    test_graphs = node_preprocessor.transform(
        test_raw
    )

    train_graphs = token_preprocessor.transform(
        train_graphs
    )

    val_graphs = token_preprocessor.transform(
        val_graphs
    )

    test_graphs = token_preprocessor.transform(
        test_graphs
    )

    refit_data = None

    if refit:

        trva_raw = list(train_raw) + list(val_raw)

        refit_node, refit_token = fit_preprocessors(
            trva_raw
        )

        refit_data = GraphRefitData(
            train_graphs=refit_token.transform(
                refit_node.transform(
                    trva_raw
                )
            ),
            test_graphs=refit_token.transform(
                refit_node.transform(
                    test_raw
                )
            ),

            train_token_mints=np.concatenate(
                [
                    train_token_mints,
                    val_token_mints,
                ]
            ),
            test_token_mints=test_token_mints,

            n_node_features=len(
                config.NODE_FEATURES
            ),
            n_edge_features=len(
                config.EDGE_FEATURES
            ),
            n_token_features=(
                len(refit_token.output_names())
                if config.USE_GLOBAL_FEATURES
                else 0
            ),
        )

    return GraphRunData(
        train_graphs=train_graphs,
        val_graphs=val_graphs,
        test_graphs=test_graphs,

        train_token_mints=train_token_mints,
        val_token_mints=val_token_mints,
        test_token_mints=test_token_mints,

        node_feature_names=list(
            config.NODE_FEATURES
        ),

        edge_feature_names=list(
            config.EDGE_FEATURES
        ),

        token_feature_names=(
            token_preprocessor.output_names()
        ),

        scope=scope,
        experiment=q,
        cohort=cohort,
        feature_variant=feature_variant,
        mask_variant=variant,

        node_preprocessor=node_preprocessor,
        token_preprocessor=token_preprocessor,

        refit=refit_data,
    )