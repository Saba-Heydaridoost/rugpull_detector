"""Central configuration for the GNN experiments.

Everything that defines the common GNN experiment protocol lives here:
artifact paths, graph schema, preprocessing, Optuna settings, training
protocol, evaluation policy, and plotting settings.

Model-specific Optuna search spaces live next to each architecture under
`gnn/models/`.
"""

import os


# =============================================================================
# Paths
# =============================================================================

# Anchored to trainer/, exactly as in the baseline package.
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

ARTIFACT_DIR = os.path.join(
    BASE_DIR,
    'artifacts'
)

COHORT_DIR = os.path.join(
    ARTIFACT_DIR,
    'final',
    'missingness_experiments'
)

# Keep GNN results separate while preserving the same output conventions.
RESULT_DIR = os.path.join(
    BASE_DIR,
    'results',
    'gnn'
)

PLOT_DIR = os.path.join(
    RESULT_DIR,
    'plots'
)


# =============================================================================
# Dataset axes
# =============================================================================

EXPERIMENTS = [
    'q1',
    'q2',
    'q3',
]

SCOPES = [
    'full',
]

COHORTS = [
    'original',
    'drop_3_or_more_missing',
    'complete_case',
]

# GNN is a graph-level view over the recovered transfer graph.
VIEW = 'gnn_graph'

# Initial experiment uses structure/timing only.
# No amount- or price-based edge attributes yet.
VARIANT = 'structure_timing'


# =============================================================================
# Input artifacts
# =============================================================================

GRAPH_META_FILE = (
    'model_graph_features_with_split.parquet'
)

EDGE_FILE = (
    'edges_with_split.parquet'
)


# =============================================================================
# Graph schema
# =============================================================================

TOKEN_COL = 'token_mint'
LP_COL = 'lp_address'
LABEL_COL = 'rug_pull_label'
SPLIT_COL = 'split'
CREATION_TIME_COL = 'creation_time'

SOURCE_COL = 'source_address'
TARGET_COL = 'target_address'
EDGE_TIME_COL = 't_frac'

SPLIT_ALIASES = {
    'validation': 'val',
}

VALID_SPLITS = {
    'train',
    'val',
    'test',
}


# =============================================================================
# Initial GNN predictors
# =============================================================================

# Endpoint-level structural / temporal node features.
# All are derived only from recovered transfers inside the observation window.
NODE_FEATURES = [
    'log1p_out_transfers',
    'log1p_in_transfers',
    'log1p_out_neighbors',
    'log1p_in_neighbors',
    'first_t_frac',
    'mean_t_frac',
]

# Stored in each PyG graph for later edge-aware architectures.
# GCN / GraphSAGE need not consume edge_attr.
EDGE_FEATURES = [
    EDGE_TIME_COL,
]


# =============================================================================
# Preprocessing
# =============================================================================

# Fit on TRAIN nodes only.
# Same numerical transformation family as the baseline pipeline.
TRANSFORM = 'quantile'


# =============================================================================
# Hyperparameter search
# =============================================================================

# Same default budget as the updated baseline framework.
# Individual GNN architectures may override this with N_TRIALS.
N_TRIALS = 60

# Fixed TPE sampler seed for reproducible proposals.
SAMPLER_SEED = 0

# Same TPE startup budget as the baseline framework.
N_STARTUP_TRIALS = 15

# No wall-clock limit unless explicitly changed.
SEARCH_TIMEOUT = None


# =============================================================================
# Evaluation protocol
# =============================================================================

# Final selected configuration is repeated over these seeds.
N_SEEDS = 5

# Fixed seed used during Optuna search.
SEARCH_SEED = 0

# Keep test frozen until architecture and ablations are final.
EVALUATE_TEST = False

# Minimum number of training graphs.
MIN_TRAIN_GRAPHS = 20


# =============================================================================
# GNN training
# =============================================================================

MAX_EPOCHS = 200
PATIENCE = 25

# Numerical tolerance for early-stopping improvement.
EARLY_STOP_MIN_DELTA = 1e-5

# CPU in your current local environment; CUDA is selected automatically
# when available on another machine.
DEFAULT_BATCH_SIZE = 8


# =============================================================================
# Reproducibility
# =============================================================================

DETERMINISTIC = True


# =============================================================================
# Plots
# =============================================================================

MAKE_PLOTS = True

PLOT_METRICS = [
    'f1',
    'recall',
]

PLOT_DPI = 160