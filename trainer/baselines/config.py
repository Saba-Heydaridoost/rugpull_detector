"""Central configuration for the baseline experiments.

Everything that defines *what* is trained lives here: paths, the predictor set for each
dataset view, and the evaluation protocol. Model hyperparameter grids live next to their
model in `baselines/models/`.
"""

import os

# =============================================================================
# Paths
# =============================================================================
#
# Anchored to the `trainer/` directory that holds this package, so a run behaves the
# same whatever the working directory is.

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Parquet artifacts written by the data-preparation notebook.
ARTIFACT_DIR = os.path.join(BASE_DIR, 'artifacts')

# Cohort folders written by the final export cell of EDA.ipynb.
COHORT_DIR = os.path.join(ARTIFACT_DIR, 'final', 'missingness_experiments')

# Where per-model runs and the cumulative summary are written.
RESULT_DIR = os.path.join(BASE_DIR, 'results')

# Where figures are written.
PLOT_DIR = os.path.join(RESULT_DIR, 'plots')


# =============================================================================
# Dataset axes
# =============================================================================

EXPERIMENTS = ['q1', 'q2', 'q3']

# The final export covers the full dataset only; the panel scope is not part of it.
SCOPES = ['full']

VIEWS = ['node', 'graph', 'node_graph']

# Missingness cohorts, exported by EDA.ipynb as one folder per cohort under
# COHORT_DIR/<cohort>/<q>/.
#
#   original                -> all model-ready samples
#   drop_3_or_more_missing  -> rows with >= 3 missing node predictors removed
#   complete_case           -> rows with any missing node predictor removed
#
# The value is the node table to read for that cohort. The first two still carry missing
# values, so the NaN-preserving export is used and imputed here from the training
# partition only; complete_case has none by construction and ships a single table.
COHORT_NODE_FILES = {
    'original': 'model_nodes_raw_nan.parquet',
    'drop_3_or_more_missing': 'model_nodes_raw_nan.parquet',
    'complete_case': 'model_nodes_complete_case.parquet',
}

COHORTS = list(COHORT_NODE_FILES)

# Graph-summary table, the same filename in every cohort folder.
GRAPH_FILE = 'model_graph_features_with_split.parquet'


# =============================================================================
# Predictor selection
# =============================================================================
#
# Edit these lists to change which columns each view trains on. A name listed here but
# absent from the corresponding Parquet file raises an error rather than being silently
# dropped, so a typo cannot quietly shrink the feature set.

# Token-level summary features (COHORT_DIR/{cohort}/{q}/, node table).
NODE_FEATURES = [
    'volume_sell_change_percent',
    'volume_buy_change_percent',
    'unique_wallet_count',
    'unique_wallet_change_percent',
    'liquidity_at_observation_end',
]

# Recovered-transfer-graph summary features
# (COHORT_DIR/{cohort}/{q}/model_graph_features_with_split.parquet).
#
# Amount-based features are excluded upstream: the collector may fall back to raw base
# units when decimal metadata is unavailable, so their scale is not comparable across
# tokens.
GRAPH_FEATURES = [
    'n_transfers',
    'n_endpoints',
    'n_edges',
    'n_senders',
    'n_receivers',
    'density',
    'avg_degree',
    'first_t_frac',
    'mean_t_frac',
]

# The `graph` and `node_graph` views are restricted to tokens with at least one recovered
# edge. Graph features are never imputed for tokens whose edge job returned nothing,
# because the edge output cannot distinguish a completed zero-edge job from an unfinished
# one, and recovered-edge presence is itself class-dependent.
FEATURE_SETS = {
    'node': NODE_FEATURES,
    'graph': GRAPH_FEATURES,
    'node_graph': NODE_FEATURES + GRAPH_FEATURES,
}


# =============================================================================
# Frame schema
# =============================================================================

LABEL_COL = 'rug_pull_label'
SPLIT_COL = 'split'

# The export spells the validation partition 'validation'; it is normalised to 'val'
# on load so the rest of the pipeline sees a single name.
SPLIT_ALIASES = {'validation': 'val'}
ID_COLS = ['lp_address', 'token_mint']
META_COLS = ID_COLS + [LABEL_COL, SPLIT_COL]


# =============================================================================
# Preprocessing
# =============================================================================

# 'quantile' -> QuantileTransformer(output_distribution='normal'); 'standard' -> z-score.
# Fitted on the training partition only.
TRANSFORM = 'quantile'

# Node-feature missingness is class-dependent, so the missingness mask is itself a
# candidate predictor rather than a neutral encoding choice. Both variants are run so the
# contribution can be reported as an ablation.
VARIANTS = [
    'no_mask',
    'with_mask'
    ]


# =============================================================================
# Evaluation protocol
# =============================================================================

# Seeds used to re-run the validation-selected configuration.
N_SEEDS = 5

# Single seed used during the hyperparameter sweep.
SEARCH_SEED = 0

# The held-out partition stays frozen until the graph models are finalised as well, so
# that the final comparison remains a single test evaluation rather than a sequence.
EVALUATE_TEST = False

# Minimum training rows before a configuration is attempted.
MIN_TRAIN_ROWS = 20


# =============================================================================
# Plots
# =============================================================================

# Write a train-vs-validation figure for every model that finishes.
MAKE_PLOTS = True

# Metrics shown in the train-vs-validation figure. Any column produced by
# `metrics.compute_metrics` is valid: roc_auc, pr_auc, f1, precision, recall, brier.
PLOT_METRICS = ['f1', 'recall']

PLOT_DPI = 160