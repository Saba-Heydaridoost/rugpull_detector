"""Central configuration for the baseline experiments.

Everything that defines *what* is trained lives here: paths, the predictor set for each
dataset view, and the evaluation protocol. Model hyperparameter search spaces live next to
their model in `baselines/models/`.
"""

import os

# =============================================================================
# Paths
# =============================================================================
#
# Anchored to the `trainer/` directory that holds this package, so a run behaves the
# same whatever the working directory is.

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Which train/val/test split to train on, chosen with the RUGPULL_SPLIT environment
# variable:
#
#   chronological -> artifacts/ exported by data_analysis/EDA.ipynb (the default)
#   stratified    -> data_analysis/artifacts_stratified/ exported by
#                    data_analysis/EDA_stratified.ipynb, with results kept apart
#                    under results_stratified/
SPLIT = os.environ.get('RUGPULL_SPLIT', 'chronological')

if SPLIT not in ('chronological', 'stratified'):
    raise ValueError(f'unknown RUGPULL_SPLIT {SPLIT!r}')

# Parquet artifacts written by the data-preparation notebook.
ARTIFACT_DIR = (
    os.path.join(BASE_DIR, 'artifacts')
    if SPLIT == 'chronological'
    else os.path.join(os.path.dirname(BASE_DIR), 'data_analysis', 'artifacts_stratified')
)

# Cohort folders written by the final export cell of EDA.ipynb.
COHORT_DIR = os.path.join(ARTIFACT_DIR, 'final', 'missingness_experiments')

# Where per-model runs and the cumulative summary are written.
RESULT_DIR = os.path.join(
    BASE_DIR, 'results' if SPLIT == 'chronological' else 'results_stratified'
)

# Where figures are written.
PLOT_DIR = os.path.join(RESULT_DIR, 'plots')


# =============================================================================
# Dataset axes
# =============================================================================

EXPERIMENTS = ['q1', 'q2', 'q3']

# `full` keeps every graph-eligible sample available in each observation
# window. `panel` will restrict all q1/q2/q3 runs to the same token set
# for controlled cross-window comparisons.
SCOPES = ['full', 'panel']

VIEWS = [
    'node',
    'node_matched',
    'graph',
    'node_graph',
    'node_graph_freeze',
]

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

FREEZE_AUTHORITY_FEATURE = (
    'has_initial_freeze_authority'
)

NODE_FEATURES_WITH_FREEZE = (
    NODE_FEATURES
    + [FREEZE_AUTHORITY_FEATURE]
)

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
    'node':
        NODE_FEATURES,

    'node_matched':
        NODE_FEATURES,

    'graph':
        GRAPH_FEATURES,

    'node_graph':
        NODE_FEATURES + GRAPH_FEATURES,

    'node_graph_freeze':
        NODE_FEATURES_WITH_FREEZE + GRAPH_FEATURES,
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
# Hyperparameter search
# =============================================================================
#
# Tuning is an Optuna study per (model, cohort, experiment, view, variant) cell, maximising
# validation ROC-AUC with the TPE sampler. The search space lives next to its model in
# `baselines/models/` as a `SPACE(trial)` function.
#
# This replaces the exhaustive grid sweep the baselines used previously. The practical
# difference is that continuous hyperparameters (learning rates, regularisation strengths,
# subsample fractions) are now sampled from a range instead of being pinned to a handful of
# values, and the trial budget is set directly rather than falling out of the product of the
# grid axes.

# Trials per study, unless the model overrides it with its own `N_TRIALS`.
N_TRIALS = 60

# Sampler seed. Fixed so that re-running a cell proposes the same configurations in the same
# order and the selected hyperparameters are reproducible.
SAMPLER_SEED = 0

# Trials drawn quasi-randomly before TPE starts modelling the objective. Below roughly a
# dozen, TPE is fitting its surrogate to too little data to beat random search.
N_STARTUP_TRIALS = 15

# Wall-clock cap per study, in seconds. None for no cap. A study stops at whichever of
# N_TRIALS and this limit comes first, which is the practical brake on the torch models.
SEARCH_TIMEOUT = None


# =============================================================================
# Evaluation protocol
# =============================================================================

# Seeds used to re-run the validation-selected configuration.
N_SEEDS = 5

# Single seed used to train each trial during the search. The winner is re-run over
# N_SEEDS afterwards, so search cost stays linear in the trial count.
SEARCH_SEED = 0

# The held-out partition stays frozen until the graph models are finalised as well, so
# that the final comparison remains a single test evaluation rather than a sequence.
EVALUATE_TEST = True

# Minimum training rows before a configuration is attempted.
MIN_TRAIN_ROWS = 20

# Threads for the estimators that parallelise their own fit (random forest, XGBoost).
#
# Not -1. These cohorts are a few hundred to a few thousand rows over five to ten
# predictors, and on a many-core machine the per-round thread synchronisation costs more
# than the work it distributes: measured on the q1 original cohort (1367 x 10), a 300-tree
# XGBoost fit takes 74.8s with one thread per core against 0.15s at eight. Raise it only
# if the training partitions grow by orders of magnitude.
N_JOBS = 8


# =============================================================================
# Plots
# =============================================================================

# Write a train-vs-validation figure for every model that finishes.
MAKE_PLOTS = True

# Metrics shown in the train-vs-validation figure. Any column produced by
# `metrics.compute_metrics` is valid: roc_auc, pr_auc, f1, precision, recall, brier.
PLOT_METRICS = ['f1', 'recall']

PLOT_DPI = 160