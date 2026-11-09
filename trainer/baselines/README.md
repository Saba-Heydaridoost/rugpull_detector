# trainer/baselines/

This package holds the non-graph tabular baselines. Run it through `trainer/run_baselines.py`.

## Models (`models/`)

| Name | File |
|------|------|
| Dummy (prior) | `dummy.py` |
| Logistic regression | `logistic_regression.py` |
| Decision tree | `decision_tree.py` |
| Random forest | `random_forest.py` |
| Gradient boosting | `gradient_boosting.py` |
| XGBoost | `xgboost_model.py` |
| MLP | `mlp.py` |
| FT-Transformer | `ft_transformer.py` |
| AutoInt | `autoint.py` |

Each module exposes:

* `NAME`;
* `SPACE(trial)`, its Optuna search space;
* `build(params)`;
* `train(...)`.

To add a model, register it in `models/__init__.py`.

In the official runs, XGBoost keeps `scale_pos_weight = 1`. The class-weight sensitivity run lives
in `trainer/run_xgb_spw_sensitivity.py`.

## Feature views (`config.py`)

| View | Predictors |
|------|-----------|
| `node` | token-level features, on all valid tokens |
| `node_matched` | token-level features, restricted to graph-bearing tokens (matched to the GNN population) |
| `graph` | graph-summary features of the transfer graph |
| `node_graph` | token-level and graph-summary features (14), the main view and the GNN's tabular input |

The views the paper reports are `node`, `node_matched`, `graph` and `node_graph`.

## Other settings (`config.py`)

* **Missingness variants.** `no_mask` (imputation only) and `with_mask` (also binary missingness
  indicators). Imputation is fitted on the training partition.
* **Cohorts.** Read from `artifacts/final/missingness_experiments/<cohort>/<q>/`.

## Other modules

| File | Purpose |
|------|---------|
| `config.py` | Paths, scopes, windows, cohorts, views, predictor lists, and the Optuna and evaluation settings |
| `data.py` | Loads the Parquet artifacts and builds fitted design matrices (train-only imputation and scaling) |
| `evaluation.py` | Optuna search on validation, then retraining over 5 seeds and scoring with a frozen threshold |
| `metrics.py` | AP, ROC-AUC, Brier and F1, and selection of the threshold on validation |
| `reporting.py`, `plotting.py` | Writing runs, sweeps and predictions; train vs validation plots |
| `torch_utils.py` | Shared components for MLP, FT-Transformer and AutoInt |
