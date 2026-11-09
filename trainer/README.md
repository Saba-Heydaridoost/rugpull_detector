# trainer/

This folder trains all models. Run every command from this folder.

* **Inputs.** The model-ready artifacts are read from `artifacts/final/`, which is a copy of
  `data_analysis/artifacts/final/` exported by `EDA.ipynb`.
* **Outputs.** Results go to `results/` for the chronological split and `results_stratified/` when
  `RUGPULL_SPLIT=stratified`.
* **Git.** Neither `artifacts/` nor the results folders are in git.

| Path | Contents |
|------|----------|
| [`baselines/`](baselines/) | The 9 non-graph models, with shared data loading, evaluation and reporting |
| [`gnn/`](gnn/) | The graph neural networks (GCN, GraphSAGE, GAT, DirSAGE, DirGCN) and the per-token graph construction |
| `run_*.py` | Experiment drivers (below) |

## Common protocol

Baselines and GNNs follow the same protocol in every cell (scope × window × cohort × view ×
missingness variant):

1. **Search.** Optuna TPE search on the validation partition, with the test partition untouched:
   * 60 trials per study for the baselines, 40 for the GNNs;
   * sampler seed 0 and 15 random start-up trials.
2. **Retrain and score.** The selected configuration is retrained with **5 seeds**.
   * The classification threshold maximizes F1 on validation.
   * Test is scored once, with that frozen threshold.
3. **Reported metrics.** AP (primary) and ROC-AUC, as mean ± sd over seeds. Predictions are saved
   per seed (`final_predictions_*.csv`) for the analyses in `result_analysis/`.

Settings used in the paper:

* **Scopes.** `full` (every valid token of a window) and `panel` (the tokens valid in all three
  windows).
* **Windows.** `q1`, `q2`, `q3`.
* **Cohorts.** `original`, `drop_3_or_more_missing`, `complete_case`.
* **Missingness variants.** `no_mask` and `with_mask`.
* **Baseline views.**
  * `node`: token-level features;
  * `node_matched`: token-level features on the graph-bearing tokens;
  * `graph`: graph-summary features;
  * `node_graph`: both, the main view.

## Official runs

```bash
python run_baselines.py                 # 9 baselines, all cells (see the switchboard in the file)
python run_gnn.py                       # GCN / GraphSAGE / GAT on per-token transfer graphs
python run_gnn.py --graph-only --cohorts original   # graph-only ablation (results/gnn_graph_only)
```

Both drivers have a switchboard at the top of the file and command-line overrides; `--list` shows
the available models.

> **Selection metric.** The official GCN / GraphSAGE / GAT runs reported in the paper were tuned
> on **validation ROC-AUC** and are stored in `results/gnn/`. The selection metric was later made
> configurable (`SELECTION_METRIC` in `gnn/config.py`, default `"ap"`; the output folder is then
> `results/gnn_selection_<metric>/`). To reproduce the official GNN runs, set
> `SELECTION_METRIC = "roc_auc"`. The direction-aware variants below were tuned on validation AP,
> as the paper states.

## Model variants and sensitivity runs reported in the paper

All variants use the original cohort, `no_mask`, and the chronological split.

| Script | Paper item | Output |
|--------|------------|--------|
| `run_gnn_improved.py` / `.sh` | DirSAGE and DirGCN: direction-aware aggregation, mean–max pooling, tuned on validation AP | `results/gnn_improved/` |
| `run_gnn_stacked.py` | Bagged GraphSAGE (5 fold models), and stacking of the out-of-fold GraphSAGE score into gradient boosting / XGBoost | `results/gnn_stacked/` |
| `run_gnn_global_ablation.py` / `.sh` | One global, time-respecting transaction graph across tokens (q1), for GCN / GraphSAGE | `results/gnn_global_ablation/` |
| `run_hnn4rp.py` | Adapted HNN4RP (Li et al., 2025): time-respecting neighbor sampling on a token–wallet graph with an attention BiLSTM | `results[_stratified]/hnn4rp/` |
| `run_xgb_spw_sensitivity.py` | XGBoost re-tuned with `scale_pos_weight ∈ {1, n_neg/n_pos}` | `results_xgb_spw/` |

**Stratified split.** Run the baselines and GNNs with `RUGPULL_SPLIT=stratified`. This needs the
artifacts from `data_analysis/EDA_stratified.ipynb`.

```bash
RUGPULL_SPLIT=stratified python run_baselines.py --models xgboost --cohorts original
RUGPULL_SPLIT=stratified python run_gnn.py --models graphsage --cohorts original
```

## Result file names

```
runs_<model>_<scope>_<q>_<view>_<variant>_<cohort>.csv              validation-selected runs per seed
final_predictions_<model>_<scope>_<q>_<view>_<variant>_<cohort>.csv per-token scores (val + test)
sweep_<...>.csv                                                    all Optuna trials
gnn/final_runs_<model>_<scope>_<q>_gnn_graph_structure_timing_node_graph_<variant>_<cohort>.csv
```
