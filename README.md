# Early Rug-Pull Detection on Solana

Code for the paper *Early Rug Pull Detection on Solana*. The question the paper asks is whether a
graph neural network (GNN) that observes a new Solana token for a short window after launch detects
later rug pulls better than non-graph models that observe the same token for the same or longer
windows.

Each token is observed from its creation time `t0` up to a cutoff `t_q`, for three windows:

| Window | Cutoff after creation |
|--------|-----------------------|
| `q1`   | 750 s (12.5 min)      |
| `q2`   | 2,127 s (35.5 min)    |
| `q3`   | 5,585 s (93 min)      |

The label is decided strictly after the cutoff, within 24 hours of creation. A token is a rug pull
when all three of these hold:

* its liquidity drops by more than 80 % from its running peak;
* its price then drops by more than 90 %;
* the liquidity loss persists for 30 minutes.

## Main findings

* **Primary analysis.** The analysis was frozen before the test set was scored. It compares GCN,
  GraphSAGE and GAT at `q1` against XGBoost at `q1`, `q2` and `q3`: nine contrasts, with
  Bonferroni-adjusted paired bootstrap intervals over 10,000 resamples. None of the nine shows a
  statistically supported AP advantage for the GNN.
* **Within each window.** The best GNN has the higher AP in most cells, but no contrast is
  significant after Holm correction. A power analysis shows that the test sets (136–159 tokens) can
  detect only large AP differences.
* **Validation versus test.** Validation scores overstate test performance by a wide margin. The
  gap comes mainly from a shift in prevalence and token population across the chronological split,
  not from hyperparameter selection.
* **Ablations and variants.** None of these improves the conclusion:
  * other feature views, graph-only GNNs, missingness indicators and filtered cohorts;
  * direction-aware GNNs (DirSAGE, DirGCN), a global transaction graph, a bagged GraphSAGE and
    GNN-to-booster stacking;
  * an adapted HNN4RP;
  * re-tuning XGBoost with a positive-class weight.

## Repository layout

| Folder | Contents |
|--------|----------|
| [`dataset/`](dataset/) | Data collection: rule-based labels, token-level features, and per-token transfer edges. The released files are the token IDs, labels and token-level features. |
| [`data_analysis/`](data_analysis/) | Data cleaning, the chronological and stratified splits, model-ready artifacts, structural graph analysis, and the multivariate EDA (UMAP figure). |
| [`trainer/`](trainer/) | Model training: 9 non-graph baselines, GCN / GraphSAGE / GAT, and the model variants and sensitivity runs reported in the paper. |
| [`result_analysis/`](result_analysis/) | The frozen primary analysis, the exploratory analyses, and the scripts that produce every results table of the paper. |

Each folder has its own README with details.

## Pipeline

All commands run from the folder named in each step.

1. **Labels and features** (`dataset/`). This step needs the SolRPDS 2024 file, a Birdeye API key
   and a Solana RPC endpoint.

   ```bash
   python build_rule_based_labels.py   # -> rule_based_labels.csv
   python collect_nodes.py             # -> graph_nodes.csv
   python collect_edges.py             # -> graph_edges.csv
   python normalize_edge_amounts.py    # -> normalized edge amounts
   ```

2. **Cleaning, split and artifacts** (`data_analysis/`).
   * Run `EDA.ipynb`. It writes the chronological-split artifacts to `data_analysis/artifacts/`.
     Copy `artifacts/final/` to `trainer/artifacts/final/`.
   * `EDA_stratified.ipynb` builds the stratified-split artifacts used only for the split
     comparison.
   * `analyze_graphs.py` and `compare_graphs.py` produce the structural statistics.

3. **Training** (`trainer/`).

   ```bash
   python run_baselines.py   # non-graph baselines
   python run_gnn.py         # GCN / GraphSAGE / GAT
   ```

   The variant and sensitivity runners are listed in `trainer/README.md`.

4. **Analysis** (`result_analysis/`).
   * Run the frozen workflow in `TEST_SUPPORT_README.md` for the primary analysis.
   * Then run the `paper_*.py` scripts to produce the tables.

## Where each paper result comes from

| Paper item | Script | Output |
|-----------|--------|--------|
| Label statuses, label definition | `dataset/build_rule_based_labels.py` | `dataset/rule_based_labels.csv` |
| Cohorts, missingness, feature completeness | `data_analysis/EDA.ipynb` | `data_analysis/artifacts/` |
| Structural differences (Mann–Whitney, BH, rank-biserial *r*) | `data_analysis/analyze_graphs.py`, `compare_graphs.py` | `data_analysis/output/statistical_comparison.csv` |
| Primary analysis (9 frozen contrasts) | `result_analysis/gnn_primary_inference.py`; summary by `paper_tables.py --tasks primary` | `paper_tables/primary_contrasts.csv`, `primary_model_ap_by_window.csv` |
| XGBoost with positive-class weight | `trainer/run_xgb_spw_sensitivity.py`, then `gnn_primary_inference.py --input input_xgb_spw` | `paper_tables/primary_original_vs_xgb_scale_pos_weight.csv` |
| Primary contrasts against every non-graph model | `result_analysis/paper_rq2_all_baselines.py` | `paper_tables/rq2_all_baselines.csv`, `rq2_val_selected.csv` |
| Within-window comparison and statistical power | `paper_tables.py --tasks within power` | `paper_tables/within_window_*.csv`, `power_mde.csv` |
| Validation–test gap | `paper_split_gap.py`, `paper_valattest_perseed.py` | `paper_tables/split_gap_*.csv`, `split_shift_diagnostics.csv` |
| UMAP of the chronological split | `data_analysis/EDA_multivariate.ipynb` | `artifacts_stratified/figures/multivariate_eda/07_chronological_split_q*.png` |
| Stratified split | `EDA_stratified.ipynb` + `RUGPULL_SPLIT=stratified` training | `trainer/results_stratified/` |
| Feature views, graph-only GNNs | `paper_tables.py --tasks views`, `paper_ablations.py --tasks graph_only` | `paper_tables/feature_view_ablation.csv`, `graph_only_vs_official_gnn.csv` |
| Missingness indicators and cohorts | `paper_ablations.py --tasks cleaning` | `paper_tables/gnn_cleaning_comparisons.csv`, `cleaning_summary_for_paper.csv` |
| Model variants | `trainer/run_gnn_improved.py`, `run_gnn_stacked.py`, `run_gnn_global_ablation.py`, `run_hnn4rp.py`; `paper_ablations.py --tasks variants` | `paper_tables/gnn_variants_vs_official.csv` |

`paper_tables/` stands for `result_analysis/output/test/paper_tables/`. These CSV files are
committed, so every number in the results tables can be checked without retraining.

## Data availability

**In the repository:**

* the code;
* token identifiers and rule-based labels (`dataset/rule_based_labels.csv`);
* token-level features (`dataset/graph_nodes.csv`);
* per-token graph-summary features (`data_analysis/data/q*/model_graph_features_with_split.parquet`).

**Not included:**

* edge-level transfer data (`graph_edges.csv`);
* the processed artifacts and per-run predictions under `trainer/results*/`;
* the external SolRPDS dataset, which can be obtained from its original source.

Those directories are listed in `.gitignore`.

## Environment

* Python 3.10 or later (tested with 3.10 and 3.13).
* The analysis dependencies are pinned in `result_analysis/requirements-lock.txt`.
* Training also needs PyTorch, PyTorch Geometric, scikit-learn, XGBoost and Optuna.
* FT-Transformer and AutoInt are implemented in `trainer/baselines/models/`.

## Responsible use

Labels follow an operational definition:

* A positive label is not a legal judgment or proof of fraud.
* A negative label does not certify that a token is safe.

Predictions are early-warning signals for review. They are not a basis for accusing projects,
blacklisting addresses, or making investment decisions.
