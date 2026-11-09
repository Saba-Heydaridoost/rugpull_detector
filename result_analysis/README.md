# result_analysis/

This folder holds the statistical analysis of the saved predictions. Nothing here retrains a model.
Run every script from this folder.

* **Inputs.**
  * Baseline run files under `input/baseline_<scope>/`.
  * Per-seed predictions in `../trainer/results*/`.
* **Outputs.** Written to `output/<partition>/`.

The analysis has two parts:

1. the **frozen workflow**, fixed before the test set was scored;
2. the **paper scripts**, which compute the exploratory tables of the paper.

## 1. Frozen workflow (validation, then a single test run)

`TEST_SUPPORT_README.md` gives the exact command sequence and the rules for using the test
partition.

| Script | Role |
|--------|------|
| `gnn_primary_inference.py` | **Primary analysis.** Nine predeclared contrasts: GCN / GraphSAGE / GAT at `q1` minus XGBoost at `q1`, `q2` and `q3` (panel, original cohort, `no_mask`, `node_graph`). The endpoint is the mean per-seed AP difference, with a paired, label-stratified token bootstrap (10,000 resamples, RNG seed 2026) and Bonferroni simultaneous intervals. |
| `gnn_baseline_comparison.py`, `gnn_baseline_pairs.csv` | Secondary GNN-vs-baseline contrasts on the matched longitudinal panel (unadjusted). |
| `gnn_baseline_curves.py` | Performance across windows on a fixed cohort (descriptive). |
| `gnn_preflight.py` | Fail-fast audit of all inputs before inference. |
| `gnn_panel_utils.py` | Construction of the fixed longitudinal panel. |
| `batch_analysis.py`, `paired_bootstrap.py`, `comparison_utils.py`, `prediction_io.py` | Paired token-level bootstrap engine and I/O. |
| `run_analysis.py`, `early_detection.py`, `representation_analysis.py`, `cleaning_analysis.py`, `model_comparison.py`, `cross_model_bootstrap.py`, `planned_cross_model_pairs.csv`, `plotting.py` | Baseline analyses: window, feature view, cleaning and missingness, and cross-model comparisons. |
| `audit.py`, `load_results.py`, `config.py` | Provenance checks of the baseline runs. |

`output/val/` holds the validation-phase outputs that were produced before the freeze.

## 2. Paper tables (exploratory, test partition)

These scripts read existing results only. They write to `output/test/paper_tables/`, whose CSV
files are committed.

| Script | Tasks | Paper item | Output |
|--------|-------|-----------|--------|
| `paper_tables.py` | `primary` | Primary-analysis table | `primary_contrasts.csv`, `primary_model_ap_by_window.csv` |
| | `within` | Within-window GNN vs non-graph table (Holm-adjusted) | `within_window_models.csv`, `within_window_contrasts.csv` |
| | `power` | Minimum detectable ΔAP at 80 % power | `power_mde.csv` |
| | `views` | Feature-view ablation | `feature_view_ablation.csv` |
| | `liqcut` | Check of the liquidity snapshot at the cutoff | `liquidity_cutoff_snapshot_check.csv` |
| `paper_rq2_all_baselines.py` | | Primary contrasts against every non-graph model (72 contrasts, Holm) and against the validation-selected one (Bonferroni) | `rq2_all_baselines.csv`, `rq2_val_selected.csv` |
| `paper_ablations.py` | `graph_only` | Graph-only GNNs | `graph_only_vs_official_gnn.csv` |
| | `cleaning` | Missingness indicators and cohorts | `gnn_cleaning_comparisons.csv`, `cleaning_summary_for_paper.csv` |
| | `variants` | Model-variants table: DirSAGE, DirGCN, global graph, bagged GraphSAGE, stacking, HNN4RP | `gnn_variants_vs_official.csv` |
| `paper_split_gap.py` | | Validation–test gap: train/val/test scores, selection optimism, prevalence and population shift | `split_gap_models.csv`, `split_gap_selection.csv`, `split_shift_diagnostics.csv` |
| `paper_valattest_perseed.py` | | Validation AP re-weighted to the test prevalence, per seed (the Val@test column) | `split_gap_valattest_perseed.csv` |
| `gnn_global_ablation_table.py` | | Detailed table for the global-graph ablation | `output/test/gnn_global_ablation/` |
| `hnn4rp_comparison.py [chronological\|stratified]` | | Detailed table for HNN4RP | `output/test/hnn4rp*/` |

**XGBoost with a positive-class weight.** This is the primary analysis re-run against the re-tuned
XGBoost from `trainer/run_xgb_spw_sensitivity.py`.

1. Build an input folder `input_xgb_spw/`.
   * Link `gnn_panel` to `input/gnn_panel`.
   * Copy `../trainer/results_xgb_spw/predictions_xgboost_panel_<q>_node_graph_no_mask_original.csv`
     to `baseline_panel/final_predictions_xgboost_panel_<q>_node_graph_no_mask_original.csv`.
2. Run:

   ```bash
   python gnn_primary_inference.py --partition test --input input_xgb_spw \
       --output output/test/paper_tables/primary_xgb_scale_pos_weight
   ```

`primary_original_vs_xgb_scale_pos_weight.csv` puts these contrasts next to the original ones.

Typical order once all trainer runs exist:

```bash
python paper_tables.py
python paper_rq2_all_baselines.py
python paper_ablations.py
python paper_split_gap.py
python paper_valattest_perseed.py
```

## Conventions

* **Bootstrap.** Resampling is paired and stratified by label over test tokens, with one draw
  shared by both models and all five seeds. The RNG seed is 2026.
* **Two estimators.**
  * The frozen primary analysis and its robustness checks use the *mean per-seed* AP difference.
  * The within-window and variant tables use the AP of the *seed-ensemble* score (the mean
    prediction over seeds).
  * The two estimators can differ.
* **Multiplicity.** Only the nine primary contrasts are confirmatory (Bonferroni). Every other
  family is exploratory and reported with Holm correction or unadjusted intervals, as each table
  states.
* **Interpretation.** Results are measured at fixed cutoffs; they are not per-token first-alert
  latencies.

`requirements-lock.txt` pins the analysis environment.
