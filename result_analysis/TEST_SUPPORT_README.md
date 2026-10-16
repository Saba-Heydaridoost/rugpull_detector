# Frozen Validation / Test evaluation workflow

Run commands from the `result_analysis/` project root. Do not use Test for training, hyperparameter selection, threshold selection, model/manifest changes, or pipeline debugging decisions. Keep training-run provenance and split identifiers separately.

## Inputs
- `input/baseline_full/`, `input/baseline_panel/`, `input/gnn_full/`, `input/gnn_panel/`.
- Validation is read from `predictions_<stem>.csv` with `partition=val`.
- **Test is read only from `final_predictions_<stem>.csv` with `partition=test`.** Do not merely rename an older file.
- Freeze input hashes and the Git revision before running final Test analyses. Preflight is mandatory for the GNN-vs-baseline analysis, not a verification of Trainer split provenance.

## Validation, before Test
```
python run_analysis.py --partition val --include-model-comparison
python gnn_preflight.py --partition val
python gnn_baseline_comparison.py --partition val --matching longitudinal --reps 1000 --seed 2026
python gnn_baseline_curves.py --partition val --reps 1000 --seed 2026
python gnn_primary_inference.py --partition val --reps 1000 --seed 2026
```
GNN commands require real Validation GNN predictions. A missing-file failure is expected until they arrive.

## Frozen Test workflow
First validate Trainer configs and dataset/split provenance externally. Then run:
```
python gnn_preflight.py --partition test
python batch_analysis.py --partition test --scope both --models all --qs all --reps 1000 --rng-seed 2026
python cross_model_bootstrap.py --partition test --manifest planned_cross_model_pairs.csv --reps 10000
python run_analysis.py --partition test --include-model-comparison
python gnn_baseline_comparison.py --partition test --matching longitudinal --reps 10000 --seed 2026
python gnn_baseline_curves.py --partition test --reps 10000 --seed 2026
python gnn_primary_inference.py --partition test --reps 10000 --seed 2026
```
**Do not** use `--allow-other-counts`, `--max-jobs`, or silently remove manifest rows for a final Test analysis. A failure means fix the pipeline/provenance and explicitly document the reason before further Test interpretation.

## Primary scientific interpretation
- Nine frozen AP contrasts: q1 each of GCN/GraphSAGE/GAT versus q1/q2/q3 XGBoost (`node_graph`, panel, original, no_mask); 10k paired token-stratified bootstrap replicates, Bonferroni simultaneous two-sided CIs across nine contrasts.
- A positive adjusted lower bound supports AP superiority subject to bootstrap assumptions; no lower-bound crossing gives an inconclusive result, never equivalence.
- All 72 manifest comparisons are secondary/exploratory, with unadjusted CIs. Pointwise temporal-curve CIs are descriptive and not pairwise-difference intervals.
- Comparisons quantify performance at three fixed cutoff times, NOT per-token first-alert latency. Six-file intersection is GNN-pair specific and may differ between GNNs.
- `audit.py` is separate Trainer-run provenance checking; it cannot infer fixed hyperparameters or absence of leakage from prediction CSVs.
