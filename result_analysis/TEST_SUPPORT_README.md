# Validation and Test comparison workflow

Start in `result_analysis/`. Legacy Validation invocation and output paths remain unchanged.

## Inputs
Put `final_predictions_<experiment>.csv` from the trainer under:
- `input/baseline_full/`
- `input/baseline_panel/`

Existing `predictions_<experiment>.csv` are also supported as fallback, but files must actually contain `partition=test` rows. Preferred naming on Test is `final_predictions_*`. Do not rename an unrelated Validation-only CSV and claim it is Test.

## Test analysis

```bash
python batch_analysis.py --partition test --scope both --models all --qs all --reps 1000
python cross_model_bootstrap.py --partition test --manifest planned_cross_model_pairs.csv --reps 10000
python run_analysis.py --partition test --allow-other-counts --include-model-comparison
```

`run_analysis.py` expects the default 1000-bootstrap batch output. In Test mode all reports/figures are placed under `output/test/`. The `--allow-other-counts` switch is useful for incomplete test sets but the requested illustration-specific figures can still require certain comparisons. Cross-model is optional; when its result is absent, its figure is skipped.

## Individual comparisons

```bash
python paired_bootstrap.py --partition test --scope panel --a <experiment-a-stem> --b <experiment-b-stem> --reps 10000
python model_comparison.py --partition test --scope both
```

## Validation, unchanged

```bash
python batch_analysis.py --partition val --scope both --models all --qs all --reps 1000
python run_analysis.py --partition val --include-model-comparison
```

## Important limitations
- These scripts analyze **baseline** prediction files. They do not yet merge GNN predictions in cross-family analyses.
- No Test labels are used for Optuna or threshold selection. Freeze model configurations and comparisons before Test inspection.
- Pairwise comparisons require shared token IDs, consistent labels, and five complete seeds. No sample counts or CI values are guaranteed until Test output arrives.
- The bootstrap CIs are unadjusted and many analyses share token IDs; don't interpret exploratory counts as independent statistical tests.
- `audit.py` checks original `runs_*` artifacts and is **not** automatically converted to final predictions: use it separately if corresponding trainer runs/sweep/gap files are available.
