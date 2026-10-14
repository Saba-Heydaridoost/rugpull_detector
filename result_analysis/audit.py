"""Independently verify baseline runs against prediction-level data."""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_score,
    recall_score, f1_score, brier_score_loss,
)
from config import SCOPE_DIRS, OUTPUT_DIR, EXPECTED_SEEDS, METRICS_TOL
from load_results import experiments, read_csv

REQUIRED_PRED = {'token_mint', 'seed', 'partition', 'y_true', 'y_score', 'y_pred', 'threshold'}
REQUIRED_RUN = {'seed', 'partition', 'roc_auc', 'pr_auc', 'precision', 'recall', 'f1', 'brier', 'threshold'}
METRICS = ('roc_auc', 'pr_auc', 'precision', 'recall', 'f1', 'brier')


def load_rerun_evidence(path):
    if not path.is_file():
        return {}
    df = pd.read_csv(path)
    required = {'experiment_key', 'seed', 'metric', 'reported_old_runs',
                'rerun_new', 'max_abs_probability_change'}
    if not required.issubset(df.columns):
        raise ValueError(f'Rerun evidence missing columns: {sorted(required - set(df.columns))}')
    return {(str(r.experiment_key), int(r.seed), str(r.metric)): r
            for r in df.itertuples(index=False)}



def add_issue(issues, scope, experiment, level, reason, **extra):
    issues.append(dict(scope=scope, experiment=experiment, level=level, reason=reason, **extra))


def compute_metrics(g):
    y = g['y_true'].to_numpy(dtype=int)
    s = g['y_score'].to_numpy(dtype=float)
    pred = g['y_pred'].to_numpy(dtype=int)
    return {
        'roc_auc': roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan,
        'pr_auc': average_precision_score(y, s),
        'precision': precision_score(y, pred, zero_division=0),
        'recall': recall_score(y, pred, zero_division=0),
        'f1': f1_score(y, pred, zero_division=0),
        'brier': brier_score_loss(y, s),
    }


def audit_scope(scope, folder, tol, rerun_evidence=None):
    rerun_evidence = rerun_evidence or {}
    issues, details = [], []
    exp_list = list(experiments(folder))
    summary_path = Path(folder) / f'summary_{scope}.csv'
    if not summary_path.exists():
        add_issue(issues, scope, '', 'ERROR', f'Missing {summary_path.name}')
    else:
        summary = read_csv(summary_path)
        if len(summary) != 2 * len(exp_list):
            add_issue(issues, scope, '', 'WARNING', f'Summary rows {len(summary)} vs expected {2 * len(exp_list)}')
    for i, exp in enumerate(exp_list, 1):
        missing = [p.name for p in (exp.predictions_path, exp.sweep_path, exp.gap_path) if not p.exists() or p.stat().st_size == 0]
        if missing:
            add_issue(issues, scope, exp.name, 'ERROR', f'Missing/empty files: {missing}')
            continue
        try:
            runs = read_csv(exp.runs_path)
            pred = read_csv(exp.predictions_path)
        except Exception as e:
            add_issue(issues, scope, exp.name, 'ERROR', f'CSV read failed: {e}')
            continue
        for label, frame, required in [('runs', runs, REQUIRED_RUN), ('predictions', pred, REQUIRED_PRED)]:
            absent = required - set(frame.columns)
            if absent:
                add_issue(issues, scope, exp.name, 'ERROR', f'{label} missing columns: {sorted(absent)}')
        if REQUIRED_RUN - set(runs.columns) or REQUIRED_PRED - set(pred.columns):
            continue
        if pred.duplicated(['token_mint', 'seed', 'partition']).any():
            add_issue(issues, scope, exp.name, 'ERROR', 'Duplicate token/seed/partition keys')
        if runs.duplicated(['seed', 'partition']).any():
            add_issue(issues, scope, exp.name, 'ERROR', 'Duplicate runs seed/partition keys')
        val_seeds = set(runs.loc[runs.partition == 'val', 'seed'].astype(int))
        if val_seeds != EXPECTED_SEEDS:
            add_issue(issues, scope, exp.name, 'WARNING', f'Expected val seeds {sorted(EXPECTED_SEEDS)}, found {sorted(val_seeds)}')
        if not pred['y_score'].between(0, 1).all() or not np.isfinite(pred['y_score']).all():
            add_issue(issues, scope, exp.name, 'ERROR', 'Invalid prediction probabilities')
            continue
        if not pred['y_true'].isin([0, 1]).all() or not pred['y_pred'].isin([0, 1]).all():
            add_issue(issues, scope, exp.name, 'ERROR', 'Nonbinary labels/predictions')
            continue
        calculated_pred = (pred.y_score.to_numpy() >= pred.threshold.to_numpy()).astype(int)
        mismatch_count = int(np.sum(calculated_pred != pred.y_pred.to_numpy(dtype=int)))
        if mismatch_count:
            add_issue(issues, scope, exp.name, 'ERROR', f'y_pred differs from score >= threshold in {mismatch_count} rows')
        cross_seed_y = pred.groupby(['partition', 'token_mint'])['y_true'].nunique()
        if (cross_seed_y > 1).any():
            add_issue(issues, scope, exp.name, 'ERROR', 'Different y_true across seeds for same token and partition')
        run_lookup = runs.set_index(['seed', 'partition'])
        pred_keys = set(pred.groupby(['seed', 'partition']).groups)
        run_keys = set(run_lookup.index)
        # Native trainer only exports token-level predictions for Validation; Train metrics remain in runs.
        required_keys = {(seed, 'val') for seed in EXPECTED_SEEDS}
        missing_val = required_keys - pred_keys
        unknown_pred = pred_keys - run_keys
        if missing_val or unknown_pred:
            add_issue(issues, scope, exp.name, 'ERROR', f'Validation predictions missing for {len(missing_val)} seeds; {len(unknown_pred)} unmatched prediction groups')
        for (seed, part), group in pred.groupby(['seed', 'partition'], sort=False):
            if (seed, part) not in run_keys:
                continue
            metrics = compute_metrics(group)
            row = run_lookup.loc[(seed, part)]
            for metric, computed in metrics.items():
                recorded = float(row[metric])
                delta = abs(computed - recorded) if np.isfinite(computed) and np.isfinite(recorded) else np.nan
                details.append(dict(scope=scope, experiment=exp.name, seed=seed, partition=part, metric=metric, reported=recorded, recomputed=computed, abs_error=delta, n=len(group)))
                if np.isfinite(delta) and delta > tol:
                    evidence = rerun_evidence.get((exp.name, int(seed), metric)) if scope == 'full' and part == 'val' else None
                    reproducible = (
                        evidence is not None
                        and metric in ('roc_auc', 'pr_auc')
                        and np.isfinite(evidence.rerun_new)
                        and np.isfinite(evidence.reported_old_runs)
                        and abs(float(evidence.reported_old_runs) - recorded) <= tol
                        and abs(float(evidence.rerun_new) - recorded) <= tol
                        and float(evidence.max_abs_probability_change) <= 1e-12
                    )
                    if reproducible:
                        add_issue(issues, scope, exp.name, 'WARNING',
                                  f'{metric} CSV-prediction mismatch seed={seed} partition={part}: '
                                  f'difference {delta:.8g}; independent rerun matches runs '
                                  f'(score drift <= 1e-12). Preserve this provenance warning.')
                    else:
                        add_issue(issues, scope, exp.name, 'ERROR',
                                  f'{metric} mismatch seed={seed} partition={part}: difference {delta:.8g}')
                if np.isnan(computed) != np.isnan(recorded):
                    add_issue(issues, scope, exp.name, 'ERROR', f'{metric} NaN mismatch seed={seed} partition={part}')
            if abs(float(row['threshold']) - float(group['threshold'].iloc[0])) > tol:
                add_issue(issues, scope, exp.name, 'ERROR', f'Run threshold != prediction threshold seed={seed} partition={part}')
        if i % 100 == 0:
            print(f'  {scope}: {i}/{len(exp_list)} experiments checked', flush=True)
    return issues, details, len(exp_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scope', choices=('full', 'panel', 'both'), default='both')
    parser.add_argument('--full-dir', type=Path, default=SCOPE_DIRS['full'])
    parser.add_argument('--panel-dir', type=Path, default=SCOPE_DIRS['panel'])
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--tolerance', type=float, default=METRICS_TOL)
    parser.add_argument('--rerun-evidence', type=Path,
                        default=OUTPUT_DIR / 'decision_tree_reruns' / 'comparison_metrics.csv',
                        help='Optional independently rerun validation metrics for flagged Decision Trees')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_issues, all_details = [], []
    rerun_evidence = load_rerun_evidence(args.rerun_evidence)
    if rerun_evidence:
        print(f'Loaded {len(rerun_evidence)} rerun metric checks from {args.rerun_evidence}', flush=True)
    for scope in (['full', 'panel'] if args.scope == 'both' else [args.scope]):
        folder = args.full_dir if scope == 'full' else args.panel_dir
        print(f'Checking {scope}: {folder}', flush=True)
        if not folder.is_dir():
            print(f'ERROR: Directory does not exist: {folder}', file=sys.stderr)
            sys.exit(2)
        issues, details, n = audit_scope(scope, folder, args.tolerance, rerun_evidence)
        all_issues.extend(issues)
        all_details.extend(details)
        print(f'  Found {n} runs files; issues: {len(issues)}', flush=True)
    pd.DataFrame(all_issues, columns=['scope', 'experiment', 'level', 'reason']).to_csv(args.output_dir / 'issues.csv', index=False)
    pd.DataFrame(all_details, columns=['scope', 'experiment', 'seed', 'partition', 'metric', 'reported', 'recomputed', 'abs_error', 'n']).to_csv(args.output_dir / 'metric_checks.csv', index=False)
    print(f'Audit reports: {args.output_dir}', flush=True)
    print(f'Errors: {sum(x["level"] == "ERROR" for x in all_issues)}; warnings: {sum(x["level"] == "WARNING" for x in all_issues)}')
    return 1 if any(x['level'] == 'ERROR' for x in all_issues) else 0

if __name__ == '__main__':
    sys.exit(main())
