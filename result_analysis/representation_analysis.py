"""Validation-only, descriptive analysis of saved representation paired-bootstrap results.

Reads the batch comparison_results.csv; never retrains or recomputes bootstrap.
Delta = metric(view A) - metric(view B). Bootstrap percentile CIs are unadjusted.
The node_graph_vs_node comparison is paired only on graph-bearing tokens; its
reported difference is NOT the Full-population performance improvement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / 'output' / 'batch' / 'reps_1000_seed_2026' / 'comparison_results.csv'
DEFAULT_OUTPUT = ROOT / 'output' / 'representation'
PAIRS = {
    'node_graph_vs_node_matched': ('node_graph', 'node_matched'),
    'node_graph_vs_node': ('node_graph', 'node'),
    'node_graph_freeze_vs_node_graph': ('node_graph_freeze', 'node_graph'),
    'graph_vs_node_matched': ('graph', 'node_matched'),
}
REQUIRED = {
    'family', 'comparison', 'scope', 'model', 'q', 'cohort', 'view', 'variant',
    'experiment_a', 'experiment_b', 'a_n_tokens', 'b_n_tokens',
    'a_prevalence', 'b_prevalence', 'n_paired', 'n_positive', 'n_negative',
    'n_seeds', 'n_bootstrap', 'ap_a', 'ap_b', 'delta_ap',
    'roc_a', 'roc_b', 'delta_roc',
    'delta_ap_ci_low', 'delta_ap_ci_high',
    'delta_roc_ci_low', 'delta_roc_ci_high',
}
NUMERIC = [
    'a_n_tokens', 'b_n_tokens', 'a_prevalence', 'b_prevalence',
    'n_paired', 'n_positive', 'n_negative', 'n_seeds', 'n_bootstrap',
    'ap_a', 'ap_b', 'delta_ap', 'roc_a', 'roc_b', 'delta_roc',
    'delta_ap_ci_low', 'delta_ap_ci_high', 'delta_roc_ci_low', 'delta_roc_ci_high',
]


def validated_rows(file: Path) -> pd.DataFrame:
    data = pd.read_csv(file)
    absent = REQUIRED - set(data.columns)
    if absent:
        raise ValueError(f'Missing columns: {sorted(absent)}')
    df = data.loc[data.family == 'representation'].copy()
    if df.empty:
        raise ValueError('No representation comparisons were found')
    if df[list(REQUIRED)].isna().any().any():
        raise ValueError('Null values in required representation columns')
    if not df.comparison.isin(PAIRS).all():
        raise ValueError('Unexpected comparison orientation')
    if not df.scope.isin(['full', 'panel']).all() or not df.q.isin(['q1','q2','q3']).all():
        raise ValueError('Unexpected scope or observation window')
    if not df.cohort.isin(['original','drop_3_or_more_missing','complete_case']).all():
        raise ValueError('Unexpected cohort')
    if not df.variant.isin(['no_mask','with_mask']).all():
        raise ValueError('Unexpected variant')
    if ((df.cohort == 'complete_case') & (df.variant != 'no_mask')).any():
        raise ValueError('Unexpected complete_case with_mask comparison')
    key = ['scope','q','model','cohort','variant','comparison']
    if df.duplicated(key).any():
        raise ValueError('Duplicate representation configurations')
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors='raise')
    if not np.isfinite(df[NUMERIC].to_numpy(dtype=float)).all():
        raise ValueError('Non-finite numeric values')
    if ((df.n_positive + df.n_negative) != df.n_paired).any():
        raise ValueError('Inconsistent paired class counts')
    if (df[['n_positive','n_negative','n_seeds','n_bootstrap']] <= 0).any().any():
        raise ValueError('Invalid class counts, seeds or repetitions')
    if (df.n_paired > df[['a_n_tokens','b_n_tokens']].min(axis=1)).any():
        raise ValueError('Paired N exceeds the available prediction sets')
    for metric in ['ap','roc']:
        if not np.allclose(df[f'{metric}_a'] - df[f'{metric}_b'],df[f'delta_{metric}'],rtol=1e-9,atol=1e-9):
            raise ValueError(f'Inconsistent delta {metric}')
        if (df[f'delta_{metric}_ci_low'] > df[f'delta_{metric}_ci_high']).any():
            raise ValueError(f'Reversed {metric} CI')
    for pair,(a,b) in PAIRS.items():
        sub = df.loc[df.comparison == pair]
        if not (sub.view == a).all():
            raise ValueError(f'{pair}: incorrect feature-view identifier')
        # Check full experiment names without relying on their ambiguous underscore splitting.
        ea = [f'{r.model}_{r.scope}_{r.q}_{a}_{r.variant}_{r.cohort}' for r in sub.itertuples()]
        eb = [f'{r.model}_{r.scope}_{r.q}_{b}_{r.variant}_{r.cohort}' for r in sub.itertuples()]
        if not (sub.experiment_a.to_numpy() == np.asarray(ea)).all():
            raise ValueError(f'{pair}: experiment A does not match metadata')
        if not (sub.experiment_b.to_numpy() == np.asarray(eb)).all():
            raise ValueError(f'{pair}: experiment B does not match metadata')
    return df.sort_values(key,kind='stable').reset_index(drop=True)


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for metric in ['ap','roc']:
        delta = df[f'delta_{metric}']
        lo = df[f'delta_{metric}_ci_low']
        hi = df[f'delta_{metric}_ci_high']
        df[f'{metric}_point_direction'] = np.select(
            [delta > 1e-12,delta < -1e-12],['view_a_better','view_b_better'],default='tie')
        df[f'{metric}_ci_direction'] = np.select(
            [lo > 0,hi < 0],['view_a_better','view_b_better'],default='includes_zero')
    df['same_population'] = (df.a_n_tokens == df.n_paired) & (df.b_n_tokens == df.n_paired)
    df['matched_subset_only'] = ~df.same_population
    df['low_n_paired_lt_50'] = df.n_paired < 50
    return df


def summarize(df: pd.DataFrame, grouping: list[str]) -> pd.DataFrame:
    records = []
    for group, sub in df.groupby(grouping,dropna=False,sort=True):
        if not isinstance(group,tuple): group = (group,)
        row = dict(zip(grouping,group))
        row.update(
            n_configurations=len(sub),
            n_models=sub.model.nunique(),
            n_paired_min=int(sub.n_paired.min()),
            n_paired_max=int(sub.n_paired.max()),
            n_subset_only=int(sub.matched_subset_only.sum()),
            n_low_paired=int(sub.low_n_paired_lt_50.sum()),
            median_delta_ap=float(sub.delta_ap.median()),
            mean_delta_ap_descriptive=float(sub.delta_ap.mean()),
            median_delta_roc=float(sub.delta_roc.median()),
            ap_a_point=int((sub.ap_point_direction == 'view_a_better').sum()),
            ap_b_point=int((sub.ap_point_direction == 'view_b_better').sum()),
            ap_tie=int((sub.ap_point_direction == 'tie').sum()),
            ap_ci_a=int((sub.ap_ci_direction == 'view_a_better').sum()),
            ap_ci_b=int((sub.ap_ci_direction == 'view_b_better').sum()),
            ap_ci_includes_zero=int((sub.ap_ci_direction == 'includes_zero').sum()),
            roc_ci_a=int((sub.roc_ci_direction == 'view_a_better').sum()),
            roc_ci_b=int((sub.roc_ci_direction == 'view_b_better').sum()),
            roc_ci_includes_zero=int((sub.roc_ci_direction == 'includes_zero').sum()),
        )
        records.append(row)
    return pd.DataFrame(records)


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=DEFAULT_INPUT)
    p.add_argument('--out',type=Path,default=DEFAULT_OUTPUT)
    args=p.parse_args()
    df=annotate(validated_rows(args.input))
    args.out.mkdir(parents=True,exist_ok=True)
    df.to_csv(args.out/'representation_comparisons_annotated.csv',index=False)
    summarize(df,['scope','comparison']).to_csv(args.out/'summary_by_scope_pair.csv',index=False)
    summarize(df,['scope','q','comparison']).to_csv(args.out/'summary_by_window.csv',index=False)
    summarize(df,['scope','cohort','comparison']).to_csv(args.out/'summary_by_cohort.csv',index=False)
    summarize(df,['scope','model','comparison']).to_csv(args.out/'summary_by_model.csv',index=False)
    summarize(df,['scope','q','model','cohort','variant','comparison']).to_csv(
        args.out/'configuration_level.csv',index=False)
    notes={
        'source':str(args.input.resolve()),
        'n_comparisons':len(df),
        'delta_orientation':'metric(view A)-metric(view B); positive favors view A',
        'population_warning':('All reported deltas are paired on the shared tokens. '
            'The node_graph vs node baseline is NOT a comparison across the whole node-only population.'),
        'independence_warning':'Overlapping configurations, models and cohorts are NOT independent hypothesis tests.',
        'multiple_comparisons':'Percentile confidence intervals are exploratory and unadjusted.',
        'scope_warning':'Full and Panel refer to different eligibility populations; do not pool their effects.',
        'n_flag':'n_paired < 50 is flagged for sensitivity review, not automatically excluded.',
        'interpretation':'Do not infer causal value of graph topology or superiority of GNN from feature-view ablations.',
    }
    (args.out/'README_analysis.json').write_text(json.dumps(notes,indent=2),encoding='utf-8')
    print(f'Validated {len(df)} representation comparisons.')
    print(summarize(df,['scope','comparison']).to_string(index=False))
    print(f'Saved 7 files to {args.out}')

if __name__ == '__main__':
    main()
