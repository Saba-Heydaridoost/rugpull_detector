"""Audit and describe validation-only matched cleaning/missingness comparisons.

Reads the existing batch comparison_results.csv. Does not re-run bootstrap or access
raw predictions. Delta = metric(A) - metric(B); 95% CIs are *unadjusted*.

mask_effect: A=with_mask, B=no_mask, SAME cohort and evaluation population.
cohort_matched: A=filtered cohort, B=original, evaluated on shared tokens;
  models were trained with different cohorts, so this is a matched-subset
  training-policy comparison, NOT an isolated test of missing-data deletion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / 'output' / 'batch' / 'reps_1000_seed_2026' / 'comparison_results.csv'
DEFAULT_OUTPUT = ROOT / 'output' / 'cleaning_summary'
COHORTS_FILTERED = {'drop_3_or_more_missing', 'complete_case'}
REQUIRED = {
    'family','comparison','model','q','cohort','view','variant','scope',
    'experiment_a','experiment_b','a_n_tokens','b_n_tokens',
    'a_prevalence','b_prevalence','n_paired','n_positive','n_negative',
    'n_seeds','n_bootstrap','ap_a','ap_b','delta_ap','roc_a','roc_b',
    'delta_roc','delta_ap_ci_low','delta_ap_ci_high',
    'delta_roc_ci_low','delta_roc_ci_high'
}
METRICS = ('ap', 'roc')
KEYS = ['scope','comparison','q','model','cohort','view','variant']


def read_validate(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED.difference(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    t = df[df['family'].eq('cleaning')].copy()
    if t.empty:
        raise ValueError('No cleaning rows in input')
    if t[list(REQUIRED)].isna().any().any():
        raise ValueError('Cleaning rows contain missing required fields')
    if t.duplicated(KEYS).any():
        raise ValueError('Duplicate cleaning comparisons')
    if not t['scope'].isin(['full','panel']).all():
        raise ValueError('Unexpected scope')
    if not t['q'].isin(['q1','q2','q3']).all():
        raise ValueError('Unexpected window')
    if not t['comparison'].isin(['mask_effect','cohort_matched']).all():
        raise ValueError('Unexpected cleaning comparison')
    numeric = ['a_n_tokens','b_n_tokens','a_prevalence','b_prevalence',
               'n_paired','n_positive','n_negative','n_seeds','n_bootstrap',
               'ap_a','ap_b','delta_ap','roc_a','roc_b','delta_roc',
               'delta_ap_ci_low','delta_ap_ci_high',
               'delta_roc_ci_low','delta_roc_ci_high']
    t[numeric] = t[numeric].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(t[numeric].to_numpy(dtype=float)).all():
        raise ValueError('Non-finite numeric values')
    if (t.n_positive + t.n_negative != t.n_paired).any():
        raise ValueError('Inconsistent paired class counts')
    if (t[['n_positive','n_negative','n_seeds','n_bootstrap']] <= 0).any().any():
        raise ValueError('Invalid counts')
    if (t.n_paired > t[['a_n_tokens','b_n_tokens']].min(axis=1)).any():
        raise ValueError('Paired sample count exceeds a parent sample size')
    if ((t[['a_prevalence','b_prevalence']] < 0) | (t[['a_prevalence','b_prevalence']] > 1)).any().any():
        raise ValueError('Prevalence outside [0,1]')
    for m in METRICS:
        if not np.allclose(t[f'{m}_a']-t[f'{m}_b'],t[f'delta_{m}'],atol=1e-9,rtol=1e-9):
            raise ValueError(f'Incorrect {m} delta direction')
        if (t[f'delta_{m}_ci_low'] > t[f'delta_{m}_ci_high']).any():
            raise ValueError(f'Reversed {m} interval')

    mask = t[t.comparison.eq('mask_effect')]
    if not mask.variant.eq('with_mask').all() or not mask.cohort.isin(['original','drop_3_or_more_missing']).all():
        raise ValueError('Unexpected mask-effect configuration')
    if not ((mask.a_n_tokens == mask.b_n_tokens) &
            (mask.n_paired == mask.a_n_tokens) &
            np.isclose(mask.a_prevalence, mask.b_prevalence)).all():
        raise ValueError('mask_effect must compare identical evaluation token sets and labels')
    if not mask.apply(lambda r: r.experiment_a.endswith(f'_with_mask_{r.cohort}') and
                      r.experiment_b.endswith(f'_no_mask_{r.cohort}'),axis=1).all():
        raise ValueError('Invalid mask_effect experiment identity')

    cohort = t[t.comparison.eq('cohort_matched')]
    if not cohort.variant.eq('no_mask').all() or not cohort.cohort.isin(COHORTS_FILTERED).all():
        raise ValueError('Unexpected cohort comparison configuration')
    if not cohort.apply(lambda r: r.experiment_a.endswith(f'_no_mask_{r.cohort}') and
                        r.experiment_b.endswith('_no_mask_original'),axis=1).all():
        raise ValueError('Invalid cohort experiment identities')
    if not (cohort.a_n_tokens == cohort.n_paired).all():
        raise ValueError('Filtered cohort must be the exact matched evaluation subset')
    if not (cohort.b_n_tokens >= cohort.a_n_tokens).all():
        raise ValueError('Unexpected parent cohort sizes')
    return t.sort_values(KEYS,kind='stable').reset_index(drop=True)


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    t=df.copy()
    for m in METRICS:
        d=t[f'delta_{m}']
        t[f'{m}_point_direction']=np.select([d>1e-12,d< -1e-12],
                                             ['A_better','B_better'],default='tie')
        t[f'{m}_ci_direction']=np.select(
            [t[f'delta_{m}_ci_low']>0,t[f'delta_{m}_ci_high']<0],
            ['A_better','B_better'],default='includes_zero')
    t['low_n_paired_lt_50']=t.n_paired<50
    t['parent_population_differs']=t.a_n_tokens.ne(t.b_n_tokens)
    t['comparison_interpretation']=np.where(t.comparison.eq('mask_effect'),
        'mask_vs_no_mask_same_cohort',
        'different_training_cohorts_matched_evaluation_subset')
    return t


def summary(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows=[]
    for grp, g in df.groupby(by, dropna=False, sort=True):
        if not isinstance(grp,tuple): grp=(grp,)
        row=dict(zip(by,grp))
        row.update(n_configurations=len(g), n_models=g.model.nunique(),
                   n_paired_min=int(g.n_paired.min()),n_paired_max=int(g.n_paired.max()),
                   n_low_paired=int(g.low_n_paired_lt_50.sum()),
                   n_parent_population_differs=int(g.parent_population_differs.sum()),
                   median_delta_ap=float(g.delta_ap.median()),
                   mean_delta_ap_descriptive=float(g.delta_ap.mean()),
                   median_delta_roc=float(g.delta_roc.median()))
        for m in METRICS:
            for value in ('A_better','B_better','tie'):
                row[f'{m}_point_{value}']=int(g[f'{m}_point_direction'].eq(value).sum())
            for value in ('A_better','B_better','includes_zero'):
                row[f'{m}_ci_{value}']=int(g[f'{m}_ci_direction'].eq(value).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,default=DEFAULT_INPUT)
    ap.add_argument('--out',type=Path,default=DEFAULT_OUTPUT)
    args=ap.parse_args()
    t=annotate(read_validate(args.input))
    args.out.mkdir(parents=True,exist_ok=True)
    outputs={
        'cleaning_comparisons_annotated.csv':t,
        'summary_by_scope_and_comparison.csv':summary(t,['scope','comparison']),
        'summary_by_cohort.csv':summary(t,['scope','comparison','cohort']),
        'summary_by_window.csv':summary(t,['scope','comparison','q']),
        'summary_by_model.csv':summary(t,['scope','comparison','model']),
        'summary_by_view.csv':summary(t,['scope','comparison','view']),
    }
    for name, frame in outputs.items(): frame.to_csv(args.out/name,index=False)
    note={
        'source':str(args.input.resolve()),'n_cleaning_comparisons':len(t),
        'delta_definition':'metric(A) - metric(B); positive favors A',
        'mask_effect':'A with_mask vs B no_mask, same cohort and matched evaluation samples',
        'cohort_matched':'A filtered-cohort model vs B original-cohort model, evaluated on filtered shared tokens; cannot isolate effects of removing incomplete records from changes in training set',
        'prevalence_warning':'a_prevalence/b_prevalence are original parent-cohort prevalences, not necessarily the matched comparison prevalence; paired prevalence = n_positive/n_paired',
        'bootstrap_intervals':'Unadjusted exploratory 95% confidence intervals; no family-wise or FDR adjustment',
        'dependence':'Comparisons share tokens and model predictions; counts of directions or intervals are NOT independent hypothesis tests',
        'selection':'Validation is exploratory; Test is held-out only if model/comparison choices were frozen before inspection',
        'small_sample':'n_paired<50 is flagged; no comparisons are silently discarded'
    }
    (args.out/'README_analysis.json').write_text(json.dumps(note,indent=2),encoding='utf-8')
    print(f'Validated {len(t)} cleaning comparisons.')
    print(outputs['summary_by_scope_and_comparison.csv'].to_string(index=False))
    print(f'Saved {len(outputs)+1} outputs to {args.out}')

if __name__=='__main__': main()
