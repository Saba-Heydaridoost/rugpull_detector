"""Batch Validation-only paired analysis of saved baseline predictions.

No retraining and NO Test access. Resumes previously computed comparisons.
Uses the exact paired token-level bootstrap in paired_bootstrap.py.

Examples from result_analysis/:
  python batch_analysis.py --scope panel --models gradient_boosting --qs q1 --reps 100 --max-jobs 3
  python batch_analysis.py --scope both --models all --qs all --reps 1000
  python batch_analysis.py --scope both --models all --qs all --reps 10000 --families representation temporal
"""
from __future__ import annotations
import argparse
import csv
import itertools
from pathlib import Path
import sys
import pandas as pd
from comparison_utils import compare, load, coverage, stem, ROOT

MODELS = ('dummy', 'logistic_regression', 'decision_tree', 'random_forest',
          'gradient_boosting', 'xgboost', 'mlp', 'ft_transformer', 'autoint')
QS = ('q1', 'q2', 'q3')
COHORTS = ('original', 'drop_3_or_more_missing', 'complete_case')
VIEWS = ('node', 'graph', 'node_matched', 'node_graph', 'node_graph_freeze')

# A-B orientation is fixed, so delta > 0 means better A on the SAME tokens.
REPRESENTATION = (('node_graph', 'node_matched'),
                  ('node_graph', 'node'),
                  ('node_graph_freeze', 'node_graph'),
                  ('graph', 'node_matched'))
TEMPORAL = (('q1', 'q2'), ('q1', 'q3'), ('q2', 'q3'))


def valid_variants(cohort):
    return ('no_mask',) if cohort == 'complete_case' else ('no_mask', 'with_mask')


def generate_jobs(scopes, models, qs, families):
    for scope, model, q in itertools.product(scopes, models, qs):
        for cohort in COHORTS:
            for view in VIEWS:
                if 'cleaning' in families:
                    if cohort != 'complete_case':
                        a = stem(model, scope, q, view, 'with_mask', cohort)
                        b = stem(model, scope, q, view, 'no_mask', cohort)
                        yield 'cleaning', 'mask_effect', scope, model, q, cohort, view, 'with_mask', a, b
                    if cohort != 'original':
                        a = stem(model, scope, q, view, 'no_mask', cohort)
                        b = stem(model, scope, q, view, 'no_mask', 'original')
                        yield 'cleaning', 'cohort_matched', scope, model, q, cohort, view, 'no_mask', a, b
            if 'representation' in families:
                for variant in valid_variants(cohort):
                    for va, vb in REPRESENTATION:
                        a = stem(model, scope, q, va, variant, cohort)
                        b = stem(model, scope, q, vb, variant, cohort)
                        yield 'representation', f'{va}_vs_{vb}', scope, model, q, cohort, va, variant, a, b
    if 'temporal' in families and 'panel' in scopes:
        # Compare the same view, cohort, mask, model and *shared token IDs* across windows.
        for model, cohort, view in itertools.product(models, COHORTS, VIEWS):
            for variant in valid_variants(cohort):
                for qa, qb in TEMPORAL:
                    if qa not in qs or qb not in qs:
                        continue
                    a = stem(model, 'panel', qa, view, variant, cohort)
                    b = stem(model, 'panel', qb, view, variant, cohort)
                    yield 'temporal', f'{qa}_vs_{qb}', 'panel', model, f'{qa}_{qb}', cohort, view, variant, a, b


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--partition', choices=['val','test'], default='val')
    parser.add_argument('--scope', choices=['full', 'panel', 'both'], default='panel')
    parser.add_argument('--models', nargs='+', default=['gradient_boosting'],
                        help='Names or all')
    parser.add_argument('--qs', nargs='+', default=['q1'], help='q1 q2 q3, or all')
    parser.add_argument('--families', nargs='+', default=['cleaning', 'representation', 'temporal'],
                        choices=['cleaning','representation','temporal'])
    parser.add_argument('--reps', type=int, default=1000)
    parser.add_argument('--rng-seed', type=int, default=2026)
    parser.add_argument('--max-jobs', type=int, default=None,
                        help='Limit NEW attempts for a smoke test (including skips)')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--force', action='store_true', help='Recalculate existing comparisons')
    args = parser.parse_args()
    models = MODELS if args.models == ['all'] else tuple(args.models)
    qs = QS if args.qs == ['all'] else tuple(args.qs)
    if set(models) - set(MODELS): parser.error('Unknown model(s)')
    if set(qs) - set(QS): parser.error('Unknown window(s)')
    if args.reps < 100: parser.error('--reps must be >=100')
    scopes = ('full', 'panel') if args.scope == 'both' else (args.scope,)
    families = set(args.families)
    jobs = list(generate_jobs(scopes, models, qs, families))
    print(f'Planned: {len(jobs)} comparisons; scopes={scopes}, models={len(models)}, qs={qs}, reps={args.reps}', flush=True)
    if args.dry_run:
        for job in jobs[:12]: print('  ', job[0], job[1], job[-2], 'VS', job[-1])
        print('Dry run: no predictions read, no outputs written.')
        return
    out = ROOT/'output'/'batch' if args.partition == 'val' else ROOT/'output'/'test'/'batch'
    out.mkdir(parents=True, exist_ok=True)
    # Distinct run folder per sampling configuration to avoid mixing inference runs.
    folder = out/f'reps_{args.reps}_seed_{args.rng_seed}'
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder/'comparison_results.csv'
    skipped_file = folder/'skipped.csv'
    keycols=['family','comparison','experiment_a','experiment_b']
    old = pd.read_csv(manifest).to_dict('records') if manifest.exists() else []
    done = {(str(r['family']),str(r['comparison']),str(r['experiment_a']),str(r['experiment_b'])) for r in old}
    skipped = pd.read_csv(skipped_file).to_dict('records') if skipped_file.exists() else []
    known_skips = {(str(r['family']),str(r['comparison']),str(r['experiment_a']),str(r['experiment_b'])) for r in skipped}
    attempts=0; successes=0; unchanged=0
    for family, comparison, scope, model, q, cohort, view, variant, a, b in jobs:
        key = family,comparison,a,b
        if not args.force and key in done:
            unchanged += 1
            continue
        if args.max_jobs is not None and attempts>=args.max_jobs:
            break
        attempts+=1
        print(f'[{attempts}] {family} {a} VS {b}',flush=True)
        try:
            result=compare(scope,a,b,args.reps,args.rng_seed,args.partition)
            old = [r for r in old if (r['family'],r['comparison'],r['experiment_a'],r['experiment_b']) != key]
            old.append({'family':family,'comparison':comparison,'model':model,'q':q,
                        'cohort':cohort,'view':view,'variant':variant,**result})
            done.add(key)
            skipped = [
                r for r in skipped
                if (r['family'], r['comparison'], r['experiment_a'], r['experiment_b']) != key
            ]
            known_skips.discard(key)
            successes+=1
            print(f'  ΔAP={result["delta_ap"]:+.5f} CI=[{result["delta_ap_ci_low"]:+.5f},{result["delta_ap_ci_high"]:+.5f}]',flush=True)
        except (FileNotFoundError,ValueError) as exc:
            skipped=[r for r in skipped if (r['family'],r['comparison'],r['experiment_a'],r['experiment_b']) != key]
            skipped.append({'family':family,'comparison':comparison,'experiment_a':a,'experiment_b':b,'reason':str(exc)})
            known_skips.add(key)
            print(f'  SKIPPED: {exc}',flush=True)
        pd.DataFrame(old).to_csv(manifest,index=False)
        pd.DataFrame(skipped,columns=['family','comparison','experiment_a','experiment_b','reason']).to_csv(skipped_file,index=False)
    print(f'Done: {successes} new successes, {len(skipped)} skips recorded, {unchanged} already processed.')
    print('Results:',manifest)
    print(f'Partition: {args.partition}; bootstrap CIs unadjusted for multiple comparisons.')
    print('A cohort-matched contrast is NOT a causal estimate of the cleaning policy effect.')

if __name__ == '__main__':
    main()
