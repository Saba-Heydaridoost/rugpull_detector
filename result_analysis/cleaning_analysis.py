"""Cleaning/missingness analysis, matched on tokens (Validation only)."""
import argparse
from comparison_utils import ROOT, COHORTS, stem, load, coverage, compare, save

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--scope', choices=['full','panel'], default='panel')
    p.add_argument('--model', default='gradient_boosting')
    p.add_argument('--q',choices=['q1','q2','q3'],default='q1')
    p.add_argument('--view',default='node_graph')
    p.add_argument('--reps',type=int,default=1000)
    p.add_argument('--rng-seed',type=int,default=2026)
    args=p.parse_args()
    if args.reps<100:p.error('--reps must be >= 100')
    kw=(args.model,args.scope,args.q,args.view)
    cov, comparisons, skipped=[],[],[]
    for cohort in COHORTS:
        for variant in (('no_mask',) if cohort=='complete_case' else ('no_mask','with_mask')):
            name=stem(*kw,variant,cohort)
            try: cov.append(dict(model=args.model,scope=args.scope,q=args.q,view=args.view,variant=variant,cohort=cohort,**coverage(load(args.scope,name))))
            except FileNotFoundError as e:skipped.append(str(e))
    for cohort in COHORTS[:2]:
        a,b=stem(*kw,'with_mask',cohort),stem(*kw,'no_mask',cohort)
        try: comparisons.append(dict(comparison='mask_effect',cohort_a=cohort,cohort_b=cohort,**compare(args.scope,a,b,args.reps,args.rng_seed)))
        except (FileNotFoundError,ValueError) as e:skipped.append(f'{a} vs {b}: {e}')
    for cohort in COHORTS[1:]:
        a,b=stem(*kw,'no_mask',cohort),stem(*kw,'no_mask','original')
        try: comparisons.append(dict(comparison='cohort_matched',cohort_a=cohort,cohort_b='original',**compare(args.scope,a,b,args.reps,args.rng_seed)))
        except (FileNotFoundError,ValueError) as e:skipped.append(f'{a} vs {b}: {e}')
    out=ROOT/'output'/'cleaning'
    tag=f'{args.scope}_{args.model}_{args.q}_{args.view}'
    save(cov,out/f'coverage_{tag}.csv');save(comparisons,out/f'paired_{tag}.csv')
    if skipped:(out/f'skipped_{tag}.txt').write_text('\n'.join(skipped),encoding='utf-8')
    for r in comparisons:
        print(f'{r["comparison"]}: {r["experiment_a"]} vs {r["experiment_b"]} n={r["n_paired"]} AP delta={r["delta_ap"]:+.5f} CI=[{r["delta_ap_ci_low"]:+.5f},{r["delta_ap_ci_high"]:+.5f}]')
    print(f'Skipped: {len(skipped)}')
if __name__=='__main__':main()
