"""Matched feature-view ablations (Validation only)."""
import argparse
from comparison_utils import ROOT,stem,compare,save
PAIRS=(('node_graph','node_matched'),('node_graph','node'),('node_graph_freeze','node_graph'),('graph','node_matched'))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--scope',choices=['full','panel'],default='panel')
    p.add_argument('--model',default='gradient_boosting')
    p.add_argument('--q',choices=['q1','q2','q3'],default='q1')
    p.add_argument('--cohort',choices=['original','drop_3_or_more_missing','complete_case'],default='original')
    p.add_argument('--variant',choices=['no_mask','with_mask'],default='no_mask')
    p.add_argument('--reps',type=int,default=1000)
    p.add_argument('--rng-seed',type=int,default=2026)
    args=p.parse_args()
    if args.reps<100:p.error('--reps must be >=100')
    if args.cohort=='complete_case' and args.variant=='with_mask':p.error('No complete_case with_mask experiments')
    rows,skipped=[],[]
    for a_view,b_view in PAIRS:
        a=stem(args.model,args.scope,args.q,a_view,args.variant,args.cohort)
        b=stem(args.model,args.scope,args.q,b_view,args.variant,args.cohort)
        try:rows.append(dict(view_a=a_view,view_b=b_view,**compare(args.scope,a,b,args.reps,args.rng_seed)))
        except (FileNotFoundError,ValueError) as e:skipped.append(f'{a} vs {b}: {e}')
    out=ROOT/'output'/'representation';tag=f'{args.scope}_{args.model}_{args.q}_{args.cohort}_{args.variant}'
    save(rows,out/f'paired_{tag}.csv')
    if skipped:(out/f'skipped_{tag}.txt').write_text('\n'.join(skipped),encoding='utf-8')
    for r in rows:
        print(f'{r["view_a"]} - {r["view_b"]}: n={r["n_paired"]} AP delta={r["delta_ap"]:+.5f} CI=[{r["delta_ap_ci_low"]:+.5f},{r["delta_ap_ci_high"]:+.5f}]')
    print(f'Skipped: {len(skipped)}')
if __name__=='__main__':main()
