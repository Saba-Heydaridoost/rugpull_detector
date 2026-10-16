"""Nine predeclared directional AP contrasts with multiplicity-protected bootstrap intervals.

Scientific interpretation: Bonferroni simultaneous percentile bootstrap intervals
are approximate, conservative multiplicity protection for nine AP contrasts.
A positive simultaneous LOWER bound supports directional superiority at
family-wise nominal alpha 0.05, subject to bootstrap regularity/selection caveats.
No score-swap permutation p-values are computed or claimed.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from paired_bootstrap import bootstrap
from gnn_panel_utils import load_fixed_panel

ROOT=Path(__file__).resolve().parent
GNNS=('gcn','graphsage','gat')
PAIRS=(('q1','q1'),('q1','q2'),('q1','q3'))
NTESTS=len(GNNS)*len(PAIRS)
ALPHA=.05

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--partition', choices=('val','test'), required=True)
    p.add_argument('--input',type=Path,default=ROOT/'input')
    p.add_argument('--output',type=Path)
    p.add_argument('--reps',type=int,default=10000)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--min-tokens',type=int,default=30)
    a=p.parse_args()
    if a.reps<1000: p.error('--reps must be >=1000')
    out=a.output or ROOT/'output'/a.partition/'gnn_vs_baseline'/f'primary_bonferroni_reps_{a.reps}_seed_{a.seed}'
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for gnn in GNNS:
        spec={'scope':'panel','model_a':gnn,'model_b':'xgboost','view_b':'node_graph','variant':'no_mask','cohort':'original'}
        ids,y,seeds,mats,digest=load_fixed_panel(spec,a.partition,a.input,a.min_tokens)
        if not np.array_equal(seeds,np.arange(5)):
            raise ValueError(f'Expected exact seeds 0..4 for {gnn}, got {seeds}')
        for wa,wb in PAIRS:
            point,diffs,ci,_,_,_,_=bootstrap(y,mats['gnn',wa],mats['baseline',wb],a.reps,a.seed,.95)
            # Two-sided family-wise intervals: each marginal confidence 1 - alpha / 9.
            lo,hi=np.quantile(diffs[:,0],[ALPHA/(2*NTESTS),1-ALPHA/(2*NTESTS)])
            rows.append(dict(model_a=gnn,model_b='xgboost',scope='panel',view_b='node_graph',variant='no_mask',cohort='original',
                             window_a=wa,window_b=wb,partition=a.partition,endpoint='AP',delta_ap=float(point['delta_ap']),
                             ap_a=float(point['ap_a']),ap_b=float(point['ap_b']),ci_95_unadjusted_low=float(ci[0,0]),
                             ci_95_unadjusted_high=float(ci[1,0]),ci_bonferroni_low=float(lo),ci_bonferroni_high=float(hi),
                             marginal_confidence=1-ALPHA/NTESTS,nominal_familywise_alpha=ALPHA,
                             supports_superiority_fwer_0p05=bool(lo>0),
                             n_tokens=len(ids),n_positive=int(y.sum()),n_negative=int((y==0).sum()),
                             n_seeds=len(seeds),matched_ids_sha256=digest,n_bootstrap=a.reps,rng_seed=a.seed))
            print(f'{gnn} {wa} vs XGBoost {wb}: ΔAP={point["delta_ap"]:+.6f}, Bonferroni CI=[{lo:+.6f},{hi:+.6f}]',flush=True)
    pd.DataFrame(rows).to_csv(out/'primary_simultaneous_ci_results.csv',index=False)
    prov=dict(partition=a.partition,hypotheses=NTESTS,endpoint='mean seed AP difference',resampling='stratified paired token bootstrap, one shared token draw per replicate across 5 seeds',
              adjustment='Bonferroni simultaneous two-sided percentile intervals across 9 fixed contrasts',
              alpha=ALPHA,marginal_confidence=1-ALPHA/NTESTS,bootstrap_reps=a.reps,seed=a.seed,
              caveat='Approximate bootstrap coverage; NOT permutation inference; positive lower bound supports directional superiority, not non-inferiority or first-alert latency.',
              cohort='GNN-specific six-file longitudinal token intersection; cohorts may differ by GNN')
    (out/'provenance.json').write_text(json.dumps(prov,indent=2),encoding='utf8')
    print('Saved:',out/'primary_simultaneous_ci_results.csv')
if __name__=='__main__':main()
