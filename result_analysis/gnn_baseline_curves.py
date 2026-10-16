"""Fixed-cohort performance curves for pre-specified GNN-versus-baseline pairs.
Pointwise stratified token-level percentile CIs are descriptive, not paired tests.
"""
import argparse
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gnn_panel_utils import unique_groups,load_fixed_panel,WINDOWS
from paired_bootstrap import fast_scores

ROOT=Path(__file__).resolve().parent
MINUTES={'q1':12.5,'q2':2127/60,'q3':5585/60}

def point_ci(y,mat,reps,seed):
    p,n=np.flatnonzero(y==1),np.flatnonzero(y==0)
    rng=np.random.default_rng(seed)
    vals=np.empty((reps,2))
    for i in range(reps):
        ix=np.concatenate([rng.choice(p,len(p),replace=True),rng.choice(n,len(n),replace=True)])
        a,r=fast_scores(y[ix],mat[ix]);vals[i]=[a.mean(),r.mean()]
    return np.quantile(vals,[.025,.975],axis=0)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--partition',choices=['val','test'],required=True)
    p.add_argument('--manifest',type=Path,default=ROOT/'gnn_baseline_pairs.csv')
    p.add_argument('--input',type=Path,default=ROOT/'input')
    p.add_argument('--output',type=Path)
    p.add_argument('--reps',type=int,default=1000)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--min-tokens',type=int,default=30)
    a=p.parse_args()
    if a.reps<100:p.error('--reps >= 100 required')
    man=pd.read_csv(a.manifest,dtype=str)
    required={'analysis','scope','window_a','window_b','model_a','model_b','view_b','variant','cohort'}
    if man.empty or not required.issubset(man.columns) or man[list(required)].isna().any().any():raise ValueError('Invalid manifest')
    out=a.output or ROOT/'output'/a.partition/'gnn_vs_baseline'/'curves'/f'reps_{a.reps}_seed_{a.seed}'
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    groups=unique_groups(man)
    if groups.empty:raise ValueError('No panel comparisons in manifest')
    for _,g in groups.iterrows():
        subset=man.copy()
        for k in g.index:subset=subset[subset[k].eq(g[k])]
        same=subset[subset.analysis.eq('same_window')]
        if set(zip(same.window_a,same.window_b))!={(q,q) for q in WINDOWS}:
            raise ValueError(f'Missing all three same-window rows for {g.to_dict()}')
        ids,y,seeds,mats,digest=load_fixed_panel(g,a.partition,a.input,a.min_tokens)
        name=f"{g.model_a}_vs_{g.model_b}_{g.view_b}_{g.variant}_{g.cohort}"
        for family,model in (('gnn',g.model_a),('baseline',g.model_b)):
            for w in WINDOWS:
                matrix=mats[(family,w)]
                ap,roc=fast_scores(y,matrix)
                bounds=point_ci(y,matrix,a.reps,a.seed)
                rows.append(dict(group=name,partition=a.partition,model=model,family=family,window=w,
                    minutes=MINUTES[w],view='gnn_graph' if family=='gnn' else g.view_b,
                    variant=g.variant,cohort=g.cohort,n_tokens=len(ids),n_positive=int(y.sum()),
                    n_negative=int((y==0).sum()),n_seeds=len(seeds),matched_ids_sha256=digest,
                    ap_mean=float(ap.mean()),ap_ci_low=float(bounds[0,0]),ap_ci_high=float(bounds[1,0]),
                    roc_auc_mean=float(roc.mean()),roc_auc_ci_low=float(bounds[0,1]),roc_auc_ci_high=float(bounds[1,1]),
                    n_bootstrap=a.reps,rng_seed=a.seed))
        print(f'{name}: longitudinal {len(ids)} tokens',flush=True)
    df=pd.DataFrame(rows)
    df.to_csv(out/'matched_performance_curves.csv',index=False)
    shutil.copy2(a.manifest,out/'manifest_used.csv')
    (out/'provenance.json').write_text(json.dumps({'partition':a.partition,'reps':a.reps,'seed':a.seed,'matching':'all 6 predictions same token set','ci':'pointwise token-stratified 95% percentile; unadjusted; not a test of differences'},indent=2))
    for metric,mean,lo,hi in (('ap','ap_mean','ap_ci_low','ap_ci_high'),('roc_auc','roc_auc_mean','roc_auc_ci_low','roc_auc_ci_high')):
        for group,part in df.groupby('group'):
            fig,ax=plt.subplots(figsize=(7.5,4.5))
            for family,sub in part.groupby('family'):
                sub=sub.sort_values('minutes')
                ax.errorbar(sub.minutes,sub[mean],yerr=[sub[mean]-sub[lo],sub[hi]-sub[mean]],fmt='-o',capsize=3,label=f'{sub.model.iloc[0]} ({family})')
            ax.set_xticks(list(MINUTES.values()),['12.5','35.45','93.08'])
            ax.set_xlabel('Observation time after creation (minutes)')
            ax.set_ylabel('Average Precision' if metric=='ap' else 'ROC-AUC')
            ax.set_ylim(0,1)
            ax.set_title(f'{group}\nSame tokens across six points; pointwise 95% CIs')
            ax.grid(alpha=.25);ax.legend();fig.tight_layout()
            fig.savefig(out/f'{group}_{metric}.pdf')
            fig.savefig(out/f'{group}_{metric}.png',dpi=180)
            plt.close(fig)
    print('Saved:',out)
if __name__=='__main__':main()
