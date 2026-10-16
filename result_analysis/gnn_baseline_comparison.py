"""Pre-specified GNN-vs-baseline comparisons on matched token IDs.

Manifest columns: analysis,scope,window_a,window_b,model_a,model_b,view_b,variant,cohort
model_a is one of gcn/graphsage/gat and uses view gnn_graph;
model_b is a baseline model using the explicitly specified view_b.
The primary comparison is on Panel; Full permits same-window comparisons only.

Examples:
 python gnn_baseline_comparison.py --partition val --manifest gnn_baseline_pairs.csv --reps 1000
 python gnn_baseline_comparison.py --partition test --manifest gnn_baseline_pairs.csv --reps 10000
"""

from __future__ import annotations
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prediction_io import prediction_file
from model_comparison import validated_predictions
from paired_bootstrap import bootstrap
from gnn_panel_utils import load_fixed_panel

ROOT = Path(__file__).resolve().parent
GNNS = {'gcn','graphsage','gat'}
BASELINES = {'dummy','logistic_regression','decision_tree','random_forest','gradient_boosting','xgboost','mlp','ft_transformer','autoint'}
VIEWS = {'node','node_matched','graph','node_graph'}
COHORTS = {'original','drop_3_or_more_missing','complete_case'}
WINDOWS = {'q1','q2','q3'}
COLS = ['analysis','scope','window_a','window_b','model_a','model_b','view_b','variant','cohort']
WINDOW_MINUTES={'q1':12.5,'q2':2127/60,'q3':5585/60}

def read_pair(row, partition, input_root):
    r={c:str(row[c]).strip() for c in COLS}
    if r['analysis'] not in {'same_window','early_vs_late'}: raise ValueError(f'Invalid analysis {r}')
    if r['scope'] not in {'full','panel'} or r['window_a'] not in WINDOWS or r['window_b'] not in WINDOWS: raise ValueError(f'Invalid scope/windows {r}')
    if r['model_a'] not in GNNS or r['model_b'] not in BASELINES or r['view_b'] not in VIEWS: raise ValueError(f'Invalid models/view {r}')
    if r['cohort'] not in COHORTS or r['variant'] not in {'no_mask','with_mask'}: raise ValueError(f'Invalid variant/cohort {r}')
    if r['cohort']=='complete_case' and r['variant']!='no_mask': raise ValueError('complete_case only permits no_mask')
    if r['analysis']=='same_window' and r['window_a']!=r['window_b']: raise ValueError('same_window requires identical windows')
    if r['analysis']=='early_vs_late':
        if r['scope']!='panel' or WINDOWS_ORDER[r['window_a']]>=WINDOWS_ORDER[r['window_b']]:
            raise ValueError('early_vs_late requires panel and GNN earlier than baseline')
    sa=f"{r['model_a']}_{r['scope']}_{r['window_a']}_gnn_graph_structure_timing_node_graph_{r['variant']}_{r['cohort']}"
    sb=f"{r['model_b']}_{r['scope']}_{r['window_b']}_{r['view_b']}_{r['variant']}_{r['cohort']}"
    pa=prediction_file(input_root/f"gnn_{r['scope']}",sa,partition)
    pb=prediction_file(input_root/f"baseline_{r['scope']}",sb,partition)
    return r,pa,pb
WINDOWS_ORDER={'q1':0,'q2':1,'q3':2}

def paired_arrays(path_a,path_b,partition,min_tokens):
    ids_a,y_a,m_a,seeds_a=validated_predictions(path_a,partition)
    ids_b,y_b,m_b,seeds_b=validated_predictions(path_b,partition)
    if not np.array_equal(seeds_a,seeds_b): raise ValueError('Seed alignment mismatch')
    ids=np.intersect1d(ids_a,ids_b)
    if len(ids)<min_tokens: raise ValueError(f'Insufficient paired tokens: {len(ids)} < {min_tokens}')
    ia=pd.Index(ids_a).get_indexer(ids);ib=pd.Index(ids_b).get_indexer(ids)
    y=y_a[ia]
    if not np.array_equal(y,y_b[ib]): raise ValueError('Label disagreement between matched tokens')
    if not np.any(y==0) or not np.any(y==1): raise ValueError('Both classes required')
    return ids,y,m_a[ia],m_b[ib],seeds_a,len(ids_a),len(ids_b)

def figure_summary(df,out):
    for analysis,grp in df.groupby('analysis'):
        grp=grp.sort_values(['window_a','window_b','model_a','model_b','view_b'])
        fig,ax=plt.subplots(figsize=(max(8, min(18,0.32*len(grp)+5)),max(4,0.29*len(grp)+1.8)))
        yy=np.arange(len(grp))
        ax.errorbar(grp.delta_ap,yy,xerr=[grp.delta_ap-grp.delta_ap_ci_low,grp.delta_ap_ci_high-grp.delta_ap],fmt='o',capsize=2)
        ax.axvline(0,color='black',linewidth=1)
        labels=[f"{r.model_a} {r.window_a} vs {r.model_b} {r.window_b} ({r.view_b}/{r.cohort}/{r.variant})" for r in grp.itertuples()]
        ax.set_yticks(yy,labels,fontsize=8);ax.invert_yaxis();ax.set_xlabel('Paired difference in Average Precision (GNN - Baseline)')
        ax.set_title(f'{analysis}: matched-token bootstrap 95% CI (unadjusted)')
        fig.tight_layout()
        fig.savefig(out/f'{analysis}_delta_ap.pdf');fig.savefig(out/f'{analysis}_delta_ap.png',dpi=180)
        plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--partition',choices=['val','test'],required=True)
    p.add_argument('--manifest',type=Path,default=ROOT/'gnn_baseline_pairs.csv')
    p.add_argument('--input',type=Path,default=ROOT/'input')
    p.add_argument('--output',type=Path)
    p.add_argument('--reps',type=int,default=1000)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--min-tokens',type=int,default=30)
    p.add_argument('--matching',choices=['longitudinal','pairwise'],default='longitudinal')
    args=p.parse_args()
    if args.reps<100: p.error('--reps >= 100 required')
    out=args.output or ROOT/'output'/args.partition/'gnn_vs_baseline'/f'reps_{args.reps}_seed_{args.seed}_{args.matching}'
    out.mkdir(parents=True,exist_ok=True)
    man=pd.read_csv(args.manifest,dtype=str)
    if man.empty or not set(COLS).issubset(man.columns) or man[COLS].isna().any().any(): raise ValueError('Invalid/empty manifest')
    if man.duplicated(COLS).any(): raise ValueError('Duplicate comparisons in manifest')
    results=[];seed_rows=[]
    for idx,row in man.iterrows():
        r,fa,fb=read_pair(row,args.partition,args.input)
        if args.matching == 'longitudinal':
            ids,y,seeds,mats,digest=load_fixed_panel(r,args.partition,args.input,args.min_tokens)
            aa=mats[('gnn',r['window_a'])]
            bb=mats[('baseline',r['window_b'])]
            na=nb=len(ids)
        else:
            ids,y,aa,bb,seeds,na,nb=paired_arrays(fa,fb,args.partition,args.min_tokens)
            digest=hashlib.sha256('\n'.join(map(str,ids)).encode()).hexdigest()
        point,diffs,ci,apa,apb,roca,rocb=bootstrap(y,aa,bb,args.reps,args.seed,.95)
        entry={**r,'partition':args.partition,'n_paired':len(ids),'n_positive':int(y.sum()),'n_negative':int((y==0).sum()),'n_a_tokens':na,'n_b_tokens':nb,'n_seeds':len(seeds),'n_bootstrap':args.reps,'rng_seed':args.seed,
               'matched_ids_sha256':digest,'matching':args.matching,
               'minutes_a':WINDOW_MINUTES[r['window_a']],'minutes_b':WINDOW_MINUTES[r['window_b']],
               **{k:float(v) for k,v in point.items()},'delta_ap_ci_low':float(ci[0,0]),'delta_ap_ci_high':float(ci[1,0]),'delta_roc_ci_low':float(ci[0,1]),'delta_roc_ci_high':float(ci[1,1])}
        results.append(entry)
        for j,seed in enumerate(seeds):
            seed_rows.append({**{k:r[k] for k in COLS},'partition':args.partition,'seed':int(seed),'ap_a':apa[j],'ap_b':apb[j],'roc_a':roca[j],'roc_b':rocb[j]})
        print(f"[{idx+1}/{len(man)}] {r['model_a']} {r['window_a']} - {r['model_b']} {r['window_b']}: paired {len(ids)}",flush=True)
    
    shutil.copy2(
        args.manifest,
        out / "manifest_used.csv"
    )
    
    result=pd.DataFrame(results)
    result.to_csv(out/'paired_comparisons.csv',index=False)
    pd.DataFrame(seed_rows).to_csv(out/'seedwise.csv',index=False)
    figure_summary(result,out)
    (out/'provenance.json').write_text(json.dumps({'partition':args.partition,'manifest':str(args.manifest.resolve()),'bootstrap_reps':args.reps,'seed':args.seed,'confidence_interval':'percentile, unadjusted','resampling_unit':'token_mint stratified by y_true','interpretation':'performance at fixed observation cutoffs; not measured first-alert latency','GNN_view':'gnn_graph','matching':args.matching+' matching; identical y_true and seed indices required'},indent=2),encoding='utf-8')
    print('Done:',out)
    
if __name__=='__main__':
    main()
