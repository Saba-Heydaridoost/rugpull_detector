"""Fail-fast input audit for all pre-registered GNN-versus-baseline comparisons."""
import argparse
from pathlib import Path
import json
import pandas as pd
from gnn_baseline_comparison import COLS,read_pair,paired_arrays
from gnn_panel_utils import unique_groups,load_fixed_panel

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--partition',choices=['val','test'],required=True)
    p.add_argument('--manifest',type=Path,default=ROOT/'gnn_baseline_pairs.csv')
    p.add_argument('--input',type=Path,default=ROOT/'input')
    p.add_argument('--output',type=Path)
    p.add_argument('--min-tokens',type=int,default=30)
    a=p.parse_args()
    man=pd.read_csv(a.manifest,dtype=str)
    if man.empty or not set(COLS).issubset(man.columns) or man[COLS].isna().any().any():raise ValueError('Bad manifest')
    if man.duplicated(COLS).any():raise ValueError('Duplicate manifest entries')
    out=a.output or ROOT/'output'/a.partition/'audit'/'gnn_preflight'
    out.mkdir(parents=True,exist_ok=True)
    errors=[];rows=[]; groups={}
    for i,r in man.iterrows():
        try:
            fields,pa,pb=read_pair(r,a.partition,a.input)
            group=tuple(fields[x] for x in ('scope','model_a','model_b','view_b','variant','cohort'))
            if fields['scope']=='panel':
                if group not in groups:groups[group]=load_fixed_panel(fields,a.partition,a.input,a.min_tokens)
                ids,y,seeds,_,digest=groups[group]
                matching='longitudinal'
            else:
                ids,y,_,_,seeds,_,_=paired_arrays(pa,pb,a.partition,a.min_tokens)
                digest='';matching='pairwise'
            rows.append({**fields,'partition':a.partition,'matching':matching,'n_tokens':len(ids),'n_positive':int(y.sum()),'n_negative':int((y==0).sum()),'n_seeds':len(seeds),'matched_ids_sha256':digest,'path_a':str(pa),'path_b':str(pb)})
        except Exception as e:
            errors.append({'row':int(i)+2,**{k:str(r.get(k,'')) for k in COLS},'error':repr(e)})
    pd.DataFrame(rows).to_csv(out/'coverage.csv',index=False)
    pd.DataFrame(errors,columns=['row',*COLS,'error']).to_csv(out/'errors.csv',index=False)
    (out/'summary.json').write_text(json.dumps({'partition':a.partition,'required_comparisons':len(man),'passed':len(rows),'failed':len(errors),'manifest':str(a.manifest.resolve())},indent=2))
    print(f'Preflight {a.partition}: {len(rows)} passed / {len(errors)} failed; {out}')
    if errors:
        for e in errors[:5]:print('FAIL',e['row'],e['error'])
        raise SystemExit(1)
if __name__=='__main__':main()
