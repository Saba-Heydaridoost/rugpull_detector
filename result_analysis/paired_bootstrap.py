"""Paired, token-stratified bootstrap for TWO saved baseline Validation experiments.

No training, no Test access. Resample token IDs *once* per replicate, shared by
both models/windows and all five seeds. Report per-seed differences and their
mean (seeds are repeated fits, not additional independent observations).

Example (execute from result_analysis):
 python paired_bootstrap.py --scope panel --a gradient_boosting_panel_q1_node_graph_no_mask_original --b gradient_boosting_panel_q3_node_graph_no_mask_original --reps 10000
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

BASE = Path(__file__).resolve().parent


def load_predictions(path: Path, partition: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, usecols=['token_mint','seed','partition','y_true','y_score'])
    df = df.loc[df['partition'].eq(partition), ['token_mint','seed','y_true','y_score']].copy()
    if df.empty:
        raise ValueError(f'No {partition} predictions: {path}')
    if df.duplicated(['token_mint','seed']).any():
        raise ValueError(f'Duplicate token/seed: {path}')
    if df[['token_mint','seed','y_true','y_score']].isna().any().any():
        raise ValueError(f'Missing values: {path}')
    if not df.y_true.isin([0,1]).all() or not np.isfinite(df.y_score).all():
        raise ValueError(f'Invalid labels/scores: {path}')
    if not df.y_score.between(0,1).all():
        raise ValueError(f'Score outside [0,1]: {path}')
    return df


def prepare(a: pd.DataFrame, b: pd.DataFrame):
    seeds_a = set(a.seed.unique()); seeds_b = set(b.seed.unique())
    if seeds_a != seeds_b:
        raise ValueError(f'Seed mismatch: A={sorted(seeds_a)} B={sorted(seeds_b)}')
    seeds = sorted(seeds_a)
    indexed = []
    for src in (a,b):
        counts = src.groupby('token_mint').seed.nunique()
        complete = set(counts[counts.eq(len(seeds))].index)
        indexed.append((src,complete))
    token_ids = sorted(indexed[0][1] & indexed[1][1])
    if not token_ids:
        raise ValueError('No common tokens with every seed in both experiments')
    arrays=[]
    for src,_ in indexed:
        pivot = src[src.token_mint.isin(token_ids)].pivot(index='token_mint',columns='seed',values='y_score').reindex(index=token_ids,columns=seeds)
        yp = src[src.token_mint.isin(token_ids)].pivot(index='token_mint',columns='seed',values='y_true').reindex(index=token_ids,columns=seeds)
        if pivot.isna().any().any() or yp.isna().any().any():
            raise ValueError('Incomplete token/seed alignment')
        if not (yp.to_numpy() == yp.to_numpy()[:,[0]]).all():
            raise ValueError('Labels differ across seeds within experiment')
        arrays.append((yp.to_numpy()[:,0].astype(int),pivot.to_numpy(dtype=float)))
    ya,pa=arrays[0]; yb,pb=arrays[1]
    if not np.array_equal(ya,yb):
        raise ValueError('A and B disagree on y_true for paired tokens')
    if len(np.unique(ya))<2:
        raise ValueError('Common token set has only one label; AP/ROC comparison undefined')
    return token_ids,seeds,ya,pa,pb


def scores(y, mat):
    ap=np.array([average_precision_score(y,mat[:,j]) for j in range(mat.shape[1])])
    roc=np.array([roc_auc_score(y,mat[:,j]) for j in range(mat.shape[1])])
    return ap,roc



def fast_scores(y, mat):
    """
    Vectorized AP and ROC-AUC across all seed columns.

    Uses the same threshold-grouped definitions as sklearn.
    Correctly handles tied prediction scores.
    Requires both positive and negative labels.
    """
    order = np.argsort(-mat, axis=0, kind="stable")

    sorted_scores = np.take_along_axis(mat, order, axis=0)
    positives = y[order]

    tp = np.cumsum(positives, axis=0)
    fp = np.cumsum(1 - positives, axis=0)

    # Only evaluate at the end of each distinct score group.
    is_end = np.empty(sorted_scores.shape, dtype=bool)
    is_end[-1, :] = True
    is_end[:-1, :] = (
        sorted_scores[:-1, :] != sorted_scores[1:, :]
    )

    tp_at_end = np.where(is_end, tp, 0)
    fp_at_end = np.where(is_end, fp, 0)

    previous_tp = np.vstack((
        np.zeros((1, mat.shape[1]), dtype=int),
        np.maximum.accumulate(tp_at_end, axis=0)[:-1],
    ))

    previous_fp = np.vstack((
        np.zeros((1, mat.shape[1]), dtype=int),
        np.maximum.accumulate(fp_at_end, axis=0)[:-1],
    ))

    group_pos = np.where(is_end, tp - previous_tp, 0)
    group_neg = np.where(is_end, fp - previous_fp, 0)

    npos = int(np.sum(y))
    nneg = len(y) - npos

    if npos == 0 or nneg == 0:
        raise ValueError("Both classes are required for ROC-AUC")

    # Non-interpolated Average Precision, matching sklearn.
    ap = np.sum(
        (group_pos / npos) * (tp / (tp + fp)),
        axis=0,
    )

    # Rank-based ROC-AUC with half credit for tied scores.
    roc = np.sum(
        group_pos * (nneg - fp + 0.5 * group_neg),
        axis=0,
    ) / (npos * nneg)

    return ap, roc


def bootstrap(y,pa,pb,reps,seed,ci):
    pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0)
    rng=np.random.default_rng(seed)
    ap_a,roc_a=scores(y,pa);ap_b,roc_b=scores(y,pb)
    point={'ap_a':ap_a.mean(),'ap_b':ap_b.mean(),'delta_ap':(ap_a-ap_b).mean(),
           'roc_a':roc_a.mean(),'roc_b':roc_b.mean(),'delta_roc':(roc_a-roc_b).mean()}
    diffs=np.empty((reps,2),dtype=float)
    for i in range(reps):
        idx=np.concatenate((rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)))
        yy=y[idx]; a=pa[idx];b=pb[idx]
        aa, ar = fast_scores(yy, a)
        ba, br = fast_scores(yy, b)
        diffs[i]=[(aa-ba).mean(),(ar-br).mean()]
        if (i+1)%1000==0: print(f'  {i+1}/{reps} resamples',flush=True)
    alpha=(1-ci)/2
    return point,diffs,np.quantile(diffs,[alpha,1-alpha],axis=0),ap_a,ap_b,roc_a,roc_b


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--scope',required=True,choices=['full','panel'])
    p.add_argument('--a',required=True,help='Experiment stem, excluding predictions_ and .csv')
    p.add_argument('--b',required=True,help='Experiment stem, excluding predictions_ and .csv')
    p.add_argument('--reps',type=int,default=10000)
    p.add_argument('--rng-seed',type=int,default=2026)
    p.add_argument('--ci',type=float,default=.95)
    p.add_argument('--partition',choices=['val'],default='val')
    p.add_argument('--output',type=Path,default=BASE/'output'/'bootstrap')
    args=p.parse_args()
    if args.reps<100 or not (0<args.ci<1):p.error('--reps must be >=100 and --ci in (0,1)')
    for stem in (args.a,args.b):
        if '/' in stem or '\\' in stem or '..' in stem or f'_{args.scope}_' not in stem:
            p.error('Experiment stem must be a filename stem with the selected scope')
    folder=BASE/'input'/f'baseline_{args.scope}'
    a=load_predictions(folder/f'predictions_{args.a}.csv',args.partition)
    b=load_predictions(folder/f'predictions_{args.b}.csv',args.partition)
    ids,seeds,y,pa,pb=prepare(a,b)
    print(f'Paired tokens: {len(ids)} (positive={int(y.sum())}, negative={int((1-y).sum())}); seeds={seeds}')
    point,diffs,interval,apa,apb,roca,rocb=bootstrap(y,pa,pb,args.reps,args.rng_seed,args.ci)
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    tag=f'{args.scope}__{args.a}__VS__{args.b}'
    summary=pd.DataFrame([{'scope':args.scope,'experiment_a':args.a,'experiment_b':args.b,'partition':args.partition,'n_common_tokens':len(ids),'n_positive':int(y.sum()),'n_negative':int((1-y).sum()),'n_seeds':len(seeds),'n_bootstrap':args.reps,'ci':args.ci,**point,'delta_ap_ci_low':interval[0,0],'delta_ap_ci_high':interval[1,0],'delta_roc_ci_low':interval[0,1],'delta_roc_ci_high':interval[1,1]}])
    seedwise=pd.DataFrame({'seed':seeds,'ap_a':apa,'ap_b':apb,'delta_ap':apa-apb,'roc_a':roca,'roc_b':rocb,'delta_roc':roca-rocb})
    summary.to_csv(output/f'summary_{tag}.csv',index=False)
    seedwise.to_csv(output/f'seedwise_{tag}.csv',index=False)
    print('\nA - B (positive means A outperforms B):')
    print(f'  AP:  {point["delta_ap"]:+.6f}  {args.ci:.0%} CI [{interval[0,0]:+.6f}, {interval[1,0]:+.6f}]')
    print(f'  ROC: {point["delta_roc"]:+.6f}  {args.ci:.0%} CI [{interval[0,1]:+.6f}, {interval[1,1]:+.6f}]')
    print('Reports saved to:',output)
    print('Method: stratified paired token-level percentile bootstrap, shared resamples across seeds.')
    print('Scope: matched common tokens only; do not interpret as each model\'s full-population AP.')

if __name__=='__main__':main()
