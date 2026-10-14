"""Fair descriptive comparison of saved *Validation* baseline predictions.

Unlike within-model comparison_results.csv, this compares different models.
For each (scope, window, view, missingness variant, cohort) configuration,
score EVERY eligible model on the exact same token intersection and seed IDs.
All models must share both labels and all expected seeds; invalid groups fail
rather than silently dropping incomplete model predictions.

No hyperparameter tuning, statistical significance claims, or Test access.
The mean of five seed-level AP/ROC scores is descriptive, not five independent
replications. Rankings are Validation-only and unsuitable as unbiased Test claims.

Usage:
    python model_comparison.py
    python model_comparison.py --scope panel --min-tokens 50
    python model_comparison.py --input path/to/input --output path/to/output/model_comparison
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
EXPECTED_SEEDS = frozenset(range(5))
REQUIRED_MODELS = frozenset({
    'dummy','logistic_regression','decision_tree','random_forest',
    'gradient_boosting','xgboost','mlp','ft_transformer','autoint',
})
COHORTS = ('drop_3_or_more_missing','complete_case','original')
VIEWS = ('node_graph_freeze','node_matched','node_graph','graph','node')


def parse_stem(stem: str, scope: str) -> dict:
    # Use a fixed suffix grammar: model names themselves can contain underscores.
    marker = f'_{scope}_'
    if marker not in stem:
        raise ValueError(f'Unexpected experiment scope: {stem}')
    model, remainder = stem.split(marker, 1)
    if model not in REQUIRED_MODELS:
        raise ValueError(f'Unexpected model in {stem}: {model}')
    for cohort in COHORTS:
        suffix = '_' + cohort
        if not remainder.endswith(suffix):
            continue
        head = remainder[:-len(suffix)]
        for variant in ('with_mask','no_mask'):
            vsuffix = '_' + variant
            if not head.endswith(vsuffix):
                continue
            prefix = head[:-len(vsuffix)]
            for view in VIEWS:
                vmarker = '_' + view
                if prefix.endswith(vmarker):
                    q = prefix[:-len(vmarker)]
                    if q not in ('q1','q2','q3'):
                        raise ValueError(f'Invalid window in {stem}')
                    return {'model':model, 'scope':scope, 'q':q,
                            'view':view, 'variant':variant, 'cohort':cohort}
    raise ValueError(f'Cannot parse experiment stem: {stem}')


def validated_predictions(path: Path) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    df = pd.read_csv(path, usecols=['token_mint','seed','partition','y_true','y_score'],low_memory=False)
    df = df[df.partition.eq('val')].copy()
    if df.empty or df[['token_mint','seed','y_true','y_score']].isna().any().any():
        raise ValueError(f'Empty/missing Validation data: {path}')
    if df.duplicated(['token_mint','seed']).any():
        raise ValueError(f'Duplicate token/seed: {path}')
    if set(df.seed.astype(int)) != EXPECTED_SEEDS or not (df.seed == df.seed.astype(int)).all():
        raise ValueError(f'Incorrect training seed set: {path}')
    if not df.y_true.isin([0,1]).all() or not np.isfinite(df.y_score).all() or not df.y_score.between(0,1).all():
        raise ValueError(f'Invalid targets/scores: {path}')
    counts = df.groupby('token_mint',sort=False).seed.nunique()
    if not counts.eq(len(EXPECTED_SEEDS)).all():
        raise ValueError(f'Incomplete token/seed predictions: {path}')
    if not df.groupby('token_mint').y_true.nunique().eq(1).all():
        raise ValueError(f'Inconsistent labels across seeds: {path}')
    wide = df.pivot(index='token_mint',columns='seed',values='y_score')
    ys = df.groupby('token_mint').y_true.first().reindex(wide.index)
    if wide.isna().any().any() or ys.isna().any():
        raise ValueError(f'Bad pivot: {path}')
    return wide.index.to_numpy(), ys.to_numpy(dtype=int), wide.to_numpy(dtype=float), np.array(wide.columns)


def analyze_folder(folder: Path, scope: str, min_tokens: int):
    paths = sorted(folder.glob('predictions_*.csv'))
    if not paths:
        raise FileNotFoundError(f'No prediction CSVs in {folder}. Restore input/baseline_{scope}/ first.')
    groups: dict[tuple,dict] = {}
    for path in paths:
        stem = path.stem[len('predictions_'):]
        info = parse_stem(stem,scope)
        key = tuple(info[x] for x in ('scope','q','view','variant','cohort'))
        if info['model'] in groups.setdefault(key,{}):
            raise ValueError(f'Duplicate model/configuration: {stem}')
        groups[key][info['model']] = path
    records=[]; diagnostics=[]
    for key, models in sorted(groups.items()):
        scope,q,view,variant,cohort=key
        # Do not silently assert a missing model is a legitimate design choice.
        # Valid classes of configurations should have all nine models.
        if set(models) != REQUIRED_MODELS:
            raise ValueError(f'Missing models for {key}: {sorted(REQUIRED_MODELS-set(models))}')
        loaded={m: validated_predictions(p) for m,p in sorted(models.items())}
        common=set.intersection(*(set(data[0]) for data in loaded.values()))
        all_n={m:len(d[0]) for m,d in loaded.items()}
        ids=sorted(common)
        if not ids:
            raise ValueError(f'No common token set for {key}')
        labels=None; scores={}
        for model,(tokens,y,mat,seeds) in loaded.items():
            ix=pd.Index(tokens).get_indexer(ids)
            if (ix<0).any() or not np.array_equal(seeds,np.array(sorted(EXPECTED_SEEDS))):
                raise ValueError(f'Seed/token mismatch for {key}/{model}')
            yi=y[ix]
            if labels is None: labels=yi
            elif not np.array_equal(labels,yi):
                raise ValueError(f'Cross-model labels inconsistent for {key}/{model}')
            scores[model]=mat[ix,:]
        npos=int(labels.sum()); nneg=len(labels)-npos
        status='ok' if len(ids)>=min_tokens and npos>0 and nneg>0 else 'insufficient_matched_data'
        diagnostics.append(dict(scope=scope,q=q,view=view,variant=variant,cohort=cohort,
                                n_models=len(models),n_common_tokens=len(ids),n_positive=npos,n_negative=nneg,
                                min_tokens_threshold=min_tokens,status=status,
                                min_individual_n=min(all_n.values()),max_individual_n=max(all_n.values())))
        if status!='ok':continue
        for model,mat in sorted(scores.items()):
            ap=np.array([average_precision_score(labels,mat[:,j]) for j in range(mat.shape[1])])
            roc=np.array([roc_auc_score(labels,mat[:,j]) for j in range(mat.shape[1])])
            records.append(dict(scope=scope,q=q,view=view,variant=variant,cohort=cohort,model=model,
                                n_matched=len(ids),n_positive=npos,n_negative=nneg,n_seeds=len(seeds),
                                ap_mean=float(ap.mean()),ap_std_between_seeds=float(ap.std(ddof=1)),
                                roc_mean=float(roc.mean()),roc_std_between_seeds=float(roc.std(ddof=1)),
                                n_original_model_tokens=all_n[model]))
    return records,diagnostics


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input',type=Path,default=ROOT/'input')
    p.add_argument('--output',type=Path,default=ROOT/'output'/'model_comparison')
    p.add_argument('--scope',choices=('full','panel','both'),default='both')
    p.add_argument('--min-tokens',type=int,default=30,help='Sensitivity cutoff, not a statistical guarantee (default 30)')
    a=p.parse_args()
    if a.min_tokens<2:p.error('--min-tokens must be >= 2')
    records=[];diagnostics=[]
    for scope in (('full','panel') if a.scope=='both' else (a.scope,)):
        r,d=analyze_folder(a.input/f'baseline_{scope}',scope,a.min_tokens)
        records.extend(r);diagnostics.extend(d)
    scores=pd.DataFrame(records)
    report=pd.DataFrame(diagnostics)
    if scores.empty:raise ValueError('No eligible model comparisons; see --min-tokens and input coverage')
    scores['ap_rank_within_matched_group']=scores.groupby(['scope','q','view','variant','cohort']).ap_mean.rank(ascending=False,method='min').astype(int)
    scores['roc_rank_within_matched_group']=scores.groupby(['scope','q','view','variant','cohort']).roc_mean.rank(ascending=False,method='min').astype(int)
    a.output.mkdir(parents=True,exist_ok=True)
    scores.to_csv(a.output/'matched_model_scores.csv',index=False)
    report.to_csv(a.output/'matched_group_coverage.csv',index=False)
    print('Validation-only descriptive matched-token model scores:',len(scores))
    print('Configurations:',len(report),'eligible:',(report.status=='ok').sum(),
          'not eligible:',(report.status!='ok').sum())
    print('Output:',a.output)
    print('Caution: ranks are VALIDATION descriptive ranks, not independent significance tests or Test performance.')

if __name__=='__main__':main()
