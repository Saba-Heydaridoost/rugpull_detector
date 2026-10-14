"""Shared, validation-only, token-matched comparisons."""
from pathlib import Path
import pandas as pd
from paired_bootstrap import load_predictions, prepare, bootstrap
ROOT = Path(__file__).resolve().parent
COHORTS = ('original', 'drop_3_or_more_missing', 'complete_case')

def stem(model, scope, q, view, variant, cohort):
    return f'{model}_{scope}_{q}_{view}_{variant}_{cohort}'

def load(scope, name):
    return load_predictions(ROOT/'input'/f'baseline_{scope}'/f'predictions_{name}.csv', 'val')

def coverage(df):
    labels = df.groupby('token_mint').y_true
    if (labels.nunique() != 1).any():
        raise ValueError('Labels inconsistent across seeds')
    y = labels.first()
    return {'n_tokens':len(y), 'n_positive':int(y.sum()), 'n_negative':int((y == 0).sum()), 'prevalence':float(y.mean())}

def compare(scope, a, b, reps, rng_seed):
    A, B = load(scope,a), load(scope,b)
    ca, cb = coverage(A), coverage(B)
    ids, seeds, y, pa, pb = prepare(A,B)
    point, _, ci, *_ = bootstrap(y,pa,pb,reps,rng_seed,.95)
    return {'scope':scope,'experiment_a':a,'experiment_b':b,
            'a_n_tokens':ca['n_tokens'],'b_n_tokens':cb['n_tokens'],
            'a_prevalence':ca['prevalence'],'b_prevalence':cb['prevalence'],
            'n_paired':len(ids),'n_positive':int(y.sum()),'n_negative':int((1-y).sum()),
            'n_seeds':len(seeds),'n_bootstrap':reps,**point,
            'delta_ap_ci_low':ci[0,0], 'delta_ap_ci_high':ci[1,0],
            'delta_roc_ci_low':ci[0,1], 'delta_roc_ci_high':ci[1,1]}

def save(rows, path):
    path.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(path,index=False)
    print(f'Saved {len(rows)} rows: {path}')
