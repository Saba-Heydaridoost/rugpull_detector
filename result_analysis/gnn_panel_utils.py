"""Shared fixed longitudinal matched cohort for GNN-vs-baseline comparisons."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from prediction_io import prediction_file
from model_comparison import validated_predictions

WINDOWS=('q1','q2','q3')
KEY=('scope','model_a','model_b','view_b','variant','cohort')

def load_fixed_panel(row, partition, input_root, min_tokens=30):
    if str(row['scope'])!='panel':
        raise ValueError('Longitudinal matching requires scope=panel')
    loaded={}
    for window in WINDOWS:
        for family,model,view in (('gnn',row['model_a'],'gnn_graph'),('baseline',row['model_b'],row['view_b'])):
            stem=f"{model}_panel_{window}_{view}_{row['variant']}_{row['cohort']}"
            path=prediction_file(Path(input_root)/f'{family}_panel',stem,partition)
            loaded[(family,window)]=validated_predictions(path,partition)
    ids=sorted(set.intersection(*(set(x[0]) for x in loaded.values())))
    if len(ids)<min_tokens: raise ValueError(f'Longitudinal paired tokens {len(ids)} < {min_tokens}: {dict(row)}')
    y=None;seeds=None; matrices={}
    for key,(these_ids,these_y,mat,these_seeds) in loaded.items():
        offsets=pd.Index(these_ids).get_indexer(ids)
        if (offsets<0).any():raise AssertionError('Intersection mismatch')
        yy=these_y[offsets]
        if y is None:y=yy;seeds=these_seeds
        elif not np.array_equal(y,yy) or not np.array_equal(seeds,these_seeds):
            raise ValueError(f'Cross-window labels/seeds mismatch for {key}: {dict(row)}')
        matrices[key]=mat[offsets]
    if len(np.unique(y))!=2:raise ValueError('Longitudinal cohort lacks both classes')
    digest=hashlib.sha256('\n'.join(map(str,ids)).encode('utf-8')).hexdigest()
    return np.asarray(ids),y,seeds,matrices,digest

def unique_groups(man):
    return man.loc[man.scope.eq('panel'),list(KEY)].drop_duplicates().reset_index(drop=True)
