"""Per-seed "Val@test" for tab:split-gap (prevalence-adjusted validation AP).

For each of the 12 models of within_window_models.csv (6 cells x selected GNN and selected
non-graph model) and each of the five seeds, validation AP is recomputed with every validation
positive weighted by w = odds(test prevalence) / odds(validation prevalence), negatives weight 1
(sklearn average_precision_score with sample_weight). This makes the weighted validation set
have the test prevalence while keeping the score ranking unchanged. Reported: mean over seeds,
directly comparable to the per-seed-mean Val and Test columns of the table.

Sources: validation and test scores per seed from trainer/results/gnn/final_predictions_* (GNNs)
and trainer/results/final_predictions_* (non-graph models); Train/Val AP per seed from the run
files (trainer/results/gnn/final_runs_* and result_analysis/input/baseline_<scope>/runs_*).
Output: output/test/paper_tables/split_gap_valattest_perseed.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import paper_split_gap as g

OUT = g.OUT


def main():
    sel = pd.read_csv(OUT / 'within_window_models.csv')
    sel = sel[sel.role.str.startswith(('GNN', 'baseline'))]
    rows = []
    for _, r in sel.iterrows():
        runs_f, pred_f, _ = g.files(r.model, r.scope, r.window)
        runs, pred = pd.read_csv(runs_f), pd.read_csv(pred_f)
        val, test = pred[pred.partition == 'val'], pred[pred.partition == 'test']
        pv, pt = val.groupby('token_mint').y_true.first().mean(), test.groupby('token_mint').y_true.first().mean()
        w_pos = (pt / (1 - pt)) / (pv / (1 - pv))
        v, vt, t = [], [], []
        for s in sorted(val.seed.unique()):
            a, b = val[val.seed == s], test[test.seed == s]
            v.append(average_precision_score(a.y_true, a.y_score))
            vt.append(average_precision_score(a.y_true, a.y_score, sample_weight=np.where(a.y_true == 1, w_pos, 1.0)))
            t.append(average_precision_score(b.y_true, b.y_score))
        run_val = runs.loc[runs.partition == 'val', 'pr_auc'].mean()
        V, VT, T = np.mean(v), np.mean(vt), np.mean(t)
        rows.append({'scope': r.scope, 'window': r.window, 'model': r.model,
                     'val prevalence': pv, 'test prevalence': pt, 'weight on positives': w_pos,
                     'Train (per-seed mean)': runs.loc[runs.partition == 'train', 'pr_auc'].mean(),
                     'Val (per-seed mean)': V, 'Val from run file': run_val,
                     'Val@test (per-seed mean)': VT, 'Test (per-seed mean)': T,
                     'Val - Val@test': V - VT, 'Val@test - Test': VT - T,
                     'share of gap (%)': 100 * (V - VT) / (V - T),
                     'runs source': str(runs_f.relative_to(g.ROOT)), 'predictions source': str(pred_f.relative_to(g.ROOT))})
    d = pd.DataFrame(rows)
    d.to_csv(OUT / 'split_gap_valattest_perseed.csv', index=False)
    pd.set_option('display.width', 250)
    print(d.drop(columns=['runs source', 'predictions source']).round(3).to_string(index=False))
    for c in ['Val - Val@test', 'share of gap (%)', 'Val@test - Test']:
        print(f'{c}: min {d[c].min():.3f}, max {d[c].max():.3f}')
    print('max |Val(pred) - Val(run file)|:', (d['Val (per-seed mean)'] - d['Val from run file']).abs().max())


if __name__ == '__main__':
    main()
