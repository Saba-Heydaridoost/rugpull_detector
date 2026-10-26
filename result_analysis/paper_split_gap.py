"""Train / validation / test performance and the reasons for the gap (paper section on the split gap).

Reads existing results only; outputs go to output/test/paper_tables/.

split_gap_models.csv
    For the models of within_window_models.csv (per cell: the GNN and the node_graph non-graph
    model with the highest mean validation ROC-AUC), per-seed mean +- sd of AP and ROC-AUC on the
    training, validation and test partitions, and the validation AP re-weighted to the test
    prevalence (positives weighted by odds(test)/odds(validation), seed-ensemble scores), which
    removes the part of the validation-test AP gap caused by the higher validation prevalence.
    Train metrics come from the run files (per seed; train scores are not stored).
    Sources: GNN trainer/results/gnn/final_runs_* and final_predictions_*; baselines
    result_analysis/input/baseline_<scope>/runs_* (train, val) and trainer/results/final_predictions_*.
split_gap_selection.csv
    Selection optimism: for the same models, the best and median validation ROC-AUC over the Optuna
    trials (sweep_* files) next to the mean validation ROC-AUC of the five retrained seeds.
split_shift_diagnostics.csv
    Per window and partition (original cohort, after embargo): samples, rug-pull rate, share of
    missing token-level predictor cells, graph-bearing share, and share of pump.fun mints
    (mint address ending in 'pump').
    Sources: trainer/artifacts/final/missingness_experiments/original/<q>/model_nodes_raw_nan.parquet
    and model_graph_features_with_split.parquet.

Usage: python paper_split_gap.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
RES = ROOT / 'trainer' / 'results'
INP = BASE / 'input'
ART = ROOT / 'trainer' / 'artifacts' / 'final' / 'missingness_experiments' / 'original'
OUT = BASE / 'output' / 'test' / 'paper_tables'
GNN_SUFFIX = 'gnn_graph_structure_timing_node_graph_no_mask_original'
GNNS = ['gcn', 'graphsage', 'gat']
NODE = ['volume_sell_change_percent', 'volume_buy_change_percent', 'unique_wallet_count',
        'unique_wallet_change_percent', 'liquidity_at_observation_end']


def files(m, s, q):
    if m in GNNS:
        return (RES / 'gnn' / f'final_runs_{m}_{s}_{q}_{GNN_SUFFIX}.csv',
                RES / 'gnn' / f'final_predictions_{m}_{s}_{q}_{GNN_SUFFIX}.csv',
                RES / 'gnn' / f'sweep_{m}_{s}_{q}_{GNN_SUFFIX}.csv')
    tag = f'{m}_{s}_{q}_node_graph_no_mask_original'
    return (INP / f'baseline_{s}' / f'runs_{tag}.csv', RES / f'final_predictions_{tag}.csv',
            INP / f'baseline_{s}' / f'sweep_{tag}.csv')


def fmt(a):
    return f'{np.mean(a):.3f} ± {np.std(a, ddof=1):.3f}'


def ensemble(pred, part):
    d = pred[pred.partition == part]
    S = d.pivot(index='token_mint', columns='seed', values='y_score')
    y = d.groupby('token_mint').y_true.first().reindex(S.index).astype(int).values
    return S.mean(1).values, y


def main():
    sel = pd.read_csv(OUT / 'within_window_models.csv')
    sel = sel[sel.role.str.startswith(('GNN', 'baseline'))]
    rows, opt = [], []
    for _, r in sel.iterrows():
        s, q, m = r.scope, r.window, r.model
        runs_f, pred_f, sweep_f = files(m, s, q)
        runs, pred = pd.read_csv(runs_f), pd.read_csv(pred_f)
        test = pred[pred.partition == 'test']
        tst = [(average_precision_score(g.y_true, g.y_score), g) for _, g in test.groupby('seed')]
        from sklearn.metrics import roc_auc_score
        t_ap = np.array([a for a, _ in tst])
        t_roc = np.array([roc_auc_score(g.y_true, g.y_score) for _, g in tst])
        tr, va = runs[runs.partition == 'train'], runs[runs.partition == 'val']
        ev, yv = ensemble(pred, 'val')
        et, yt = ensemble(pred, 'test')
        pv, pt = yv.mean(), yt.mean()
        w = np.where(yv == 1, (pt / (1 - pt)) / (pv / (1 - pv)), 1.0)
        rows.append({'scope': s, 'window': q, 'role': 'GNN' if m in GNNS else 'non-graph', 'model': m,
                     'prevalence train': np.nan, 'prevalence val': pv, 'prevalence test': pt,
                     'train AP': fmt(tr.pr_auc), 'val AP': fmt(va.pr_auc), 'test AP': fmt(t_ap),
                     'train ROC-AUC': fmt(tr.roc_auc), 'val ROC-AUC': fmt(va.roc_auc), 'test ROC-AUC': fmt(t_roc),
                     'val AP (ensemble)': average_precision_score(yv, ev),
                     'val AP at test prevalence (ensemble)': average_precision_score(yv, ev, sample_weight=w),
                     'test AP (ensemble)': average_precision_score(yt, et),
                     'runs source': str(runs_f.relative_to(ROOT)), 'predictions source': str(pred_f.relative_to(ROOT))})
        sw = pd.read_csv(sweep_f)
        opt.append({'scope': s, 'window': q, 'model': m, 'trials': len(sw),
                    'best trial val ROC-AUC': sw.val_roc_auc.max(), 'median trial val ROC-AUC': sw.val_roc_auc.median(),
                    'retrained seeds val ROC-AUC (mean)': va.roc_auc.mean(), 'test ROC-AUC (mean)': t_roc.mean(),
                    'sweep source': str(sweep_f.relative_to(ROOT))})
    g = pd.DataFrame(rows)
    for q in ['q1', 'q2', 'q3']:
        n = pd.read_parquet(ART / q / 'model_nodes_raw_nan.parquet')
        g.loc[g.window == q, 'prevalence train'] = np.nan
    g.to_csv(OUT / 'split_gap_models.csv', index=False)
    pd.DataFrame(opt).to_csv(OUT / 'split_gap_selection.csv', index=False)

    diag = []
    for q in ['q1', 'q2', 'q3']:
        n = pd.read_parquet(ART / q / 'model_nodes_raw_nan.parquet')
        gb = set(pd.read_parquet(ART / q / 'model_graph_features_with_split.parquet', columns=['token_mint']).token_mint)
        for part, d in n.groupby('split'):
            diag.append({'window': q, 'partition': part, 'samples': len(d), 'rug-pull rate': d.rug_pull_label.mean(),
                         'missing predictor cells': d[NODE].isna().to_numpy().mean(),
                         'graph-bearing share': d.token_mint.isin(gb).mean(),
                         'pump.fun mint share': d.token_mint.str.endswith('pump').mean(),
                         'creation from': pd.to_datetime(d.creation_time, unit='s').min().date(),
                         'creation to': pd.to_datetime(d.creation_time, unit='s').max().date()})
    pd.DataFrame(diag).to_csv(OUT / 'split_shift_diagnostics.csv', index=False)
    pd.set_option('display.width', 320); pd.set_option('display.max_columns', 30)
    print(g.drop(columns=['runs source', 'predictions source', 'prevalence train']).round(3).to_string(index=False))
    print(pd.DataFrame(opt).drop(columns='sweep source').round(3).to_string(index=False))
    print(pd.DataFrame(diag).round(3).to_string(index=False))


if __name__ == '__main__':
    main()
