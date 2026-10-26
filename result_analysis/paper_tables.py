"""Tables for the paper, computed from the existing final results (nothing is retrained or overwritten).

Setting: chronological split, original cohort, no_mask; GNNs are the official ROC-tuned
GCN / GraphSAGE / GAT (trainer/results/gnn final_*); baselines are the official runs
(validation metrics: result_analysis/input/baseline_<scope>/runs_*; test predictions:
trainer/results/final_predictions_*).

Tasks (all outputs in output/test/paper_tables/):
  within    Exploratory within-window table. Per cell (panel/full x q1-q3, node_graph view):
            GNN with the highest mean validation ROC-AUC vs the non-graph model with the highest
            mean validation ROC-AUC, plus XGBoost as a fixed reference and the dummy model.
            Test AP / ROC-AUC / Brier per seed (mean +- sd); paired label-stratified bootstrap
            of the seed-ensemble scores (10,000 reps, RNG 2026) with 95% percentile CI and
            p = min(1, 2 * min((#{d<=0}+1)/(B+1), (#{d>=0}+1)/(B+1))). Contrasts are
            de-duplicated (one contrast when the selected baseline is XGBoost); Holm is
            applied across these contrasts, separately for AP and ROC-AUC.
  primary   The nine frozen primary contrasts with each model's mean per-seed test AP per window.
  views     Feature-view ablation: best baseline by mean validation ROC-AUC within each view
            (node, node_matched, graph, node_graph); graph-only GNNs vs the official GNNs (panel).
  power     Minimum detectable Delta AP (80% power) for GNN vs XGBoost per cell (see power()).
  liqcut    Samples whose liquidity-at-cutoff snapshot could lie exactly at t_q (see liqcut()).

Usage: python paper_tables.py --tasks within primary views power liqcut
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
RES = ROOT / 'trainer' / 'results'
INP = BASE / 'input'
OUT = BASE / 'output' / 'test' / 'paper_tables'
GNN_SUFFIX = 'gnn_graph_structure_timing_node_graph_no_mask_original'
GNNS = ['gcn', 'graphsage', 'gat']
BASELINES = ['logistic_regression', 'decision_tree', 'random_forest', 'gradient_boosting', 'xgboost', 'mlp',
             'ft_transformer', 'autoint']
CELLS = [(s, q) for s in ['panel', 'full'] for q in ['q1', 'q2', 'q3']]
B, RNG = 10000, 2026


# ------------------------------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------------------------------
def gnn_files(m, s, q, folder='gnn', suffix=GNN_SUFFIX):
    return RES / folder / f'final_runs_{m}_{s}_{q}_{suffix}.csv', RES / folder / f'final_predictions_{m}_{s}_{q}_{suffix}.csv'


def base_files(m, s, q, view='node_graph'):
    tag = f'{m}_{s}_{q}_{view}_no_mask_original'
    return INP / f'baseline_{s}' / f'runs_{tag}.csv', RES / f'final_predictions_{tag}.csv'


def val_roc(runs_path):
    r = pd.read_csv(runs_path)
    return r.loc[r.partition == 'val', 'roc_auc'].mean()


def test_matrix(pred_path):
    d = pd.read_csv(pred_path)
    d = d[d.partition == 'test']
    S = d.pivot(index='token_mint', columns='seed', values='y_score').sort_index()
    y = d.groupby('token_mint').y_true.first().reindex(S.index).astype(int).values
    return S, y


def per_seed(S, y):
    ap = [average_precision_score(y, S[c]) for c in S.columns]
    roc = [roc_auc_score(y, S[c]) for c in S.columns]
    br = [brier_score_loss(y, S[c]) for c in S.columns]
    return np.array(ap), np.array(roc), np.array(br)


def fmt(a):
    return f'{np.mean(a):.3f} ± {np.std(a, ddof=1):.3f}'


def boot_p(d):
    return min(1.0, 2 * min(((d <= 0).sum() + 1) / (len(d) + 1), ((d >= 0).sum() + 1) / (len(d) + 1)))


def paired_boot(y, a, b, rng, reps=B):
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    da, dr = np.empty(reps), np.empty(reps)
    for r in range(reps):
        i = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        da[r] = average_precision_score(y[i], a[i]) - average_precision_score(y[i], b[i])
        dr[r] = roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i])
    return da, dr


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[o])
    out = np.empty_like(p)
    out[o] = np.minimum(adj, 1)
    return out


# ------------------------------------------------------------------------------------------
# Task 1: within-window table
# ------------------------------------------------------------------------------------------
def within():
    rng = np.random.default_rng(RNG)
    models, contrasts = [], []
    for s, q in CELLS:
        gv = {m: val_roc(gnn_files(m, s, q)[0]) for m in GNNS}
        bv = {m: val_roc(base_files(m, s, q)[0]) for m in BASELINES}
        g, b = max(gv, key=gv.get), max(bv, key=bv.get)
        mats = {}
        for role, m, path, vroc in [('GNN (best val ROC)', g, gnn_files(g, s, q)[1], gv[g]),
                                    ('baseline (best val ROC)', b, base_files(b, s, q)[1], bv[b]),
                                    ('XGBoost (fixed reference)', 'xgboost', base_files('xgboost', s, q)[1], bv['xgboost']),
                                    ('dummy (prevalence)', 'dummy', base_files('dummy', s, q)[1], np.nan)]:
            S, y = test_matrix(path)
            mats[m] = (S, y)
            ap, roc, br = per_seed(S, y)
            models.append({'scope': s, 'window': q, 'role': role, 'model': m, 'n_test': len(y), 'n_pos': int(y.sum()),
                           'prevalence': y.mean(), 'val ROC-AUC (mean)': vroc, 'test AP': fmt(ap), 'test ROC-AUC': fmt(roc),
                           'test Brier (mean)': br.mean(), 'test AP (seed-ensemble)': average_precision_score(y, S.mean(1)),
                           'test ROC-AUC (seed-ensemble)': roc_auc_score(y, S.mean(1))})
        Sg, yg = mats[g]
        for ref in dict.fromkeys([b, 'xgboost']):              # de-duplicated, order kept
            Sr, yr = mats[ref]
            assert (Sg.index == Sr.index).all() and (yg == yr).all()
            a, c = Sg.mean(1).values, Sr.mean(1).values
            da, dr = paired_boot(yg, a, c, rng)
            contrasts.append({'scope': s, 'window': q, 'GNN': g, 'reference': ref,
                              'reference role': 'best val ROC baseline' + (' = XGBoost' if ref == b == 'xgboost' else '')
                              if ref == b else 'XGBoost (fixed reference)',
                              'n_test': len(yg), 'n_pos': int(yg.sum()),
                              'ΔAP (ensemble)': average_precision_score(yg, a) - average_precision_score(yg, c),
                              'ΔAP 95% CI low': np.percentile(da, 2.5), 'ΔAP 95% CI high': np.percentile(da, 97.5),
                              'p (ΔAP)': boot_p(da),
                              'ΔROC (ensemble)': roc_auc_score(yg, a) - roc_auc_score(yg, c),
                              'ΔROC 95% CI low': np.percentile(dr, 2.5), 'ΔROC 95% CI high': np.percentile(dr, 97.5),
                              'p (ΔROC)': boot_p(dr), 'bootstrap reps': B, 'rng seed': RNG})
        print(f'within: {s} {q} done (GNN={g}, baseline={b})', flush=True)
    mt, ct = pd.DataFrame(models), pd.DataFrame(contrasts)
    ct['p (ΔAP) Holm'] = holm(ct['p (ΔAP)'])
    ct['p (ΔROC) Holm'] = holm(ct['p (ΔROC)'])
    ct['Holm family size'] = len(ct)
    mt.to_csv(OUT / 'within_window_models.csv', index=False)
    ct.to_csv(OUT / 'within_window_contrasts.csv', index=False)
    return mt, ct


# ------------------------------------------------------------------------------------------
# Task 2: primary contrasts with per-window model AP
# ------------------------------------------------------------------------------------------
def primary():
    p = pd.read_csv(BASE / 'output' / 'test' / 'gnn_vs_baseline' / 'primary_bonferroni_reps_10000_seed_2026'
                    / 'primary_simultaneous_ci_results.csv')
    rows = []
    for m in GNNS + ['xgboost']:
        for q in ['q1', 'q2', 'q3']:
            path = (INP / 'gnn_panel' / f'final_predictions_{m}_panel_{q}_{GNN_SUFFIX}.csv' if m in GNNS
                    else INP / 'baseline_panel' / f'final_predictions_{m}_panel_{q}_node_graph_no_mask_original.csv')
            S, y = test_matrix(path)
            ap, roc, _ = per_seed(S, y)
            rows.append({'model': m, 'window': q, 'n_test': len(y), 'n_pos': int(y.sum()),
                         'mean per-seed test AP': ap.mean(), 'sd': ap.std(ddof=1), 'mean per-seed test ROC-AUC': roc.mean()})
    ap = pd.DataFrame(rows)
    p.to_csv(OUT / 'primary_contrasts.csv', index=False)
    ap.to_csv(OUT / 'primary_model_ap_by_window.csv', index=False)
    return p, ap


# ------------------------------------------------------------------------------------------
# Task 4: feature-view ablation
# ------------------------------------------------------------------------------------------
def views():
    rows = []
    for s, q in CELLS:
        for view in ['node', 'node_matched', 'graph', 'node_graph']:
            vr = {m: val_roc(base_files(m, s, q, view)[0]) for m in BASELINES}
            b = max(vr, key=vr.get)
            S, y = test_matrix(base_files(b, s, q, view)[1])
            ap, roc, _ = per_seed(S, y)
            rows.append({'scope': s, 'window': q, 'family': 'baseline', 'view': view, 'model (best val ROC)': b,
                         'val ROC-AUC': vr[b], 'n_test': len(y), 'n_pos': int(y.sum()), 'test AP': fmt(ap),
                         'test ROC-AUC': fmt(roc)})
        for label, folder, suffix in [('GNN, node_graph (official)', 'gnn', GNN_SUFFIX),
                                      ('GNN, graph only', 'gnn_graph_only',
                                       'gnn_graph_structure_timing_graph_only_no_mask_original')]:
            for m in GNNS:
                runs, pred = gnn_files(m, s, q, folder, suffix)
                if not pred.is_file():
                    continue
                S, y = test_matrix(pred)
                ap, roc, _ = per_seed(S, y)
                rows.append({'scope': s, 'window': q, 'family': label, 'view': 'gnn_graph', 'model (best val ROC)': m,
                             'val ROC-AUC': val_roc(runs), 'n_test': len(y), 'n_pos': int(y.sum()),
                             'test AP': fmt(ap), 'test ROC-AUC': fmt(roc)})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / 'feature_view_ablation.csv', index=False)
    return t


# ------------------------------------------------------------------------------------------
# Task 5: power
# ------------------------------------------------------------------------------------------
def power():
    """Normal-approximation minimum detectable effect from the paired bootstrap.

    For each cell and each official GNN against XGBoost, the seed-ensemble test scores of both
    models are resampled jointly (label-stratified paired token bootstrap, 10,000 reps, RNG 2026);
    the standard deviation of the bootstrap Delta AP is the standard error of Delta AP on this
    test set, and it already contains the observed correlation of the two models' scores
    (reported as the Spearman correlation of the scores). For a two-sided test at level alpha
    with power 0.8, MDE = (z_{1-alpha/2} + z_{0.8}) * SE. Reported for alpha = 0.05 (one
    contrast) and alpha = 0.05/9 (Bonferroni, primary family). The SE is estimated at the
    observed effect, not under the null, and assumes approximate normality of Delta AP.
    """
    rng = np.random.default_rng(RNG)
    z80 = norm.ppf(0.8)
    rows = []
    for s, q in CELLS:
        Sx, yx = test_matrix(base_files('xgboost', s, q)[1])
        for m in GNNS:
            Sg, yg = test_matrix(gnn_files(m, s, q)[1])
            assert (Sg.index == Sx.index).all()
            a, c = Sg.mean(1).values, Sx.mean(1).values
            da, _ = paired_boot(yg, a, c, rng)
            se = da.std(ddof=1)
            rows.append({'scope': s, 'window': q, 'GNN': m, 'reference': 'xgboost', 'n_test': len(yg),
                         'n_pos': int(yg.sum()), 'Spearman(score GNN, score XGB)': spearmanr(a, c)[0],
                         'observed ΔAP': average_precision_score(yg, a) - average_precision_score(yg, c),
                         'SE(ΔAP) bootstrap': se,
                         'MDE ΔAP, α=0.05, 80% power': (norm.ppf(0.975) + z80) * se,
                         'MDE ΔAP, α=0.05/9, 80% power': (norm.ppf(1 - 0.05 / 18) + z80) * se})
        print(f'power: {s} {q} done', flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / 'power_mde.csv', index=False)
    return t


# ------------------------------------------------------------------------------------------
# Task 6: liquidity snapshot at exactly t_q
# ------------------------------------------------------------------------------------------
def liqcut():
    """The collector stores the liquidity value but not the snapshot timestamp, so exact matches
    cannot be counted directly. Birdeye 1-minute snapshots are checked for alignment to the unix
    minute using the stored liquidity-drop times (liquidity_drop_factor_time, also 1-minute
    snapshots); if they are aligned, a snapshot can equal t_q only when t_q is a multiple of 60 s,
    which gives an upper bound on the number of exact matches."""
    lab = pd.read_csv(ROOT / 'dataset' / 'rule_based_labels.csv')
    tau = pd.to_numeric(lab.liquidity_drop_factor_time, errors='coerce').dropna()
    nodes = pd.read_csv(ROOT / 'dataset' / 'graph_nodes.csv')
    rows = []
    for q, g in nodes.groupby('experiment_id'):
        has = g.liquidity_at_observation_end.notna()
        aligned = (g.observation_end % 60 == 0)
        rows.append({'window': q, 'samples': len(g), 'liquidity at cutoff present': int(has.sum()),
                     't_q multiple of 60 s': int(aligned.sum()),
                     'upper bound: present and t_q multiple of 60 s': int((has & aligned).sum()),
                     'snapshot timestamp stored': False})
    t = pd.DataFrame(rows)
    t['liquidity snapshot times (tau) on unix-minute grid'] = f'{(tau % 60 == 0).mean():.4f} of {len(tau)}'
    t.to_csv(OUT / 'liquidity_cutoff_snapshot_check.csv', index=False)
    return t


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tasks', nargs='+', default=['within', 'primary', 'views', 'power', 'liqcut'])
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option('display.width', 320); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 200)
    for t in a.tasks:
        res = globals()[t]()
        for df in (res if isinstance(res, tuple) else (res,)):
            print(df.round(4).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
