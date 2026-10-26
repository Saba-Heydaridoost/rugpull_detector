"""Ablation numbers for the paper, from existing final predictions (nothing is retrained or overwritten).

Outputs go to output/test/paper_tables/. Chronological split; test partition.

  graph_only      Graph-only GNNs (trainer/results/gnn_graph_only, panel q1-q3): architecture with
                  the highest mean validation ROC-AUC, next to the official GNN selected the same way.
  cleaning        a) Mask and cohort comparisons. Baselines: summarised from cleaning_summary/
                  (all views pooled, as written there, and node_graph view without the dummy, from
                  cleaning_comparisons_annotated.csv). GNNs (official GCN/GraphSAGE/GAT,
                  trainer/results/gnn): computed here with the same estimator as cleaning_summary
                  (paired_bootstrap.bootstrap: mean per-seed AP/ROC difference, label-stratified token
                  bootstrap shared across seeds, 10,000 reps, RNG 2026, 95% percentile CI).
                    mask_effect     A = with_mask, B = no_mask, same cohort, matched tokens
                    cohort_matched  A = filtered cohort (no_mask), B = original (no_mask), on the
                                    filtered cohort's test tokens
  variants        b) DirSAGE / DirGCN, c) global-graph GraphSAGE / GCN (q1), d) HNN4RP, bagged
                  GraphSAGE and the two GNN->booster stacks, each against the official GNN of its
                  cell (the GNN with the highest mean validation ROC-AUC, as in
                  within_window_models.csv; for c) the stored final model of the same architecture).
                  Test AP / ROC-AUC per seed (mean +- sd) and the paired label-stratified bootstrap
                  Delta of the seed-ensemble scores (10,000 reps, RNG 2026, 95% percentile CI,
                  p = 2 min((#{d<=0}+1)/(B+1), (#{d>=0}+1)/(B+1))). Unadjusted, exploratory.

Usage: python paper_ablations.py --tasks graph_only cleaning variants
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from paired_bootstrap import bootstrap, fast_scores

BASE = Path(__file__).resolve().parent
RES = BASE.parent / 'trainer' / 'results'
OUT = BASE / 'output' / 'test' / 'paper_tables'
CLEAN = BASE / 'output' / 'test' / 'cleaning_summary'
GNNS = ['gcn', 'graphsage', 'gat']
CELLS = [(s, q) for s in ['panel', 'full'] for q in ['q1', 'q2', 'q3']]
B, RNG = 10000, 2026


def gnn_stem(m, s, q, variant='no_mask', cohort='original', kind='node_graph'):
    return f'{m}_{s}_{q}_gnn_graph_structure_timing_{kind}_{variant}_{cohort}'


def test_matrix(path, model=None):
    d = pd.read_csv(path)
    if model is not None and 'model' in d:
        d = d[d.model == model]
    d = d[d.partition == 'test']
    S = d.pivot(index='token_mint', columns='seed', values='y_score').sort_index()
    y = d.groupby('token_mint').y_true.first().reindex(S.index).astype(int).values
    return S, y


def val_roc(runs_path):
    r = pd.read_csv(runs_path)
    return r.loc[r.partition == 'val', 'roc_auc'].mean()


def per_seed(S, y):
    ap = np.array([average_precision_score(y, S[c]) for c in S.columns])
    roc = np.array([roc_auc_score(y, S[c]) for c in S.columns])
    return ap, roc


def fmt(a):
    return f'{a.mean():.3f} ± {a.std(ddof=1):.3f}'


def boot_p(d):
    return min(1.0, 2 * min(((d <= 0).sum() + 1) / (len(d) + 1), ((d >= 0).sum() + 1) / (len(d) + 1)))


def ens_boot(y, a, b, rng):
    """Paired label-stratified bootstrap of seed-ensemble scores; returns Delta AP and Delta ROC draws."""
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    m = np.column_stack([a, b])
    d = np.empty((B, 2))
    for r in range(B):
        i = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        ap, roc = fast_scores(y[i], m[i])
        d[r] = [ap[0] - ap[1], roc[0] - roc[1]]
    return d


def official_gnn(s, q):
    v = {m: val_roc(RES / 'gnn' / f'final_runs_{gnn_stem(m, s, q)}.csv') for m in GNNS}
    return max(v, key=v.get), v


# ------------------------------------------------------------------------------------------
def graph_only():
    rows = []
    for s, q in CELLS[:3]:
        for label, folder, kind in [('official GNN (node_graph)', 'gnn', 'node_graph'),
                                    ('graph-only GNN', 'gnn_graph_only', 'graph_only')]:
            v = {m: val_roc(RES / folder / f'final_runs_{gnn_stem(m, s, q, kind=kind)}.csv') for m in GNNS}
            m = max(v, key=v.get)
            src = RES / folder / f'final_predictions_{gnn_stem(m, s, q, kind=kind)}.csv'
            S, y = test_matrix(src)
            ap, roc = per_seed(S, y)
            rows.append({'scope': s, 'window': q, 'family': label, 'selected (best mean val ROC-AUC)': m,
                         'val ROC-AUC': v[m], 'n_test': len(y), 'n_pos': int(y.sum()), 'test AP': fmt(ap),
                         'test ROC-AUC': fmt(roc), 'source': str(src.relative_to(BASE.parent))})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / 'graph_only_vs_official_gnn.csv', index=False)
    return t


# ------------------------------------------------------------------------------------------
def cleaning():
    keep = ['scope', 'comparison', 'cohort', 'n_configurations', 'median_delta_ap', 'median_delta_roc',
            'ap_ci_A_better', 'ap_ci_B_better', 'ap_ci_includes_zero', 'roc_ci_A_better', 'roc_ci_B_better',
            'roc_ci_includes_zero']
    pooled = pd.read_csv(CLEAN / 'summary_by_cohort.csv')[keep].assign(
        family='baselines (9 models incl. dummy, 5 views pooled)', source='cleaning_summary/summary_by_cohort.csv')

    ann = pd.read_csv(CLEAN / 'cleaning_comparisons_annotated.csv')
    ann = ann[(ann.view == 'node_graph') & (ann.model != 'dummy')]
    ann = ann.assign(cohort=np.where(ann.comparison == 'mask_effect', ann.cohort, ann.cohort))
    ng = summarise(ann, 'baselines (8 models, node_graph view)', 'cleaning_summary/cleaning_comparisons_annotated.csv')

    rows = []
    for s, q in CELLS:
        for m in GNNS:
            pairs = [('mask_effect', c, gnn_stem(m, s, q, 'with_mask', c), gnn_stem(m, s, q, 'no_mask', c))
                     for c in ['original', 'drop_3_or_more_missing']]
            pairs += [('cohort_matched', c, gnn_stem(m, s, q, 'no_mask', c), gnn_stem(m, s, q, 'no_mask', 'original'))
                      for c in ['drop_3_or_more_missing', 'complete_case']]
            for comp, coh, a, b in pairs:
                pa, pb = RES / 'gnn' / f'final_predictions_{a}.csv', RES / 'gnn' / f'final_predictions_{b}.csv'
                if not (pa.is_file() and pb.is_file()):
                    rows.append({'scope': s, 'window': q, 'model': m, 'comparison': comp, 'cohort': coh,
                                 'status': 'missing file'})
                    continue
                Sa, ya = test_matrix(pa)
                Sb, yb = test_matrix(pb)
                ids = Sa.index.intersection(Sb.index)
                y = pd.Series(ya, Sa.index).loc[ids].values
                assert (y == pd.Series(yb, Sb.index).loc[ids].values).all()
                point, diffs, ci, *_ = bootstrap(y, Sa.loc[ids].values, Sb.loc[ids].values, B, RNG, .95)
                rows.append({'scope': s, 'window': q, 'model': m, 'comparison': comp, 'cohort': coh,
                             'status': 'ok', 'n_paired': len(ids), 'n_pos': int(y.sum()),
                             'delta_ap': point['delta_ap'], 'delta_ap_ci_low': ci[0, 0], 'delta_ap_ci_high': ci[1, 0],
                             'delta_roc': point['delta_roc'], 'delta_roc_ci_low': ci[0, 1],
                             'delta_roc_ci_high': ci[1, 1], 'source_a': f'trainer/results/gnn/final_predictions_{a}.csv',
                             'source_b': f'trainer/results/gnn/final_predictions_{b}.csv'})
            print(f'cleaning: {s} {q} done', flush=True)
    g = pd.DataFrame(rows)
    g.to_csv(OUT / 'gnn_cleaning_comparisons.csv', index=False)
    gs = summarise(g[g.status == 'ok'], 'official GNNs (GCN/GraphSAGE/GAT)', 'paper_tables/gnn_cleaning_comparisons.csv')
    t = pd.concat([pooled, ng, gs], ignore_index=True)
    t.to_csv(OUT / 'cleaning_summary_for_paper.csv', index=False)
    return t


def summarise(d, family, source):
    rows = []
    for (scope, comp, coh), x in d.groupby(['scope', 'comparison', 'cohort']):
        rows.append({'scope': scope, 'comparison': comp, 'cohort': coh, 'n_configurations': len(x),
                     'median_delta_ap': x.delta_ap.median(), 'median_delta_roc': x.delta_roc.median(),
                     'ap_ci_A_better': int((x.delta_ap_ci_low > 0).sum()),
                     'ap_ci_B_better': int((x.delta_ap_ci_high < 0).sum()),
                     'ap_ci_includes_zero': int(((x.delta_ap_ci_low <= 0) & (x.delta_ap_ci_high >= 0)).sum()),
                     'roc_ci_A_better': int((x.delta_roc_ci_low > 0).sum()),
                     'roc_ci_B_better': int((x.delta_roc_ci_high < 0).sum()),
                     'roc_ci_includes_zero': int(((x.delta_roc_ci_low <= 0) & (x.delta_roc_ci_high >= 0)).sum()),
                     'family': family, 'source': source})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------
def variants():
    rng = np.random.default_rng(RNG)
    rows = []
    for s, q in CELLS:
        ref, _ = official_gnn(s, q)
        ref_src = RES / 'gnn' / f'final_predictions_{gnn_stem(ref, s, q)}.csv'
        cand = [('b) direction-aware', 'dirsage', RES / 'gnn_improved' / f'final_predictions_{gnn_stem("dirsage", s, q)}.csv', None, ref),
                ('b) direction-aware', 'dirgcn', RES / 'gnn_improved' / f'final_predictions_{gnn_stem("dirgcn", s, q)}.csv', None, ref),
                ('d) HNN4RP (adapted)', 'hnn4rp', RES / 'hnn4rp' / f'final_predictions_hnn4rp_{s}_{q}_node_graph_no_mask_original.csv', None, ref),
                ('d) bagged / stacked', 'graphsage_bagged', RES / 'gnn_stacked' / f'predictions_graphsage_bagged_{s}_{q}.csv', None, ref),
                ('d) bagged / stacked', 'gradient_boosting+graphsage_stack', RES / 'gnn_stacked' / f'predictions_gradient_boosting+graphsage_stack_{s}_{q}.csv', None, ref),
                ('d) bagged / stacked', 'xgboost+graphsage_stack', RES / 'gnn_stacked' / f'predictions_xgboost+graphsage_stack_{s}_{q}.csv', None, ref)]
        if q == 'q1':
            for m in ['graphsage', 'gcn']:
                cand.append(('c) global graph (q1)', f'{m} (global graph)',
                             RES / 'gnn_global_ablation' / 'global' / f'final_predictions_{gnn_stem(m, s, q)}.csv', None, m))
        for group, name, src, filt, against in cand:
            against_src = RES / 'gnn' / f'final_predictions_{gnn_stem(against, s, q)}.csv'
            Sa, ya = test_matrix(src, filt)
            Sr, yr = test_matrix(against_src)
            ids = Sa.index.intersection(Sr.index)
            y = pd.Series(ya, Sa.index).loc[ids].values
            assert (y == pd.Series(yr, Sr.index).loc[ids].values).all() and len(ids) == len(Sr)
            ap_a, roc_a = per_seed(Sa.loc[ids], y)
            ap_r, roc_r = per_seed(Sr.loc[ids], y)
            a, r = Sa.loc[ids].mean(1).values, Sr.loc[ids].mean(1).values
            d = ens_boot(y, a, r, rng)
            rows.append({'group': group, 'scope': s, 'window': q, 'model': name,
                         'reference (official GNN)': against + (' (stored final per-token)' if group.startswith('c') else ''),
                         'n_test': len(y), 'n_pos': int(y.sum()), 'n_seeds': Sa.shape[1],
                         'test AP': fmt(ap_a), 'test ROC-AUC': fmt(roc_a),
                         'reference test AP': fmt(ap_r), 'reference test ROC-AUC': fmt(roc_r),
                         'ΔAP (ensemble)': average_precision_score(y, a) - average_precision_score(y, r),
                         'ΔAP 95% CI low': np.percentile(d[:, 0], 2.5), 'ΔAP 95% CI high': np.percentile(d[:, 0], 97.5),
                         'p (ΔAP)': boot_p(d[:, 0]),
                         'ΔROC (ensemble)': roc_auc_score(y, a) - roc_auc_score(y, r),
                         'ΔROC 95% CI low': np.percentile(d[:, 1], 2.5), 'ΔROC 95% CI high': np.percentile(d[:, 1], 97.5),
                         'p (ΔROC)': boot_p(d[:, 1]),
                         'source': str(src.relative_to(BASE.parent)), 'reference source': str(against_src.relative_to(BASE.parent))})
        print(f'variants: {s} {q} done (reference {ref})', flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / 'gnn_variants_vs_official.csv', index=False)
    return t


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tasks', nargs='+', default=['graph_only', 'cleaning', 'variants'])
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option('display.width', 320); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 300)
    for t in a.tasks:
        print(globals()[t]().round(4).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
