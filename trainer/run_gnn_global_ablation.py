"""Ablation: GCN / GraphSAGE on one global, time-respecting transaction graph (q1).

run_gnn.py builds one graph per token from that token's own recovered transfers. This
ablation keeps everything else of run_gnn.py unchanged (tokens, splits, token-level
features, preprocessing, Optuna search, seeds, thresholds, metrics) and replaces only the
graph construction:

  * All recovered q1 transfers of all tokens form ONE wallet-level transaction graph.
  * Token i is classified from its own wallets inside that global graph. Its graph is
    i's own transfers plus the transfers its wallets made in OTHER tokens, expanded for
    `HOPS` hops, using only transfers with timestamp strictly before i's prediction cutoff
    (time-respecting; nothing after the cutoff is used).
  * Hub wallets (present in more than HUB_MAX_TOKENS tokens; routers/programs) are not
    expanded through. To bound graph size, at most MAX_PER_WALLET[h] most recent foreign
    transfers per wallet are added at hop h, and at most MAX_FOREIGN_EDGES per token.
  * Edge time is re-expressed relative to token i's window, (timestamp - t0_i) / window,
    so earlier activity has negative time. Wallet features are therefore computed over
    this global neighbourhood, and an extra binary node feature marks i's own wallets.
  * Graph-level pooling runs over i's own wallets only.

Usage
-----
  python run_gnn_global_ablation.py --graph per_token   # unchanged run_gnn.py graphs (paired reference)
  python run_gnn_global_ablation.py --graph global
Results go to results/gnn_global_ablation/<graph>/ (chronological split by default).
"""
import argparse
import inspect
import os
import warnings

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings('ignore')

from gnn import config, data  # noqa: E402
import run_gnn  # noqa: E402

HUB_MAX_TOKENS = 50
HOPS = 2
MAX_PER_WALLET = (10, 3)
MAX_FOREIGN_EDGES = 4000
OWN_FEATURE = 'is_own_wallet'


class GlobalTransactionGraph:
    """Index over all recovered transfers of one observation window."""

    def __init__(self, q):
        folder = data.cohort_folder('full', q, 'original')
        cols = [config.TOKEN_COL, config.SOURCE_COL, config.TARGET_COL, config.EDGE_TIME_COL,
                config.SPLIT_COL, 'timestamp', 'tx_signature', 'observation_start', 'observation_end']
        e = data._normalise_split(pd.read_parquet(os.path.join(folder, config.EDGE_FILE), columns=cols))
        e = e.sort_values('timestamp', ascending=False, kind='mergesort').reset_index(drop=True)
        self.edges = e
        self.ts = e['timestamp'].to_numpy()
        self.tok = e[config.TOKEN_COL].to_numpy()
        src = e[config.SOURCE_COL].astype(str).to_numpy()
        dst = e[config.TARGET_COL].astype(str).to_numpy()
        self.src, self.dst = src, dst
        per_wallet = pd.concat([pd.DataFrame({'w': src, 't': self.tok}), pd.DataFrame({'w': dst, 't': self.tok})])
        n_tok = per_wallet.drop_duplicates().groupby('w').size()
        self.hubs = set(n_tok[n_tok > HUB_MAX_TOKENS].index)
        rows = np.concatenate([np.arange(len(e)), np.arange(len(e))])
        wal = np.concatenate([src, dst])
        order = np.lexsort((rows, wal))          # by wallet, then by row (= newest first)
        wal, rows = wal[order], rows[order]
        cut = np.flatnonzero(wal[1:] != wal[:-1]) + 1
        starts = np.r_[0, cut]
        ends = np.r_[cut, len(wal)]
        self.index = {wal[s]: rows[s:t] for s, t in zip(starts, ends)}
        win = e.groupby(config.TOKEN_COL)[['observation_start', 'observation_end']].first()
        self.t0 = win['observation_start'].to_dict()
        self.t_end = win['observation_end'].to_dict()
        self.stats = []
        print(f'global graph: {len(e)} transfers, {len(self.index)} wallets, {len(self.hubs)} hubs')

    def augment(self, token, own):
        t0, cutoff = self.t0[token], self.t_end[token]
        window = cutoff - t0
        own_w = set(own[config.SOURCE_COL].astype(str)) | set(own[config.TARGET_COL].astype(str))
        frontier = own_w - self.hubs
        seen = set(own_w)
        picked = []
        for hop in range(HOPS):
            nxt = set()
            for w in frontier:
                rows = self.index.get(w)
                if rows is None:
                    continue
                rows = rows[(self.ts[rows] < cutoff) & (self.tok[rows] != token)]
                rows = [r for r in rows if self.src[r] not in self.hubs and self.dst[r] not in self.hubs]
                rows = rows[:MAX_PER_WALLET[hop]]
                picked.extend(rows)
                for r in rows:
                    for v in (self.src[r], self.dst[r]):
                        if v not in seen:
                            nxt.add(v)
            seen |= nxt
            frontier = nxt
        picked = sorted(set(picked))[:MAX_FOREIGN_EDGES]   # rows are newest first
        foreign = self.edges.iloc[picked][own.columns.intersection(self.edges.columns)].copy()
        own = own.copy()
        own['own_edge'], foreign['own_edge'] = 1, 0
        aug = pd.concat([own, foreign], ignore_index=True)
        aug[config.EDGE_TIME_COL] = (aug['timestamp'] - t0) / window
        self.stats.append({'token': token, 'own_edges': len(own), 'foreign_edges': len(foreign),
                           'nodes': len(set(aug[config.SOURCE_COL].astype(str)) | set(aug[config.TARGET_COL].astype(str)))})
        return aug, own_w


def install_global_graph(q):
    """Monkeypatch run_gnn's graph construction and pooling for the global-graph ablation."""
    pool = GlobalTransactionGraph(q)
    original_build, original_nodes = data._build_graph, data._node_feature_frame
    base_features = list(config.NODE_FEATURES)
    extended_features = base_features + [OWN_FEATURE]
    config.NODE_FEATURES = extended_features

    def node_features(edges, node_order):
        config.NODE_FEATURES = base_features      # the original builder validates its own columns
        try:
            out = original_nodes(edges, node_order)
        finally:
            config.NODE_FEATURES = extended_features
        own = edges[edges['own_edge'] == 1] if 'own_edge' in edges else edges
        own_nodes = set(own[config.SOURCE_COL].astype(str)) | set(own[config.TARGET_COL].astype(str))
        out[OWN_FEATURE] = [float(n in own_nodes) for n in node_order]
        return out

    def build_graph(edges, sample_id, label, token_features):
        token = edges[config.TOKEN_COL].iloc[0]
        aug, own_w = pool.augment(token, edges)
        graph = original_build(aug, sample_id=sample_id, label=label, token_features=token_features)
        node_order = sorted(pd.concat([aug[config.SOURCE_COL], aug[config.TARGET_COL]]).dropna().astype(str).unique())
        graph.own_mask = torch.tensor([n in own_w for n in node_order], dtype=torch.bool)
        assert graph.own_mask.numel() == graph.num_nodes and graph.own_mask.any()
        return graph

    data._node_feature_frame = node_features
    data._build_graph = build_graph

    from gnn.models import common, gcn, graphsage
    current = {'mask': None}
    original_pool = common.graph_pool

    def own_only_pool(x, batch, mode):
        m = current['mask']
        if m is not None:
            x, batch = x[m], batch[m]
        return original_pool(x, batch, mode)

    for module in (gcn, graphsage):
        module.graph_pool = own_only_pool
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if issubclass(cls, torch.nn.Module) and cls.__module__ == module.__name__:
                fwd = cls.forward

                def wrapped(self, batch, *a, _fwd=fwd, **k):
                    current['mask'] = getattr(batch, 'own_mask', None)
                    try:
                        return _fwd(self, batch, *a, **k)
                    finally:
                        current['mask'] = None
                cls.forward = wrapped
    return pool


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--graph', choices=['per_token', 'global'], required=True)
    p.add_argument('--scopes', nargs='+', default=['panel', 'full'])
    p.add_argument('--models', nargs='+', default=['gcn', 'graphsage'])
    p.add_argument('--n-trials', type=int, default=None)
    p.add_argument('--n-seeds', type=int, default=None)
    p.add_argument('--result-dir', default=None)
    a = p.parse_args()

    config.RESULT_DIR = os.path.abspath(a.result_dir or os.path.join(config.RESULT_ROOT, 'gnn_global_ablation', a.graph))
    config.PLOT_DIR = os.path.join(config.RESULT_DIR, 'plots')
    os.makedirs(config.PLOT_DIR, exist_ok=True)
    print(f'split={config.SPLIT} graph={a.graph} -> {config.RESULT_DIR}')

    pool = install_global_graph('q1') if a.graph == 'global' else None
    common = dict(models=a.models, experiments=['q1'], cohorts=['original'], feature_variants=['base'],
                  variants=['no_mask'], n_trials=a.n_trials, n_seeds=a.n_seeds, refit=False, skip_existing=True)
    for scope in a.scopes:
        # Same two-stage protocol as run_gnn.py: select on validation with test frozen,
        # then re-run the frozen configuration with test scoring enabled (final_* files).
        config.EVALUATE_TEST = False
        run_gnn.run(scopes=[scope], reuse_best_params=False, **common)
        config.EVALUATE_TEST = True
        run_gnn.run(scopes=[scope], reuse_best_params=True, **common)
    if pool is not None:
        pd.DataFrame(pool.stats).drop_duplicates('token').to_csv(
            os.path.join(config.RESULT_DIR, 'global_graph_sizes.csv'), index=False)


if __name__ == '__main__':
    main()
