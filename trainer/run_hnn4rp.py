"""HNN4RP (Li, Jin, Wu & Xiao, JEIT 2025, doi:10.11999/JEIT250160) adapted to this project.

HNN4RP classifies nodes of a heterogeneous transaction graph (contract nodes + regular address
nodes) with three modules: (1) random walk with restart (RWR) neighbour sampling, keeping per
node type the top-k most frequently visited nodes; (2) heterogeneous node feature extraction
with a 2-layer MLP per type (contract nodes additionally carry a Par2Vec opcode embedding);
(3) temporal fusion: a BiLSTM over each type's neighbour sequence, node-level attention,
type-level attention, concatenation with the node's own representation and a sigmoid output.

Adaptation (Solana, early prediction):
  * Node types: token nodes (the mint plays the role of the token contract) and wallet nodes.
    Edges: token-wallet participation and wallet->wallet transfers from the recovered transfer
    pool of the window (trainer artifacts, full scope, original cohort). Targets are token nodes.
  * Contract code (Par2Vec over EVM opcodes) has no Solana counterpart: SPL tokens share the
    Token Program and carry no per-token bytecode. Token-node attribute features are the 14
    node_graph predictors of the matched baselines (quantile-transformed, fitted on training).
  * Wallet-node transaction features are computed at the target's prediction cutoff:
    out/in transfers, distinct counterparties, distinct tokens, age and recency (log1p).
  * Leakage control: walks only traverse transfers with timestamp < target cutoff and only
    enter tokens whose own observation window has closed by then.
  * Neighbour sequences are time-ordered (wallets by first activity, tokens by creation).
  * Hyperparameters from the paper: k = 16 neighbours per type, d = 128, Adam lr 1e-3,
    batch 128, 100 epochs, gradient clipping 5, Xavier (input) / orthogonal (recurrent)
    LSTM initialisation. Unspecified in the paper, taken from HetGNN: restart probability
    0.5, walk length 100. Best epoch selected on validation AP (the project's selection
    metric); 5 seeds; F1-max threshold on validation.

Ablations: --variant full | no_token_nbrs | no_wallet_nbrs | no_lstm
Outputs: results[_stratified]/hnn4rp/{runs,final_predictions}_hnn4rp[-variant]_<scope>_<q>_node_graph_no_mask_original.csv
"""
import argparse
import copy
import os
import time
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score  # noqa: E402

from baselines import data as bdata  # noqa: E402
from gnn import config as gconfig, data as gdata  # noqa: E402

OUT = Path(gconfig.RESULT_ROOT) / 'hnn4rp'          # results/ (chronological) or results_stratified/ (RUGPULL_SPLIT=stratified)
P_RESTART, WALK_LEN, TOPK = 0.5, 100, 16
DIM, LR, BATCH, EPOCHS, CLIP = 128, 1e-3, 128, 100, 5.0
SEEDS = range(5)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------------------------------------------
# Heterogeneous transaction graph with time-respecting access
# ---------------------------------------------------------------------------------------------
class TxGraph:
    def __init__(self, q):
        folder = gdata.cohort_folder('full', q, 'original')
        e = pd.read_parquet(os.path.join(folder, gconfig.EDGE_FILE),
                            columns=['token_mint', 'source_address', 'target_address', 'timestamp', 'observation_end'])
        self.tokens = pd.Index(sorted(e.token_mint.unique()))
        self.T = len(self.tokens)
        wallets = pd.Index(pd.unique(np.concatenate([e.source_address.values, e.target_address.values])))
        self.W = len(wallets)
        t = self.tokens.get_indexer(e.token_mint)
        s = wallets.get_indexer(e.source_address) + self.T
        d = wallets.get_indexer(e.target_address) + self.T
        ts = e.timestamp.to_numpy(np.int64)
        self.cutoff = e.groupby('token_mint').observation_end.first().reindex(self.tokens).to_numpy(np.int64)

        # Walk adjacency over all nodes (undirected for walking), sorted by time per node.
        u = np.concatenate([t, s, t, d, s, d])
        v = np.concatenate([s, t, d, t, d, s])
        tt = np.tile(ts, 6)
        o = np.lexsort((tt, u))
        self.nbr, self.nts = v[o], tt[o]
        self.indptr = np.r_[0, np.cumsum(np.bincount(u, minlength=self.T + self.W))]

        # Wallet transaction histories (time-sorted) with cumulative statistics.
        df = pd.DataFrame({'u': np.concatenate([s, d]) - self.T, 'cp': np.concatenate([d, s]),
                           'tok': np.concatenate([t, t]), 'out': np.r_[np.ones(len(s)), np.zeros(len(d))],
                           'ts': np.concatenate([ts, ts])})
        df = df.sort_values(['u', 'ts'], kind='mergesort').reset_index(drop=True)
        g = df.groupby('u', sort=False)
        self.w_ts = df.ts.to_numpy(np.int64)
        self.w_out = g.out.cumsum().to_numpy()
        self.w_in = (1 - df.out).groupby(df.u, sort=False).cumsum().to_numpy()
        self.w_cp = (~df.duplicated(['u', 'cp'])).groupby(df.u, sort=False).cumsum().to_numpy()
        self.w_tok = (~df.duplicated(['u', 'tok'])).groupby(df.u, sort=False).cumsum().to_numpy()
        self.w_ptr = np.r_[0, np.cumsum(np.bincount(df.u, minlength=self.W))]
        print(f'[{q}] graph: {self.T} tokens, {self.W} wallets, {len(e)} transfers', flush=True)

    def n_valid(self, node, c):
        a, b = self.indptr[node], self.indptr[node + 1]
        return a, int(np.searchsorted(self.nts[a:b], c, side='left'))

    def wallet_features(self, w, c):
        a, b = self.w_ptr[w - self.T], self.w_ptr[w - self.T + 1]
        k = a + int(np.searchsorted(self.w_ts[a:b], c, side='left'))
        if k == a:
            return np.zeros(6), np.inf
        i = k - 1
        f = [self.w_out[i], self.w_in[i], self.w_cp[i], self.w_tok[i],
             (c - self.w_ts[a]) / 60.0, (c - self.w_ts[i]) / 60.0]
        return np.log1p(np.maximum(f, 0)), self.w_ts[a]

    def rwr(self, i, featured, rng):
        """Random walk with restart from token i; returns visit counts of admissible nodes."""
        c = self.cutoff[i]
        counts, cur, steps, tries = Counter(), i, 0, 0
        while steps < WALK_LEN and tries < WALK_LEN * 50:
            tries += 1
            if cur != i and rng.random() < P_RESTART:
                cur = i
                continue
            a, n = self.n_valid(cur, c)
            if n == 0:
                if cur == i:
                    break
                cur = i
                continue
            nxt = int(self.nbr[a + rng.integers(n)])
            if nxt < self.T and nxt != i and (not featured[nxt] or self.cutoff[nxt] > c):
                continue                                        # token not yet observable / no features
            if nxt == i:
                cur = i
                continue
            cur = nxt
            steps += 1
            counts[nxt] += 1
        return counts


def sample_neighbourhoods(g, featured, targets):
    """Top-k tokens and wallets per target (time-ordered), plus wallet features at the cutoff."""
    rng = np.random.default_rng(0)
    out = {}
    for i in targets:
        counts = g.rwr(i, featured, rng)
        toks = sorted([n for n in counts if n < g.T], key=lambda n: (-counts[n], n))[:TOPK]
        wals = sorted([n for n in counts if n >= g.T], key=lambda n: (-counts[n], n))[:TOPK]
        toks = sorted(toks, key=lambda n: g.cutoff[n])
        wf = [g.wallet_features(w, g.cutoff[i]) for w in wals]
        order = np.argsort([first for _, first in wf], kind='stable')
        out[i] = (np.array(toks, dtype=np.int64), np.array([wf[j][0] for j in order]).reshape(-1, 6))
    return out


# ---------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------
def mlp(i, d):
    return nn.Sequential(nn.Linear(i, d), nn.ReLU(), nn.Linear(d, d))


class HNN4RP(nn.Module):
    def __init__(self, tok_dim, wal_dim, d=DIM, use_tok=True, use_wal=True, use_lstm=True):
        super().__init__()
        self.types = [t for t, on in [('tok', use_tok), ('wal', use_wal)] if on]
        self.use_lstm = use_lstm
        self.f = nn.ModuleDict({'tok': mlp(tok_dim, d), 'wal': mlp(wal_dim, d)})
        self.lstm = nn.ModuleDict({t: nn.LSTM(d, d // 2, batch_first=True, bidirectional=True) for t in ['tok', 'wal']})
        self.W_node = nn.ModuleDict({t: nn.Linear(d, d, bias=False) for t in ['tok', 'wal']})
        self.W_type = nn.ModuleDict({t: nn.Linear(d, d) for t in ['tok', 'wal']})
        self.q = nn.Parameter(torch.empty(d).uniform_(-0.1, 0.1))
        self.out = nn.Linear(2 * d, 1)
        for lstm in self.lstm.values():
            for name, p in lstm.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(p)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(p)
                else:
                    nn.init.zeros_(p)

    def encode(self, t, x, mask):
        h = self.f[t](x)
        if not self.use_lstm:
            return h
        lengths = mask.sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(h, lengths, batch_first=True, enforce_sorted=False)
        y, _ = self.lstm[t](packed)
        y, _ = nn.utils.rnn.pad_packed_sequence(y, batch_first=True, total_length=h.shape[1])
        return y

    def forward(self, x_self, seqs):
        e = self.f['tok'](x_self)                                           # target = token node
        aggs, present = [], []
        for t in self.types:
            x, mask = seqs[t]
            H = self.encode(t, x, mask)
            s = F.leaky_relu((self.W_node[t](H) * self.W_node[t](e).unsqueeze(1)).sum(-1))   # eq. (3)
            s = s.masked_fill(~mask, float('-inf'))
            a = torch.nan_to_num(torch.softmax(s, dim=1), nan=0.0)          # eq. (4); empty -> 0
            aggs.append((a.unsqueeze(-1) * H).sum(1))                       # eq. (5)
            present.append(mask.any(1))
        if aggs:
            A = torch.stack(aggs, 1)
            st = torch.stack([(torch.tanh(self.W_type[t](A[:, j])) * self.q).sum(-1)
                              for j, t in enumerate(self.types)], 1)        # type-level attention
            st = st.masked_fill(~torch.stack(present, 1), float('-inf'))
            beta = torch.nan_to_num(torch.softmax(st, dim=1), nan=0.0)      # eq. (6)
            Hv = (beta.unsqueeze(-1) * A).sum(1)
        else:
            Hv = torch.zeros_like(e)
        return self.out(torch.cat([Hv, e], -1)).squeeze(-1)                 # sigmoid in the loss


# ---------------------------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------------------------
def select_threshold(y, p):
    cand = np.unique(np.round(p, 4))
    return 0.5 if len(cand) < 2 else max((f1_score(y, (p >= t).astype(int), zero_division=0), t) for t in cand)[1]


def metrics(y, p, thr):
    yp = (p >= thr).astype(int)
    return {'roc_auc': roc_auc_score(y, p), 'pr_auc': average_precision_score(y, p),
            'f1': f1_score(y, yp, zero_division=0), 'precision': precision_score(y, yp, zero_division=0),
            'recall': recall_score(y, yp, zero_division=0)}


def run_scope(g, nbhd, scope, q, variant):
    frame, feats = bdata.load_view(scope, q, 'node_graph', cohort='original')
    full, _ = bdata.load_view('full', q, 'node_graph', cohort='original')
    pre = bdata.Preprocessor(add_mask=False).fit(frame.loc[frame.split == 'train', feats])
    tok_feat = np.zeros((g.T, len(feats)), dtype=np.float32)
    tok_feat[g.tokens.get_indexer(full.token_mint)] = pre.transform_(full[feats])
    ids = g.tokens.get_indexer(frame.token_mint)
    assert (ids >= 0).all()
    y = frame.rug_pull_label.to_numpy(int)
    split = frame.split.to_numpy()

    # Wallet-feature standardisation fitted on neighbourhoods of training targets only.
    tr_w = np.vstack([nbhd[i][1] for i, sp in zip(ids, split) if sp == 'train' and len(nbhd[i][1])])
    mu, sd = tr_w.mean(0), np.where(tr_w.std(0) < 1e-9, 1, tr_w.std(0))

    n = len(ids)
    X_self = torch.tensor(tok_feat[ids])
    tok_seq = np.zeros((n, TOPK, len(feats)), np.float32)
    tok_mask = np.zeros((n, TOPK), bool)
    wal_seq = np.zeros((n, TOPK, 6), np.float32)
    wal_mask = np.zeros((n, TOPK), bool)
    for r, i in enumerate(ids):
        tk, wf = nbhd[i]
        tok_seq[r, :len(tk)] = tok_feat[tk]
        tok_mask[r, :len(tk)] = True
        wal_seq[r, :len(wf)] = (wf - mu) / sd
        wal_mask[r, :len(wf)] = True
    tensors = [X_self, torch.tensor(tok_seq), torch.tensor(tok_mask), torch.tensor(wal_seq), torch.tensor(wal_mask)]
    tensors = [t.to(DEVICE) for t in tensors]
    yt = torch.tensor(y, dtype=torch.float32, device=DEVICE)
    parts = {p: np.flatnonzero(split == p) for p in ['train', 'val', 'test']}
    kw = dict(use_tok=variant != 'no_token_nbrs', use_wal=variant != 'no_wallet_nbrs', use_lstm=variant != 'no_lstm')

    def predict(model, idx):
        model.eval()
        with torch.no_grad():
            Xs, ts_, tm, ws, wm = [t[idx] for t in tensors]
            return torch.sigmoid(model(Xs, {'tok': (ts_, tm), 'wal': (ws, wm)})).cpu().numpy()

    runs, preds = [], []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = HNN4RP(len(feats), 6, **kw).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
        tr = torch.tensor(parts['train'], device=DEVICE)
        best, best_state, best_epoch = -1, None, 0
        for epoch in range(EPOCHS):
            model.train()
            perm = tr[torch.randperm(len(tr), device=DEVICE)]
            for b in range(0, len(perm), BATCH):
                idx = perm[b:b + BATCH]
                Xs, ts_, tm, ws, wm = [t[idx] for t in tensors]
                loss = F.binary_cross_entropy_with_logits(model(Xs, {'tok': (ts_, tm), 'wal': (ws, wm)}), yt[idx])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
                opt.step()
            ap = average_precision_score(y[parts['val']], predict(model, torch.tensor(parts['val'], device=DEVICE)))
            if ap > best:
                best, best_state, best_epoch = ap, copy.deepcopy(model.state_dict()), epoch + 1
        model.load_state_dict(best_state)
        scores = {p: predict(model, torch.tensor(ix, device=DEVICE)) for p, ix in parts.items()}
        thr = select_threshold(y[parts['val']], scores['val'])
        for p, ix in parts.items():
            runs.append({'model': f'hnn4rp{"" if variant == "full" else "-" + variant}', 'seed': seed, 'partition': p,
                         **metrics(y[ix], scores[p], thr), 'threshold': thr, 'best_epoch': best_epoch,
                         'scope': scope, 'experiment': q, 'view': 'node_graph', 'variant': 'no_mask', 'cohort': 'original'})
            preds.append(pd.DataFrame({'token_mint': frame.token_mint.to_numpy()[ix], 'seed': seed, 'partition': p,
                                       'y_true': y[ix], 'y_score': scores[p], 'y_pred': (scores[p] >= thr).astype(int),
                                       'threshold': thr}))
    name = f'hnn4rp{"" if variant == "full" else "-" + variant}'
    tag = f'{name}_{scope}_{q}_node_graph_no_mask_original'
    pd.DataFrame(runs).to_csv(OUT / f'runs_{tag}.csv', index=False)
    pd.concat(preds).assign(model=name, scope=scope, experiment=q).to_csv(OUT / f'final_predictions_{tag}.csv', index=False)
    r = pd.DataFrame(runs).groupby('partition')[['roc_auc', 'pr_auc', 'f1', 'precision', 'recall']].mean()
    print(f'[{q} {scope} {variant}]\n{r.round(3)}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--experiments', nargs='+', default=['q1', 'q2', 'q3'])
    p.add_argument('--scopes', nargs='+', default=['panel', 'full'])
    p.add_argument('--variants', nargs='+', default=['full', 'no_token_nbrs', 'no_wallet_nbrs', 'no_lstm'])
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for q in a.experiments:
        t0 = time.time()
        g = TxGraph(q)
        full, _ = bdata.load_view('full', q, 'node_graph', cohort='original')
        featured = np.zeros(g.T, bool)
        featured[g.tokens.get_indexer(full.token_mint)] = True
        targets = np.flatnonzero(featured)
        nbhd = sample_neighbourhoods(g, featured, targets)
        stats = pd.DataFrame([{'token': g.tokens[i], 'token_nbrs': len(v[0]), 'wallet_nbrs': len(v[1])} for i, v in nbhd.items()])
        stats.to_csv(OUT / f'neighbourhood_sizes_{q}.csv', index=False)
        print(f'[{q}] sampled {len(nbhd)} neighbourhoods in {time.time() - t0:.0f}s; '
              f'mean token nbrs {stats.token_nbrs.mean():.1f}, wallet nbrs {stats.wallet_nbrs.mean():.1f}', flush=True)
        for scope in a.scopes:
            for variant in a.variants:
                run_scope(g, nbhd, scope, q, variant)


if __name__ == '__main__':
    main()
