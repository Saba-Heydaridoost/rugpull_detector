# dataset/

This folder collects the raw data: rule-based labels, token-level features, and per-token transfer
edges. Run the scripts from this folder, in this order.

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 1 | `build_rule_based_labels.py` | `2024.csv` (SolRPDS 2024, not included) | `rule_based_labels.csv` |
| 2 | `collect_nodes.py` | `rule_based_labels.csv` | `graph_nodes.csv` |
| 3 | `collect_edges.py` | `rule_based_labels.csv` | `graph_edges.csv` |
| 4 | `normalize_edge_amounts.py` | `graph_edges.csv` | per-edge token amounts, `edge_amount_normalization.csv` |

Each collector keeps a progress file (`processed_*.txt`), so an interrupted run resumes where it
stopped. Progress files are not committed. Setting `TEST_MODE = 1` in a script runs it on a small
sample and writes to `*_test.csv`.

## Credentials

* Steps 1 and 2 read the Birdeye API key from the `BIRDEYE_API_KEY` environment variable.
* Steps 3 and 4 use the public Solana RPC endpoint by default. `normalize_edge_amounts.py` reads
  `SOLANA_RPC_URL` if it is set.
* No key is stored in the code.

## Labels (`build_rule_based_labels.py`)

For every token and window `q ∈ {q1, q2, q3}` (cutoff `t_q` = 750 s, 2,127 s or 5,585 s after
creation `t0`), the script inspects the outcome horizon `(t_q, t0 + 24 h]`. A token is labeled a
rug pull (`rug_pull_label = 1`) when three conditions all hold:

1. **Liquidity drop.** Liquidity falls more than 80 % below its running peak, at the first such
   time `τ > t_q`.
2. **Price drop.** The price falls more than 90 % after `τ`.
3. **Persistence.** The loss persists: liquidity stays at or below 20 % of the pre-event peak for
   at least 80 % of the 30-minute confirmation period `[τ, τ + 30 min]`.

Only `label_status = VALID` rows carry a binary label. The other rows record why no label could be
assigned, for example:

* `IN_WINDOW_EVENT`: the drop happened before the cutoff;
* `UNKNOWN_MISSING_LIQUIDITY_HISTORY`, and similar `UNKNOWN_*` statuses for missing or insufficient
  data.

The paper's appendix lists every status with its counts.

## Token-level features (`collect_nodes.py`)

The script writes one row per valid (token, window) pair. It uses only data observed in
`[t0, t_q)`, plus the liquidity snapshot at the cutoff. The features include:

* buy/sell volume changes;
* unique-wallet counts and their changes;
* mint and freeze authorities;
* liquidity at the observation end.

## Transfer edges (`collect_edges.py`, `normalize_edge_amounts.py`)

These scripts reconstruct the token transfers inside the same observation window: sender,
receiver, time and amount. `normalize_edge_amounts.py` recovers an exact per-edge token amount from
the stored transaction signatures, using the Solana RPC only. The token graphs used by the GNNs are
built from these edges in `data_analysis/EDA.ipynb`.

## Files in the repository

| File | Released | Notes |
|------|----------|-------|
| `rule_based_labels.csv` | yes | token IDs, label status, labels and the recorded drop ratios |
| `graph_nodes.csv` | yes | token-level features per window |
| `graph_edges.csv`, `edge_amount_normalization.csv` | no | edge-level data, not released |
| `2024.csv` | no | SolRPDS, obtain from the original source |
