# trainer/gnn/

This package holds the graph neural networks. Each token is one graph-level classification sample.
Run it through `trainer/run_gnn.py`.

## Graph construction (`data.py`)

* **Nodes.** The transfer endpoints recovered inside the token's observation window `[t0, t_q)`.
* **Edges.** One directed edge per transfer. Parallel edges are kept.
* **Node features.** Structural and temporal endpoint features, for example degree and the first
  and mean transfer time as a fraction of the window. Their transforms are fitted on the nodes of
  training graphs only.
* **Late fusion.** The pooled graph embedding is concatenated with the 14 tabular predictors of the
  `node_graph` baseline view (5 token-level and 9 graph-summary features) before the classification
  head. GNN and baseline therefore share the same predictors, and the GNN additionally learns from
  topology.
* **Graph-only.** `run_gnn.py --graph-only` turns the late fusion off, so only the graph is used.

## Models (`models/`)

| Model | File | Used in the paper |
|-------|------|-------------------|
| GCN | `gcn.py` | official GNN |
| GraphSAGE | `graphsage.py` | official GNN |
| GAT | `gat.py` | official GNN |
| DirSAGE | `dirsage.py` | variant: separate in- and out-neighbor aggregation (Dir-GNN style) |
| DirGCN | `dirgcn.py` | variant: Dir-GCN with a tuned in/out mixing weight α ∈ {0.25, 0.5, 0.75} |

Each model module exposes `NAME`, `SPACE(trial)` (its Optuna search space), `build(params)` and
`train(...)`. Models are registered in `models/__init__.py`. `common.py` holds the shared pooling
and fusion-head code.

## Other modules

| File | Purpose |
|------|---------|
| `config.py` | Paths, data axes, graph schema, preprocessing, and the Optuna and evaluation settings. Key settings: `SELECTION_METRIC`, `EVALUATE_TEST`, and `RUGPULL_SPLIT` from the environment. |
| `evaluation.py` | Optuna search on validation, then retraining over 5 seeds and scoring with a frozen threshold |
| `metrics.py` | AP, ROC-AUC, Brier and F1, and selection of the threshold on validation |
| `reporting.py`, `plotting.py` | Writing runs, sweeps and predictions; train vs validation plots |
| `torch_utils.py` | Device selection, seeding and training loops |

See `trainer/README.md` for the selection-metric note on reproducing the official runs.
