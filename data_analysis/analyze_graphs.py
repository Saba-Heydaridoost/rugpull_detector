from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
Q_VALUES = ("q1", "q2", "q3")
TRAIN_SPLIT = "train"
EDGES_FILE = "edges_with_split.parquet"
FEATURES_FILE = "model_graph_features_with_split.parquet"
TOKEN_CANDIDATES = ("token_mint", "mint", "token_address")
SOURCE_CANDIDATES = ("source_address", "source", "src")
TARGET_CANDIDATES = ("target_address", "target", "dst")
SPLIT_CANDIDATES = ("split",)
LABEL_CANDIDATES = ("rug_pull_label", "rug_label", "label", "y")


def first_existing(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    cols = set(columns)
    for name in candidates:
        if name in cols:
            return name
    return None


def require_column(df: pd.DataFrame, candidates: Iterable[str], what: str) -> str:
    col = first_existing(df.columns, candidates)
    if col is None:
        raise KeyError(f"Could not find {what}. Tried {list(candidates)}. Available: {list(df.columns)}")
    return col


def clean_address_series(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip()
    return s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})


def safe_max(values) -> float:
    vals = list(values)
    return float(max(vals)) if vals else 0.0


def top_share(degrees: Dict[str, float], k: int) -> float:
    vals = np.asarray(list(degrees.values()), dtype=float)
    total = vals.sum()
    if vals.size == 0 or total <= 0:
        return 0.0
    vals = np.sort(vals)[::-1]
    return float(vals[:k].sum() / total)


def weighted_simple_digraph(edges: pd.DataFrame, source_col: str, target_col: str) -> nx.DiGraph:
    grouped = edges.groupby([source_col, target_col], dropna=False).size().reset_index(name="weight")
    g = nx.DiGraph()
    for row in grouped.itertuples(index=False):
        src = getattr(row, source_col)
        dst = getattr(row, target_col)
        g.add_edge(src, dst, weight=int(row.weight))
    return g


def compute_metrics(token_mint: str, q: str, rug_pull_label: int, token_edges: pd.DataFrame,
                    source_col: str, target_col: str) -> Dict[str, float]:
    e = token_edges[[source_col, target_col]].copy()
    e[source_col] = clean_address_series(e[source_col])
    e[target_col] = clean_address_series(e[target_col])
    e = e.dropna(subset=[source_col, target_col])
    e = e[e[source_col] != e[target_col]]

    if e.empty:
        return {
            "q": q, "token_mint": token_mint, "rug_pull_label": int(rug_pull_label),
            "n_nodes": 0, "n_edges": 0, "n_unique_directed_edges": 0,
            "n_parallel_edges": 0, "density": 0.0, "avg_degree": 0.0,
            "n_senders": 0, "n_receivers": 0, "largest_wcc_ratio": 0.0,
            "reciprocity": 0.0, "top1_degree_share": 0.0, "top5_degree_share": 0.0,
            "max_in_degree_centrality": 0.0, "max_out_degree_centrality": 0.0,
            "max_betweenness_centrality": 0.0, "max_pagerank": 0.0,
        }

    n_transfers = int(len(e))
    g = weighted_simple_digraph(e, source_col, target_col)
    n_nodes = g.number_of_nodes()
    n_unique_edges = g.number_of_edges()
    n_parallel = n_transfers - n_unique_edges
    density = nx.density(g) if n_nodes > 1 else 0.0
    avg_degree = (2.0 * n_unique_edges / n_nodes) if n_nodes else 0.0
    n_senders = int(e[source_col].nunique())
    n_receivers = int(e[target_col].nunique())

    wcc_sizes = [len(c) for c in nx.weakly_connected_components(g)] if n_nodes else []
    largest_wcc_ratio = (max(wcc_sizes) / n_nodes) if wcc_sizes else 0.0

    reciprocity_raw = nx.reciprocity(g)
    reciprocity = 0.0 if reciprocity_raw is None or not np.isfinite(reciprocity_raw) else float(reciprocity_raw)

    degree = dict(g.degree())
    top1 = top_share(degree, 1)
    top5 = top_share(degree, 5)

    if n_nodes > 1:
        in_cent = nx.in_degree_centrality(g)
        out_cent = nx.out_degree_centrality(g)
    else:
        in_cent = {n: 0.0 for n in g.nodes}
        out_cent = {n: 0.0 for n in g.nodes}

    if n_nodes > 1 and n_unique_edges > 0:
        betweenness = nx.betweenness_centrality(g, normalized=True)
    else:
        betweenness = {n: 0.0 for n in g.nodes}

    if n_nodes > 0 and n_unique_edges > 0:
        try:
            pagerank = nx.pagerank(g, weight="weight")
        except Exception:
            pagerank = {n: 1.0 / n_nodes for n in g.nodes}
    elif n_nodes > 0:
        pagerank = {n: 1.0 / n_nodes for n in g.nodes}
    else:
        pagerank = {}

    return {
        "q": q,
        "token_mint": token_mint,
        "rug_pull_label": int(rug_pull_label),
        "n_nodes": int(n_nodes),
        "n_edges": int(n_transfers),
        "n_unique_directed_edges": int(n_unique_edges),
        "n_parallel_edges": int(n_parallel),
        "density": float(density),
        "avg_degree": float(avg_degree),
        "n_senders": int(n_senders),
        "n_receivers": int(n_receivers),
        "largest_wcc_ratio": float(largest_wcc_ratio),
        "reciprocity": float(reciprocity),
        "top1_degree_share": float(top1),
        "top5_degree_share": float(top5),
        "max_in_degree_centrality": safe_max(in_cent.values()),
        "max_out_degree_centrality": safe_max(out_cent.values()),
        "max_betweenness_centrality": safe_max(betweenness.values()),
        "max_pagerank": safe_max(pagerank.values()),
    }


def summarize_numeric(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "rug_pull_label"]
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        base["n_graphs"] = int(len(group))
        for col in numeric_cols:
            x = pd.to_numeric(group[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            base[f"{col}_mean"] = float(x.mean()) if len(x) else np.nan
            base[f"{col}_median"] = float(x.median()) if len(x) else np.nan
            base[f"{col}_std"] = float(x.std(ddof=1)) if len(x) > 1 else 0.0
            base[f"{col}_min"] = float(x.min()) if len(x) else np.nan
            base[f"{col}_max"] = float(x.max()) if len(x) else np.nan
        rows.append(base)
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics = []

    for q in Q_VALUES:
        edges_path = DATA_ROOT / q / EDGES_FILE
        features_path = DATA_ROOT / q / FEATURES_FILE
        if not edges_path.exists():
            raise FileNotFoundError(edges_path)
        if not features_path.exists():
            raise FileNotFoundError(features_path)

        edges = pd.read_parquet(edges_path)
        features = pd.read_parquet(features_path)

        edge_split_col = require_column(edges, SPLIT_CANDIDATES, "edge split")
        feature_split_col = require_column(features, SPLIT_CANDIDATES, "feature split")

        # IMPORTANT: train split only. Validation and test are excluded entirely.
        edges = edges[edges[edge_split_col].astype(str).str.lower() == TRAIN_SPLIT].copy()
        features = features[features[feature_split_col].astype(str).str.lower() == TRAIN_SPLIT].copy()

        if edges.empty:
            raise RuntimeError(f"{q}: no training edges after split filter")
        if features.empty:
            raise RuntimeError(f"{q}: no training feature rows after split filter")

        assert set(edges[edge_split_col].astype(str).str.lower().unique()) == {TRAIN_SPLIT}
        assert set(features[feature_split_col].astype(str).str.lower().unique()) == {TRAIN_SPLIT}

        edge_token_col = require_column(edges, TOKEN_CANDIDATES, "edge token mint")
        feature_token_col = require_column(features, TOKEN_CANDIDATES, "feature token mint")
        source_col = require_column(edges, SOURCE_CANDIDATES, "source address")
        target_col = require_column(edges, TARGET_CANDIDATES, "target address")
        label_col = require_column(features, LABEL_CANDIDATES, "rug-pull label")

        labels = features[[feature_token_col, label_col]].dropna().drop_duplicates()
        conflicts = labels.groupby(feature_token_col)[label_col].nunique()
        conflicts = conflicts[conflicts > 1]
        if len(conflicts):
            raise RuntimeError(f"{q}: conflicting labels for {len(conflicts)} token(s)")

        labels = labels.drop_duplicates(subset=[feature_token_col], keep="first")
        label_map = dict(zip(labels[feature_token_col].astype(str), pd.to_numeric(labels[label_col]).astype(int)))

        edge_tokens = set(edges[edge_token_col].astype(str).unique())
        missing = edge_tokens - set(label_map)
        if missing:
            raise RuntimeError(f"{q}: {len(missing)} training token(s) in edges have no training label")

        print("=" * 60)
        print(f"{q.upper()} - TRAIN ONLY")
        print("=" * 60)
        print(f"Training edge rows: {len(edges):,}")
        print(f"Training graphs/tokens: {len(edge_tokens):,}")
        vc = pd.Series([label_map[t] for t in edge_tokens]).value_counts().sort_index()
        print("Label counts: " + ", ".join(f"class {k}={v}" for k, v in vc.items()))

        groups = list(edges.groupby(edge_token_col, sort=False))
        for idx, (token, token_edges) in enumerate(groups, start=1):
            token = str(token)
            all_metrics.append(
                compute_metrics(token, q, label_map[token], token_edges, source_col, target_col)
            )
            if idx % 100 == 0 or idx == len(groups):
                print(f"  processed {idx:,}/{len(groups):,} training graphs")

    metrics_df = pd.DataFrame(all_metrics)
    if metrics_df.empty:
        raise RuntimeError("No graph metrics generated")

    numeric = metrics_df.select_dtypes(include=[np.number])
    if np.isinf(numeric.to_numpy()).any():
        raise RuntimeError("Generated metrics contain +/-inf")

    metrics_df.to_csv(OUTPUT_DIR / "graph_metrics_per_token.csv", index=False)
    summarize_numeric(metrics_df, ["q"]).to_csv(OUTPUT_DIR / "summary_by_q.csv", index=False)
    summarize_numeric(metrics_df, ["q", "rug_pull_label"]).to_csv(
        OUTPUT_DIR / "summary_by_q_and_label.csv", index=False
    )

    print("\n" + "=" * 60)
    print("TRAIN-ONLY GRAPH ANALYSIS COMPLETE")
    print("=" * 60)
    print(metrics_df.groupby(["q", "rug_pull_label"]).size())
    print(f"\nSaved outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
