from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import networkx as nx


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"

Q_WINDOWS = ("q1", "q2", "q3")

EDGES_FILE = "edges_with_split.parquet"
META_FILE = "model_graph_features_with_split.parquet"

TOKEN_COL = "token_mint"

SOURCE_CANDIDATES = ("source_address", "source", "src")
TARGET_CANDIDATES = ("target_address", "target", "dst")
LABEL_CANDIDATES = ("rug_pull_label", "label")
SPLIT_CANDIDATES = ("split",)


def _find_column(df: pd.DataFrame, candidates: Iterable[str], name: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(
        f"Could not find {name}. Tried: {list(candidates)}. "
        f"Available columns: {list(df.columns)}"
    )


def _safe_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    return float(np.mean(arr)) if arr.size else np.nan


def _safe_median(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    return float(np.median(arr)) if arr.size else np.nan


def _safe_max(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    return float(np.max(arr)) if arr.size else np.nan


def _top_k_degree_share(G: nx.MultiDiGraph, k: int) -> float:
    degrees = np.asarray([d for _, d in G.degree()], dtype=float)
    total = degrees.sum()
    if degrees.size == 0 or total <= 0:
        return np.nan
    k = min(k, degrees.size)
    return float(np.sort(degrees)[-k:].sum() / total)


def _collapse_multidigraph(G_multi: nx.MultiDiGraph) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(G_multi.nodes())

    for u, v in G_multi.edges():
        if G.has_edge(u, v):
            G[u][v]["weight"] += 1.0
        else:
            G.add_edge(u, v, weight=1.0)

    return G


def _build_graph(
    token_edges: pd.DataFrame,
    source_col: str,
    target_col: str,
) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()

    for src, dst in token_edges[[source_col, target_col]].itertuples(
        index=False, name=None
    ):
        if pd.isna(src) or pd.isna(dst):
            continue

        src = str(src)
        dst = str(dst)

        if src == dst:
            continue

        G.add_edge(src, dst)

    return G


def _compute_graph_metrics(
    token_mint: str,
    q: str,
    G_multi: nx.MultiDiGraph,
) -> Dict[str, float]:
    n_nodes = G_multi.number_of_nodes()
    n_edges = G_multi.number_of_edges()

    if n_nodes == 0 or n_edges == 0:
        raise ValueError(
            f"Empty graph encountered for token={token_mint}, window={q}"
        )

    G = _collapse_multidigraph(G_multi)
    n_unique_directed_edges = G.number_of_edges()

    in_degrees = [d for _, d in G_multi.in_degree()]
    out_degrees = [d for _, d in G_multi.out_degree()]

    density = nx.density(G)

    weak_components = list(nx.weakly_connected_components(G))
    n_wcc = len(weak_components)
    largest_wcc_size = max((len(c) for c in weak_components), default=0)
    largest_wcc_ratio = largest_wcc_size / n_nodes if n_nodes else np.nan

    reciprocity = nx.reciprocity(G)
    if reciprocity is None:
        reciprocity = 0.0

    in_degree_c = nx.in_degree_centrality(G)
    out_degree_c = nx.out_degree_centrality(G)

    pagerank = nx.pagerank(G, alpha=0.85, weight="weight")

    betweenness = nx.betweenness_centrality(
        G,
        normalized=True,
        weight=None,
    )

    return {
        "q": q,
        TOKEN_COL: token_mint,
        "n_nodes": int(n_nodes),
        "n_edges": int(n_edges),
        "n_unique_directed_edges": int(n_unique_directed_edges),
        "parallel_edge_count": int(n_edges - n_unique_directed_edges),
        "density": float(density),
        "mean_in_degree": _safe_mean(in_degrees),
        "median_in_degree": _safe_median(in_degrees),
        "max_in_degree": _safe_max(in_degrees),
        "mean_out_degree": _safe_mean(out_degrees),
        "median_out_degree": _safe_median(out_degrees),
        "max_out_degree": _safe_max(out_degrees),
        "top1_degree_share": _top_k_degree_share(G_multi, 1),
        "top5_degree_share": _top_k_degree_share(G_multi, 5),
        "n_weakly_connected_components": int(n_wcc),
        "largest_wcc_size": int(largest_wcc_size),
        "largest_wcc_ratio": float(largest_wcc_ratio),
        "reciprocity": float(reciprocity),
        "mean_in_degree_centrality": _safe_mean(in_degree_c.values()),
        "max_in_degree_centrality": _safe_max(in_degree_c.values()),
        "mean_out_degree_centrality": _safe_mean(out_degree_c.values()),
        "max_out_degree_centrality": _safe_max(out_degree_c.values()),
        "mean_betweenness_centrality": _safe_mean(betweenness.values()),
        "max_betweenness_centrality": _safe_max(betweenness.values()),
        "mean_pagerank": _safe_mean(pagerank.values()),
        "max_pagerank": _safe_max(pagerank.values()),
    }


def _load_window(q: str) -> Tuple[pd.DataFrame, pd.DataFrame, str, str, str, str]:
    q_dir = DATA_DIR / q
    edge_path = q_dir / EDGES_FILE
    meta_path = q_dir / META_FILE

    if not edge_path.exists():
        raise FileNotFoundError(f"Missing: {edge_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing: {meta_path}")

    edges = pd.read_parquet(edge_path)
    meta = pd.read_parquet(meta_path)

    if TOKEN_COL not in edges.columns:
        raise KeyError(f"{edge_path} does not contain '{TOKEN_COL}'.")
    if TOKEN_COL not in meta.columns:
        raise KeyError(f"{meta_path} does not contain '{TOKEN_COL}'.")

    source_col = _find_column(edges, SOURCE_CANDIDATES, "source address column")
    target_col = _find_column(edges, TARGET_CANDIDATES, "target address column")
    label_col = _find_column(meta, LABEL_CANDIDATES, "label column")
    split_col = _find_column(meta, SPLIT_CANDIDATES, "split column")

    if meta[TOKEN_COL].duplicated().any():
        duplicates = meta.loc[meta[TOKEN_COL].duplicated(), TOKEN_COL].head(10).tolist()
        raise ValueError(f"Duplicate token_mint rows in metadata for {q}: {duplicates}")

    return edges, meta, source_col, target_col, label_col, split_col


def analyze_window(q: str) -> pd.DataFrame:
    print(f"\n[{q}] Loading data...")

    edges, meta, source_col, target_col, label_col, split_col = _load_window(q)

    print(
        f"[{q}] edges={len(edges):,}, "
        f"edge tokens={edges[TOKEN_COL].nunique():,}, "
        f"metadata tokens={meta[TOKEN_COL].nunique():,}"
    )

    edge_tokens = set(edges[TOKEN_COL].astype(str))
    meta_tokens = set(meta[TOKEN_COL].astype(str))

    missing_meta = edge_tokens - meta_tokens
    if missing_meta:
        example = sorted(missing_meta)[:10]
        raise ValueError(
            f"{q}: {len(missing_meta)} edge tokens are missing from metadata. "
            f"Examples: {example}"
        )

    metadata = meta[[TOKEN_COL, label_col, split_col]].copy()
    metadata[TOKEN_COL] = metadata[TOKEN_COL].astype(str)
    metadata = metadata.rename(
        columns={
            label_col: "rug_pull_label",
            split_col: "split",
        }
    )

    rows: List[Dict[str, float]] = []

    grouped = edges.groupby(TOKEN_COL, sort=False)
    n_graphs = grouped.ngroups

    for idx, (token_mint, token_edges) in enumerate(grouped, start=1):
        token_mint = str(token_mint)

        G_multi = _build_graph(
            token_edges=token_edges,
            source_col=source_col,
            target_col=target_col,
        )

        if G_multi.number_of_edges() == 0:
            continue

        rows.append(
            _compute_graph_metrics(
                token_mint=token_mint,
                q=q,
                G_multi=G_multi,
            )
        )

        if idx % 100 == 0 or idx == n_graphs:
            print(f"[{q}] processed {idx:,}/{n_graphs:,} graphs")

    metrics_df = pd.DataFrame(rows)

    metrics_df = metrics_df.merge(
        metadata,
        on=TOKEN_COL,
        how="left",
        validate="one_to_one",
    )

    if metrics_df["rug_pull_label"].isna().any():
        bad = metrics_df.loc[
            metrics_df["rug_pull_label"].isna(), TOKEN_COL
        ].head(10).tolist()
        raise ValueError(f"{q}: missing labels after merge. Examples: {bad}")

    return metrics_df


def _metric_columns(df: pd.DataFrame) -> List[str]:
    excluded = {"q", TOKEN_COL, "rug_pull_label", "split"}
    return [
        c
        for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]


def make_summary(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    metric_cols = _metric_columns(df)
    rows = []

    grouper = group_cols[0] if len(group_cols) == 1 else group_cols

    for group_key, group in df.groupby(grouper, dropna=False, sort=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        base = dict(zip(group_cols, group_key))
        base["n_graphs"] = len(group)

        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()

            row = dict(base)
            row.update(
                {
                    "metric": metric,
                    "mean": values.mean() if len(values) else np.nan,
                    "std": values.std(ddof=1) if len(values) > 1 else np.nan,
                    "median": values.median() if len(values) else np.nan,
                    "q25": values.quantile(0.25) if len(values) else np.nan,
                    "q75": values.quantile(0.75) if len(values) else np.nan,
                    "min": values.min() if len(values) else np.nan,
                    "max": values.max() if len(values) else np.nan,
                }
            )
            rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_windows = [analyze_window(q) for q in Q_WINDOWS]
    metrics = pd.concat(all_windows, ignore_index=True)

    per_token_path = OUTPUT_DIR / "graph_metrics_per_token.csv"
    metrics.to_csv(per_token_path, index=False)

    summary_q = make_summary(metrics, ["q"])
    summary_q_path = OUTPUT_DIR / "summary_by_q.csv"
    summary_q.to_csv(summary_q_path, index=False)

    summary_q_label = make_summary(metrics, ["q", "rug_pull_label"])
    summary_q_label_path = OUTPUT_DIR / "summary_by_q_and_label.csv"
    summary_q_label.to_csv(summary_q_label_path, index=False)

    print("\nDone.")
    print(f"Per-token metrics:      {per_token_path}")
    print(f"Summary by q:           {summary_q_path}")
    print(f"Summary by q and label: {summary_q_label_path}")


if __name__ == "__main__":
    main()