#!/usr/bin/env bash
# Runs the global-graph GNN ablation (q1) with automatic retry; resumes via --skip-existing semantics.
# Usage: bash run_gnn_global_ablation.sh per_token|global
set -u
cd "$(dirname "$(readlink -f "$0")")"
G=${1:?per_token|global}
OUT=results/gnn_global_ablation/$G
mkdir -p "$OUT"
for attempt in 1 2 3 4 5; do
    echo "RUNNING attempt $attempt since $(date -Is)" > "$OUT/STATUS"
    if ../.venv/bin/python -u run_gnn_global_ablation.py --graph "$G" >> "$OUT/run.log" 2>&1; then
        echo "DONE $(date -Is)" > "$OUT/STATUS"; exit 0
    fi
    echo "=== attempt $attempt failed $(date -Is) ===" >> "$OUT/run.log"; sleep 30
done
echo "FAILED $(date -Is)" > "$OUT/STATUS"; exit 1
