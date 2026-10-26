#!/usr/bin/env bash
# Direction-aware GNNs (dirsage, dirgcn), one window per call, with retry; resumes finished cells.
# Usage: bash run_gnn_improved.sh q1|q2|q3
set -u
cd "$(dirname "$(readlink -f "$0")")"
Q=${1:?q1|q2|q3}
OUT=results/gnn_improved
mkdir -p "$OUT"
for attempt in 1 2 3 4 5; do
    echo "RUNNING attempt $attempt since $(date -Is)" > "$OUT/STATUS_$Q"
    if ../.venv/bin/python -u run_gnn_improved.py --experiments "$Q" >> "$OUT/run_$Q.log" 2>&1; then
        echo "DONE $(date -Is)" > "$OUT/STATUS_$Q"; exit 0
    fi
    echo "=== attempt $attempt failed $(date -Is) ===" >> "$OUT/run_$Q.log"; sleep 30
done
echo "FAILED $(date -Is)" > "$OUT/STATUS_$Q"; exit 1
