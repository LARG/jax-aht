#!/usr/bin/env bash
set -euo pipefail

TASKS=(
    # "lbf/lbf_7x7_nolevels" # excluded because mixed sweep strategies used
    "lbf/lbf_12x12"
    "overcooked-v1/cramped_room"
    # "overcooked-v1/coord_ring" # excluded because mixed sweep strategies used
)

for task in "${TASKS[@]}"; do
    for algo_type in ego unified; do
        echo "=== task=${task}  algo-type=${algo_type} ==="
        PYTHONPATH=. python scripts/paper_vis/plot_sweep_distribution.py \
            --algo-type "${algo_type}" \
            --task "${task}"
    done
done
