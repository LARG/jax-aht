#!/bin/bash
# Wrapper that validates args before submitting the sweep job.
# Usage: bash scripts/slurm/submit_sweep.sh <sweep_id> [extra sbatch args...]
# Example: bash scripts/slurm/submit_sweep.sh dqsezvy1 --partition gpu

if [ -z "$1" ]; then
    echo "Usage: $0 <sweep_id> [extra sbatch args...]"
    echo "Example: $0 dqsezvy1"
    exit 1
fi

SWEEP_ID="$1"
shift  # remaining args forwarded to sbatch

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sbatch \
    --job-name="wandb_sweep_$SWEEP_ID" \
    --output="results/slurm_logs/wandb_sweep_${SWEEP_ID}_%j.out" \
    --error="results/slurm_logs/wandb_sweep_${SWEEP_ID}_%j.err" \
    "$@" "$SCRIPT_DIR/slurm_hparam_sweep.sh" "$SWEEP_ID"
