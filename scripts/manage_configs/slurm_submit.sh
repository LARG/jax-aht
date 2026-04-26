#!/bin/bash
# Slurm submit script for TACC Vista Grace Hopper (GH) nodes.
# Grace Hopper node specs: 72 CPU cores, 116 GB DDR5, 1x H200 GPU (96 GB HBM3).
#
# Usage:
#   sbatch scripts/manage_configs/slurm_submit.sh
#
# To test on the dev queue (2-hr limit, faster queue time):
#   PARTITION=gh-dev sbatch scripts/manage_configs/slurm_submit.sh
#
# Override defaults at submission time, e.g.:
#   ALGO=fcp TASK=lbf LABEL=my_run NUM_SEEDS=5 sbatch scripts/manage_configs/slurm_submit.sh

#SBATCH -J jax-aht
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH -o results/slurm_logs/%j_%x.out
#SBATCH -e results/slurm_logs/%j_%x.err
#SBATCH -A ASC25021     # <-- replace with your TACC allocation name

# Allow partition override at submission time via environment variable.
# Example: PARTITION=gh-dev sbatch ...
if [[ -n "${PARTITION:-}" ]]; then
    # Slurm doesn't support runtime partition changes, so this is informational only.
    # To switch partitions, edit the #SBATCH -p line above or pass --partition to sbatch.
    echo "Note: to change partition, use: sbatch --partition=${PARTITION} $0"
fi

set -euo pipefail

# ── Config (override via env vars at sbatch time) ──────────────────────────────
ALGO="${ALGO:-fcp}"
TASK="${TASK:-lbf}"
LABEL="${LABEL:-fcp_lbf_test}"
NUM_SEEDS="${NUM_SEEDS:-10}"
LOG_TRAIN_OUT="${LOG_TRAIN_OUT:-false}"
LOG_LOCAL_OUT="${LOG_LOCAL_OUT:-false}"

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

mkdir -p results/slurm_logs

# ── Environment setup ──────────────────────────────────────────────────────────
# module load cuda

# source /scratch/cluster/clw4542/miniconda3/etc/profile.d/conda.sh
conda activate bench311

echo "=== Job info ==="
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURM_NODELIST"
echo "Root dir: $ROOT_DIR"
echo "Algo:     $ALGO/$TASK"
echo "Label:    $LABEL"
echo "Seeds:    $NUM_SEEDS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "================"

# ── Run ────────────────────────────────────────────────────────────────────────
python teammate_generation/run.py \
    algorithm="${ALGO}/${TASK}" \
    task="${TASK}" \
    label="${LABEL}" \
    algorithm.NUM_SEEDS="${NUM_SEEDS}" \
    logger.log_train_out="${LOG_TRAIN_OUT}" \
    local_logger.save_train_out="${LOG_LOCAL_OUT}" \
    local_logger.save_eval_out="${LOG_LOCAL_OUT}"
