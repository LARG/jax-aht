#!/bin/bash

# Slurm submit script for TACC Vista Grace Hopper (GH) nodes.
# Grace Hopper node specs: 72 CPU cores, 116 GB DDR5, 1x H200 GPU (96 GB HBM3).
#
# Usage:
#   sbatch scripts/manage_configs/slurm_submit.sh
#
# To run on the dev queue (2-hr walltime limit, faster queue time):
#   sbatch --partition=gh-dev scripts/manage_configs/slurm_submit.sh
#
# Override default config variables at submission time:
#   ALGO=ippo TASK=lbf LABEL=my_run NUM_SEEDS=5 sbatch scripts/manage_configs/slurm_submit.sh
#
# All overridable variables and their defaults:
#   ALGO          Algorithm to run           (default: fcp)
#   TASK          Task/environment           (default: lbf)
#   LABEL         Run label for logging      (default: fcp_lbf_test)
#   NUM_SEEDS     Number of random seeds     (default: 10)
#   LOG_TRAIN_OUT Log training output        (default: false)
#   LOG_LOCAL_OUT Save output locally        (default: false)

#SBATCH -J jax-aht              # Job name (shown in squeue)
#SBATCH -p gh                   # Partition/queue (gh = Grace Hopper production)
#SBATCH -N 1                    # Number of nodes
#SBATCH --ntasks-per-node=1     # Number of MPI tasks per node (1 for single-process JAX)
#SBATCH -t 2:00:00             # Walltime limit (HH:MM:SS)
#SBATCH -o results/slurm_logs/%j_%x.out  # Stdout log (%j=job ID, %x=job name)
#SBATCH -e results/slurm_logs/%j_%x.err  # Stderr log
#SBATCH -A ASC25021             # Allocation account to charge

set -euo pipefail   # Exit on error (-e), undefined variable (-u), or pipe failure (-o pipefail)

# ── Config ─────────────────────────────────────────────────────────────────────
ALGO="${ALGO:-fcp}"                      # Algorithm (default: fcp)
TASK="${TASK:-lbf}"                      # Task/environment (default: lbf)
LABEL="${LABEL:-fcp_lbf_test}"           # Run label for logging
NUM_SEEDS="${NUM_SEEDS:-5}"             # Number of random seeds
LOG_TRAIN_OUT="${LOG_TRAIN_OUT:-false}"  # Whether to log training output to wandb
LOG_LOCAL_OUT="${LOG_LOCAL_OUT:-false}"  # Whether to save output locally

# ── Paths ──────────────────────────────────────────────────────────────────────
# ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # Repo root, resolved relative to this script
ROOT_DIR="$SCRATCH/jax-aht"
cd "$ROOT_DIR"
 mkdir -p results/slurm_logs  # Ensure log directory exists before SLURM tries to write to it

# ── Environment ────────────────────────────────────────────────────────────────
export LD_LIBRARY_PATH=""  # Clear LD_LIBRARY_PATH to avoid system CUDA/cuDNN libraries conflicting with conda-installed ones
# export LD_LIBRARY_PATH=/scratch/08090/clw4542/conda_envs/bench311/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH  # Add conda cuDNN to library path

source $(conda info --base)/etc/profile.d/conda.sh  # Initialize conda for non-interactive shell
conda activate bench311                              # Activate the project environment

# ── Info ───────────────────────────────────────────────────────────────────────
echo "=== Job info ==="
echo "Job ID:   $SLURM_JOB_ID"    # Set by SLURM at job start
echo "Node:     $SLURM_NODELIST"  # Hostname(s) of allocated node(s)
echo "Root dir: $ROOT_DIR"
echo "Algo:     $ALGO/$TASK"
echo "Label:    $LABEL"
echo "Seeds:    $NUM_SEEDS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader  # Confirm GPU assignment
echo "================"

# ── Run ────────────────────────────────────────────────────────────────────────
XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=. python teammate_generation/run.py \
    algorithm="${ALGO}/${TASK}" \
    task="${TASK}" \
    label="${LABEL}" \
    algorithm.NUM_SEEDS="${NUM_SEEDS}" \
    logger.log_train_out="${LOG_TRAIN_OUT}" \
    local_logger.save_train_out="${LOG_LOCAL_OUT}" \
    local_logger.save_eval_out="${LOG_LOCAL_OUT}"
