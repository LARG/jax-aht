#!/usr/bin/env bash
#SBATCH -J wandb-sweep
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu-h100
#SBATCH -t 24:00:00
#SBATCH -A YOUR_TACC_ALLOCATION
#SBATCH -o sweep-%j.out
#SBATCH --mem=64G
#
# Slurm wrapper for a wandb sweep agent on TACC LS6 (H100 nodes).
#
# Usage (on LS6 login node, AFTER initializing the sweep):
#   SWEEP_ID=aht-project/aht-parameter-sweep/abc123 \
#     COUNT=15 \
#     sbatch scripts/launch_ls6_sweep.sh
#
# Each agent process consumes COUNT sweep configs from wandb's controller
# before exiting. Launch multiple sbatch jobs with the same SWEEP_ID to
# parallelize across nodes.
#
# Local dry-run (no slurm needed):
#   SWEEP_ID=...  COUNT=1  bash scripts/launch_ls6_sweep.sh
# Slurm directives are no-ops when run via plain bash.

set -euo pipefail

if [ -z "${SWEEP_ID:-}" ]; then
    echo "ERROR: SWEEP_ID env var required."
    echo "  Initialize a sweep first: PYTHONPATH=\$WORK/jax-aht wandb sweep <yaml>"
    echo "  Then: SWEEP_ID=<output_id> sbatch scripts/launch_ls6_sweep.sh"
    exit 1
fi

COUNT="${COUNT:-15}"
REPO_ROOT="${REPO_ROOT:-$WORK/jax-aht}"
VENV_PATH="${VENV_PATH:-$WORK/jax-aht/.venv/bin/activate}"

echo "=== launch_ls6_sweep ==="
echo "  REPO_ROOT:     $REPO_ROOT"
echo "  VENV_PATH:     $VENV_PATH"
echo "  SWEEP_ID:      $SWEEP_ID"
echo "  COUNT:         $COUNT"
echo "  SLURM_JOB_ID:  ${SLURM_JOB_ID:-(none, running outside slurm)}"
echo

# Module load (TACC only; harmless when run locally if module unavailable)
if command -v module >/dev/null 2>&1; then
    module load cuda/12 || echo "WARN: cuda/12 module not available"
fi

# Activate venv
if [ -f "$VENV_PATH" ]; then
    # shellcheck disable=SC1090
    source "$VENV_PATH"
else
    echo "WARN: venv not found at $VENV_PATH; using current Python"
fi

if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "WARN: WANDB_API_KEY not set. Agents will fail to authenticate."
fi

export PYTHONPATH="$REPO_ROOT"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"

cd "$REPO_ROOT"
exec wandb agent "$SWEEP_ID" --count "$COUNT"
