#!/bin/bash
#----------------------------------------------------
# Phase E Multi-Seed: CREPPO — H100 (Stampede3)
#
# Seeds: 721280, 721281, 721282, 721283
# (PI convention: base seed 72128, append 0, increment)
#
# H100 has 1 GPU per node, so all experiments run
# sequentially. Estimated runtime: ~30-40h total.
# Requesting 48h to be safe.
#----------------------------------------------------
#SBATCH -J creppo_multiseed_h100
#SBATCH -o logs/creppo_multiseed_%j.out
#SBATCH -e logs/creppo_multiseed_%j.err
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH -t 48:00:00
#SBATCH --mail-user=jeffreychen287@utexas.edu
#SBATCH --mail-type=end,fail

mkdir -p logs

module purge
module load python
module load cuda/12

cd /scratch/11079/jeffreychen287/jax-aht

export PYTHONPATH=$PYTHONPATH:$PWD
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_DEFAULT_MATMUL_PRECISION=highest
export WANDB_DIR=/scratch/11079/jeffreychen287/wandb_cache
export WANDB_CACHE_DIR=/scratch/11079/jeffreychen287/wandb_cache
mkdir -p /scratch/11079/jeffreychen287/wandb_cache

VENV_PYTHON=/scratch/11079/jeffreychen287/jax-aht/venv/bin/python

# Helper: run one experiment and wait for it to finish
run_exp() {
    local TASK=$1
    local N=$2
    local LABEL=$3
    local SEED=$4

    echo ">>> Starting: $LABEL (seed=$SEED, N=$N)"
    $VENV_PYTHON social_laws/run.py \
        task=$TASK \
        algorithm=creppo/continuous/coop_recon \
        algorithm.TRAIN_SEED=$SEED \
        algorithm.USE_SAME_SEED=true \
        algorithm.FIXED_EVAL=true \
        NUM_EXPT_AGENTS=$N \
        label=$LABEL \
        logger.project=aht-benchmark \
        logger.entity=jeffreychen287-the-university-of-texas-at-austin \
        logger.mode=online \
        +algorithm.ALPHA_VERIFICATION=false \
        >> logs/${LABEL}_seed${SEED}_${SLURM_JOB_ID}.out 2>&1
    echo "<<< Finished: $LABEL (seed=$SEED)"
}

echo "========================================"
echo "CREPPO Multi-Seed Phase E — H100 TACC"
echo "========================================"

# PI seed convention: base 72128 → 721280, 721281, 721282, 721283
for SEED in 721280 721281 721282 721283; do
    echo ""
    echo "=== SEED=$SEED ==="

    # No Law runs
    run_exp continuous/coop_recon_compare_no_law_2_agent 2 creppo_no_law_2_agent $SEED
    run_exp continuous/coop_recon_compare_no_law_3_agent 3 creppo_no_law_3_agent $SEED
    run_exp continuous/coop_recon_compare_no_law_4_agent 4 creppo_no_law_4_agent $SEED
    run_exp continuous/coop_recon_compare_no_law_5_agent 5 creppo_no_law_5_agent $SEED

    # Law runs
    run_exp continuous/coop_recon_compare_law_2_agent 2 creppo_law_2_agent $SEED
    run_exp continuous/coop_recon_compare_law_3_agent 3 creppo_law_3_agent $SEED
    run_exp continuous/coop_recon_compare_law_4_agent 4 creppo_law_4_agent $SEED
    run_exp continuous/coop_recon_compare_law_5_agent 5 creppo_law_5_agent $SEED

    echo "=== Seed $SEED complete ==="
done

echo ""
echo "All CREPPO multi-seed runs complete!"
