#!/bin/bash
# Phase E: MARL Comparison IPPO (Hazard 4-GPU tmux launcher)

mkdir -p logs

export PYTHONPATH=$PYTHONPATH:$PWD
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_DEFAULT_MATMUL_PRECISION=highest

# Ensure venv is active
source /scratch/cluster/jeffrey9/jax-aht/venv/bin/activate

# Redirect WandB artifacts to scratch to save home folder space
mkdir -p /scratch/cluster/jeffrey9/wandb_cache
export WANDB_DIR=/scratch/cluster/jeffrey9/wandb_cache
export WANDB_CACHE_DIR=/scratch/cluster/jeffrey9/wandb_cache

run_exp() {
    local GPU=$1
    local TASK=$2
    local LABEL=$3
    local SEED=$4

    CUDA_VISIBLE_DEVICES=$GPU \
    ./marl_train social_laws/experiments/run_marl_compare_coop_recon.py \
        task=$TASK \
        algorithm=ippo/continuous/coop_recon \
        algorithm.TRAIN_SEED=$SEED \
        algorithm.USE_SAME_SEED=true \
        algorithm.FIXED_EVAL=true \
        label=$LABEL \
        logger.project=aht-benchmark \
        logger.entity=jeffreychen287-the-university-of-texas-at-austin \
        logger.mode=online \
        >> logs/${LABEL}_seed${SEED}_${SLURM_JOB_ID:+$SLURM_JOB_ID}.out 2>&1 &
}

echo "Starting IPPO Phase E Multi-Seed Comparisons (4 jobs on 4 GPUs per batch)..."

for SEED in 42 123 999; do
    echo "=== Running IPPO with SEED=$SEED ==="

    # Batch 1: No Law (Baseline)
    run_exp 0 continuous/coop_recon_compare_no_law_2_agent ippo_no_law_2_agent $SEED
    run_exp 1 continuous/coop_recon_compare_no_law_3_agent ippo_no_law_3_agent $SEED
    run_exp 2 continuous/coop_recon_compare_no_law_4_agent ippo_no_law_4_agent $SEED
    run_exp 3 continuous/coop_recon_compare_no_law_5_agent ippo_no_law_5_agent $SEED

    wait
    echo "Seed $SEED Batch 1 finished! Starting Batch 2..."

    # Batch 2: Social Law
    run_exp 0 continuous/coop_recon_compare_law_2_agent ippo_law_2_agent $SEED
    run_exp 1 continuous/coop_recon_compare_law_3_agent ippo_law_3_agent $SEED
    run_exp 2 continuous/coop_recon_compare_law_4_agent ippo_law_4_agent $SEED
    run_exp 3 continuous/coop_recon_compare_law_5_agent ippo_law_5_agent $SEED

    wait
    echo "Seed $SEED Batch 2 finished!"
done

echo "All IPPO multi-seed comparisons finished."
