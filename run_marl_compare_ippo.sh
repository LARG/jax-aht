#!/bin/bash
# Phase E: MARL Comparison IPPO (Hazard 4-GPU tmux launcher)

mkdir -p logs

export PYTHONPATH=$PYTHONPATH:$PWD
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_DEFAULT_MATMUL_PRECISION=highest

# Redirect WandB artifacts to scratch to save home folder space
mkdir -p /scratch/cluster/jeffrey9/wandb_cache
export WANDB_DIR=/scratch/cluster/jeffrey9/wandb_cache
export WANDB_CACHE_DIR=/scratch/cluster/jeffrey9/wandb_cache

run_exp() {
    local GPU=$1
    local TASK=$2
    local LABEL=$3

    CUDA_VISIBLE_DEVICES=$GPU \
    python social_laws/experiments/run_marl_compare_coop_recon.py \
        task=$TASK \
        algorithm=ippo/continuous/coop_recon \
        algorithm.TRAIN_SEED=72128 \
        algorithm.USE_SAME_SEED=true \
        algorithm.FIXED_EVAL=true \
        label=$LABEL \
        logger.project=NEURIPS-2026 \
        logger.mode=online \
        >> logs/${LABEL}_${SLURM_JOB_ID:+$SLURM_JOB_ID}.out 2>&1 &
}

echo "Starting IPPO Phase E Comparisons Batch 1 (4 jobs on 4 GPUs)..."

# Batch 1: No Law (Baseline)
run_exp 0 continuous/coop_recon_compare_no_law_2_agent ippo_no_law_2_agent
run_exp 1 continuous/coop_recon_compare_no_law_3_agent ippo_no_law_3_agent
run_exp 2 continuous/coop_recon_compare_no_law_4_agent ippo_no_law_4_agent
run_exp 3 continuous/coop_recon_compare_no_law_5_agent ippo_no_law_5_agent

wait
echo "Batch 1 finished! Starting Batch 2..."

# Batch 2: Social Law
run_exp 0 continuous/coop_recon_compare_law_2_agent ippo_law_2_agent
run_exp 1 continuous/coop_recon_compare_law_3_agent ippo_law_3_agent
run_exp 2 continuous/coop_recon_compare_law_4_agent ippo_law_4_agent
run_exp 3 continuous/coop_recon_compare_law_5_agent ippo_law_5_agent

wait
echo "All IPPO comparisons finished."
