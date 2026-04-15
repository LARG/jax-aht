#!/bin/bash
# Phase E: MARL Comparison MAPPO — Multi-Seed (aaronson 4-GPU tmux launcher)
# Seeds follow PI convention: base seed 72128 → 721280, 721281, 721282, 721283

mkdir -p logs

export PYTHONPATH=$PYTHONPATH:$PWD
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_DEFAULT_MATMUL_PRECISION=highest

# Ensure venv is active
if [ -f "venv_aaronson/bin/python" ]; then
    VENV_PYTHON="$PWD/venv_aaronson/bin/python"
elif [ -f "venv/bin/python" ]; then
    VENV_PYTHON="$PWD/venv/bin/python"
elif [ -f ".venv/bin/python" ]; then
    VENV_PYTHON="$PWD/.venv/bin/python"
else
    VENV_PYTHON="python"
fi

# Redirect WandB artifacts to scratch to save home folder space
mkdir -p /scratch/cluster/jeffrey9/wandb_cache
export WANDB_DIR=/scratch/cluster/jeffrey9/wandb_cache
export WANDB_CACHE_DIR=/scratch/cluster/jeffrey9/wandb_cache

run_exp() {
    local GPU=$1
    local TASK=$2
    local N=$3
    local LABEL=$4
    local SEED=$5

    CUDA_VISIBLE_DEVICES=$GPU \
    $VENV_PYTHON social_laws/experiments/run_marl_compare_coop_recon.py \
        task=$TASK \
        algorithm=mappo/continuous/coop_recon \
        algorithm.TRAIN_SEED=$SEED \
        algorithm.USE_SAME_SEED=true \
        algorithm.FIXED_EVAL=true \
        NUM_EXPT_AGENTS=$N \
        label=$LABEL \
        logger.project=aht-benchmark \
        logger.entity=jeffreychen287-the-university-of-texas-at-austin \
        logger.mode=online \
        +task.ENV_KWARGS.world_state=true \
        >> logs/${LABEL}_seed${SEED}_${SLURM_JOB_ID:+$SLURM_JOB_ID}.out 2>&1 &
}

echo "Starting MAPPO Phase E Multi-Seed Comparisons (3 jobs parallel on GPUs 1,2,3 - avoid GPU 0)..."

# PI seed convention: original seed 72128 → append zero + increment
for SEED in 721280 721281 721282 721283; do
    echo "=== Running MAPPO with SEED=$SEED ==="

    # Batch 1: No Law (Baseline) — 3 parallel on GPUs 1,2,3 then N=5 serialized
    run_exp 0 continuous/coop_recon_compare_no_law_2_agent 2 mappo_no_law_2_agent $SEED
    run_exp 2 continuous/coop_recon_compare_no_law_3_agent 3 mappo_no_law_3_agent $SEED
    run_exp 3 continuous/coop_recon_compare_no_law_4_agent 4 mappo_no_law_4_agent $SEED
    wait
    run_exp 0 continuous/coop_recon_compare_no_law_5_agent 5 mappo_no_law_5_agent $SEED

    wait
    echo "Seed $SEED Batch 1 finished! Starting Batch 2..."

    # Batch 2: Social Law — 3 parallel on GPUs 0,2,3 then N=5 serialized
    run_exp 0 continuous/coop_recon_compare_law_2_agent 2 mappo_law_2_agent $SEED
    run_exp 2 continuous/coop_recon_compare_law_3_agent 3 mappo_law_3_agent $SEED
    run_exp 3 continuous/coop_recon_compare_law_4_agent 4 mappo_law_4_agent $SEED
    wait
    run_exp 0 continuous/coop_recon_compare_law_5_agent 5 mappo_law_5_agent $SEED

    wait
    echo "Seed $SEED Batch 2 finished!"
done

echo "All MAPPO multi-seed comparisons finished."
