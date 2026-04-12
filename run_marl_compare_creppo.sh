#!/bin/bash
# Phase E: MARL Comparison CREPPO (Hazard 4-GPU tmux launcher)

mkdir -p logs

export PYTHONPATH=$PYTHONPATH:$PWD
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_DEFAULT_MATMUL_PRECISION=highest

# Ensure venv is active
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Redirect WandB artifacts to a local directory to save home folder space
mkdir -p wandb_cache
export WANDB_DIR=$PWD/wandb_cache
export WANDB_CACHE_DIR=$PWD/wandb_cache

run_exp() {
    local GPU=$1
    local TASK=$2
    local N=$3
    local LABEL=$4
    local SEED=$5

    CUDA_VISIBLE_DEVICES=$GPU \
    python social_laws/run.py \
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
        >> logs/${LABEL}_seed${SEED}_${SLURM_JOB_ID:+$SLURM_JOB_ID}.out 2>&1 &
}

echo "Starting CREPPO Phase E Multi-Seed Comparisons (4 jobs on 4 GPUs per batch)..."

for SEED in 42 123 999; do
    echo "=== Running CREPPO with SEED=$SEED ==="

    # Batch 1: No Law (Baseline)
    run_exp 0 continuous/coop_recon_compare_no_law_2_agent 2 creppo_no_law_2_agent $SEED
    run_exp 1 continuous/coop_recon_compare_no_law_3_agent 3 creppo_no_law_3_agent $SEED
    run_exp 2 continuous/coop_recon_compare_no_law_4_agent 4 creppo_no_law_4_agent $SEED
    run_exp 3 continuous/coop_recon_compare_no_law_5_agent 5 creppo_no_law_5_agent $SEED

    wait
    echo "Seed $SEED Batch 1 finished! Starting Batch 2..."

    # Batch 2: Social Law
    run_exp 0 continuous/coop_recon_compare_law_2_agent 2 creppo_law_2_agent $SEED
    run_exp 1 continuous/coop_recon_compare_law_3_agent 3 creppo_law_3_agent $SEED
    run_exp 2 continuous/coop_recon_compare_law_4_agent 4 creppo_law_4_agent $SEED
    run_exp 3 continuous/coop_recon_compare_law_5_agent 5 creppo_law_5_agent $SEED

    wait
    echo "Seed $SEED Batch 2 finished!"
done

echo "All CREPPO multi-seed comparisons finished."
