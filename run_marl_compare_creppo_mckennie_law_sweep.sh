#!/bin/bash
# Phase E: MARL Comparison CREPPO — Social Law Sweep (aaronson/mckennie)
# Seeds: 72128, 721280, 721281, 721282, 721283

mkdir -p logs

export PYTHONPATH=$PYTHONPATH:$PWD
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_DEFAULT_MATMUL_PRECISION=highest

# Ensure venv is active
if [ -f "venv/bin/python" ]; then
    VENV_PYTHON="$PWD/venv/bin/python"
elif [ -f "venv_aaronson/bin/python" ]; then
    VENV_PYTHON="$PWD/venv_aaronson/bin/python"
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

    echo ">>> Starting: $LABEL (seed=$SEED, N=$N) on GPU $GPU"
    CUDA_VISIBLE_DEVICES=$GPU \
    $VENV_PYTHON social_laws/run_w_best_case.py \
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
        algorithm.ALPHA_VERIFICATION=false \
        >> logs/${LABEL}_seed${SEED}.out 2>&1 &
    echo "    PID=$! | log: logs/${LABEL}_seed${SEED}.out"
}

run_cond() {
    local TASK_PREFIX=$1   # e.g. "coop_recon_compare_law_0.1"
    local LABEL_PREFIX=$2  # e.g. "creppo_law_0.1"
    local SEED=$3

    # All 4 N-values in parallel across all 4 GPUs
    run_exp 0 continuous/${TASK_PREFIX}_2_agent 2 ${LABEL_PREFIX}_2_agent $SEED
    run_exp 1 continuous/${TASK_PREFIX}_3_agent 3 ${LABEL_PREFIX}_3_agent $SEED
    run_exp 2 continuous/${TASK_PREFIX}_4_agent 4 ${LABEL_PREFIX}_4_agent $SEED
    run_exp 3 continuous/${TASK_PREFIX}_5_agent 5 ${LABEL_PREFIX}_5_agent $SEED
    wait
    echo "  All N done for seed=$SEED"
}

echo "Starting CREPPO Social Law Sweep Comparisons (0.0 and 0.1 only)..."
for SEED in 72128 721280 721281 721282 721283; do
    echo "=== Running CREPPO with SEED=$SEED ==="

    echo "--- Variation: Law 0.0 ---"
    run_cond "coop_recon_compare_law_0.0" "creppo_law_0.0" $SEED
    echo "--- Finished Law 0.0 for seed $SEED ---"

    echo "--- Variation: Law 0.1 ---"
    run_cond "coop_recon_compare_law_0.1" "creppo_law_0.1" $SEED
    echo "--- Finished Law 0.1 for seed $SEED ---"

done

echo "All CREPPO social law sweep comparisons finished."
