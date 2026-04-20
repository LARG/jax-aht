#!/bin/bash
# CREPPO Full Rerun w/ Best Case — mckennie (all 4 A100s)
#
# Uses run_w_best_case.py which trains:
#   1. SAP (single-agent projections, worst-case)
#   2. Worst-case joint policy
#   3. Best-case joint policy  ← new, required for alpha comparison
#
# Seeds: 42, 72128, 721280, 721281, 721282, 721283
# Conditions: no_law and law, N=2,3,4,5


mkdir -p logs

export PYTHONPATH=$PYTHONPATH:$PWD
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_DEFAULT_MATMUL_PRECISION=highest

if [ -f "venv/bin/python" ]; then
    VENV_PYTHON="$PWD/venv/bin/python"
elif [ -f "venv_aaronson/bin/python" ]; then
    VENV_PYTHON="$PWD/venv_aaronson/bin/python"
else
    VENV_PYTHON="python"
fi

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
    local TASK_PREFIX=$1   # e.g. "coop_recon_compare_no_law"
    local LABEL_PREFIX=$2  # e.g. "creppo_no_law"
    local SEED=$3

    # All 4 N-values in parallel across all 4 GPUs
    run_exp 0 continuous/${TASK_PREFIX}_2_agent 2 ${LABEL_PREFIX}_2_agent $SEED
    run_exp 1 continuous/${TASK_PREFIX}_3_agent 3 ${LABEL_PREFIX}_3_agent $SEED
    run_exp 2 continuous/${TASK_PREFIX}_4_agent 4 ${LABEL_PREFIX}_4_agent $SEED
    run_exp 3 continuous/${TASK_PREFIX}_5_agent 5 ${LABEL_PREFIX}_5_agent $SEED
    wait
    echo "  All N done for seed=$SEED"
}

echo "========================================================"
echo "CREPPO Full Rerun w/ Best Case — mckennie (4x A100)"
echo "Entry point: social_laws/run_w_best_case.py"
echo "Seeds: 42, 72128, 721280, 721281, 721282, 721283"
echo "Conditions: no_law + law, N=2,3,4,5"
echo "========================================================"

# ── No Law ───────────────────────────────────────────────────────────────────
for SEED in 42 72128 721280 721281 721282 721283; do
    echo ""
    echo "=== NO-LAW SEED=$SEED ==="
    run_cond "coop_recon_compare_no_law" "creppo_no_law" $SEED
    echo "=== No-Law Seed $SEED complete! ==="
done

# ── Law ──────────────────────────────────────────────────────────────────────
for SEED in 42 72128 721280 721281 721282 721283; do
    echo ""
    echo "=== LAW SEED=$SEED ==="
    run_cond "coop_recon_compare_law" "creppo_law" $SEED
    echo "=== Law Seed $SEED complete! ==="
done

echo ""
echo "All CREPPO best-case runs complete!"
