#!/bin/bash
# CREPPO Missing-Seed Resume — aaronson A100s (GPUs 0 and 1 only)
#
# Audit (from TACC scratch):
#   no_law N=2,3,4,5: seeds 42 ✅  721280 ✅  721281 ✅  721282 ❌  721283 ❌
#   law    N=2,3,4,5: seeds         721280 ✅  721281 ❌  721282 ❌  721283 ❌
#
# GPUs 2 & 3 occupied — using GPUs 0 and 1 only.
# Runs N=2,N=3 in parallel → wait → N=4,N=5 in parallel per seed.

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
        >> logs/${LABEL}_seed${SEED}.out 2>&1 &
}

run_cond() {
    local TASK_PREFIX=$1   # e.g. "coop_recon_compare_no_law"
    local LABEL_PREFIX=$2  # e.g. "creppo_no_law"
    local SEED=$3

    # Sub-batch A: N=2 on GPU 0, N=3 on GPU 1
    run_exp 0 continuous/${TASK_PREFIX}_2_agent 2 ${LABEL_PREFIX}_2_agent $SEED
    run_exp 1 continuous/${TASK_PREFIX}_3_agent 3 ${LABEL_PREFIX}_3_agent $SEED
    wait
    echo "  Sub-batch A (N=2,3) done for seed=$SEED"

    # Sub-batch B: N=4 on GPU 0, N=5 on GPU 1
    run_exp 0 continuous/${TASK_PREFIX}_4_agent 4 ${LABEL_PREFIX}_4_agent $SEED
    run_exp 1 continuous/${TASK_PREFIX}_5_agent 5 ${LABEL_PREFIX}_5_agent $SEED
    wait
    echo "  Sub-batch B (N=4,5) done for seed=$SEED"
}

echo "========================================"
echo "CREPPO Missing-Seed Resume — aaronson"
echo "GPUs: 0, 1 only (2 & 3 occupied)"
echo "No-Law missing: 721282, 721283"
echo "Law missing:    721281, 721282, 721283"
echo "========================================"

# ── No Law: seeds 721282 and 721283 ──────────────────────────────────────────
for SEED in 721282 721283; do
    echo ""
    echo "=== NO-LAW SEED=$SEED ==="
    run_cond "coop_recon_compare_no_law" "creppo_no_law" $SEED
    echo "=== No-Law Seed $SEED complete! ==="
done

# ── Law: seeds 721281, 721282, 721283 ────────────────────────────────────────
for SEED in 721281 721282 721283; do
    echo ""
    echo "=== LAW SEED=$SEED ==="
    run_cond "coop_recon_compare_law" "creppo_law" $SEED
    echo "=== Law Seed $SEED complete! ==="
done

echo ""
echo "All missing CREPPO seeds complete!"
