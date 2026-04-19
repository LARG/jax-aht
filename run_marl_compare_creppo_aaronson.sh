#!/bin/bash
# CREPPO Missing-Seed Resume

mkdir -p logs

export PYTHONPATH=$PYTHONPATH:$PWD
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_DEFAULT_MATMUL_PRECISION=highest

# Prioritize the main venv (works on both aaronson and debruyne)
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

echo "========================================"
echo "CREPPO Missing-Seed Resume — aaronson A100"
echo "No-Law missing: 721282, 721283"
echo "Law missing:    721281, 721282, 721283"
echo "========================================"

# ── No Law: seeds 721282 and 721283 ──────────────────────────────────────────
for SEED in 721282 721283; do
    echo ""
    echo "=== NO-LAW SEED=$SEED ==="
    run_exp 0 continuous/coop_recon_compare_no_law_2_agent 2 creppo_no_law_2_agent $SEED
    run_exp 1 continuous/coop_recon_compare_no_law_3_agent 3 creppo_no_law_3_agent $SEED
    run_exp 2 continuous/coop_recon_compare_no_law_4_agent 4 creppo_no_law_4_agent $SEED
    run_exp 3 continuous/coop_recon_compare_no_law_5_agent 5 creppo_no_law_5_agent $SEED
    wait
    echo "=== No-Law Seed $SEED complete! ==="
done

# ── Law: seeds 721281, 721282, 721283 ────────────────────────────────────────
for SEED in 721281 721282 721283; do
    echo ""
    echo "=== LAW SEED=$SEED ==="
    run_exp 0 continuous/coop_recon_compare_law_2_agent 2 creppo_law_2_agent $SEED
    run_exp 1 continuous/coop_recon_compare_law_3_agent 3 creppo_law_3_agent $SEED
    run_exp 2 continuous/coop_recon_compare_law_4_agent 4 creppo_law_4_agent $SEED
    run_exp 3 continuous/coop_recon_compare_law_5_agent 5 creppo_law_5_agent $SEED
    wait
    echo "=== Law Seed $SEED complete! ==="
done

echo ""
echo "All missing CREPPO seeds complete!"
