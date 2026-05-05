#!/bin/bash
# Phase E: Alpha Estimation CREPPO — Social Law Sweep (aaronson A100s)
#
# Setup before running:
#   cd /scratch/cluster/jeffrey9/jax-aht
#   git pull origin marl-compare
#   source venv_aaronson/bin/activate  (or whichever venv is on aaronson)
#   bash run_creppo_alpha_estimation_aaronson.sh

mkdir -p logs
mkdir -p /scratch/cluster/jeffrey9/wandb_cache

export PYTHONPATH=$PYTHONPATH:$PWD
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_DEFAULT_MATMUL_PRECISION=highest
export WANDB_DIR=/scratch/cluster/jeffrey9/wandb_cache
export WANDB_CACHE_DIR=/scratch/cluster/jeffrey9/wandb_cache

# Resolve venv python binary
# Prefer the currently activated venv (if user sourced one before running this script)
if [ -n "$VIRTUAL_ENV" ]; then
    VENV_PYTHON="$VIRTUAL_ENV/bin/python"
elif [ -f "venv_aaronson/bin/python" ]; then
    VENV_PYTHON="$PWD/venv_aaronson/bin/python"
elif [ -f "venv/bin/python" ]; then
    VENV_PYTHON="$PWD/venv/bin/python"
elif [ -f ".venv/bin/python" ]; then
    VENV_PYTHON="$PWD/.venv/bin/python"
else
    VENV_PYTHON="python"
fi

echo "Using Python: $VENV_PYTHON"

# ---------------------------------------------------------------------------
# run_exp GPU TASK N LABEL TRAIN_SEED EVAL_SEED USE_SAME_SEED [SOCIAL_LAW]
# ---------------------------------------------------------------------------
run_exp() {
    local GPU=$1
    local TASK=$2
    local N=$3
    local LABEL=$4
    local TRAIN_SEED=$5
    local EVAL_SEED=$6
    local USE_SAME_SEED=$7
    local SOCIAL_LAW_LABEL=${8:-"none"}
    local LOGFILE="logs/${LABEL}_tr${TRAIN_SEED}_ev${EVAL_SEED}_N${N}.out"

    echo ">>> GPU=$GPU | $LABEL | N=$N | train=$TRAIN_SEED eval=$EVAL_SEED"

    if [ "$SOCIAL_LAW_LABEL" != "none" ]; then
        CUDA_VISIBLE_DEVICES=$GPU \
        $VENV_PYTHON social_laws/run_w_best_case.py \
            task=$TASK \
            algorithm=creppo/continuous/coop_recon \
            algorithm.TRAIN_SEED=$TRAIN_SEED \
            algorithm.EVAL_SEED=$EVAL_SEED \
            algorithm.USE_SAME_SEED=$USE_SAME_SEED \
            algorithm.FIXED_EVAL=true \
            algorithm.ALPHA_VERIFICATION=true \
            NUM_EXPT_AGENTS=$N \
            label=$LABEL \
            social_law_label=$SOCIAL_LAW_LABEL \
            logger.project=aht-benchmark \
            logger.entity=jeffreychen287-the-university-of-texas-at-austin \
            logger.mode=online \
            >> $LOGFILE 2>&1 &
    else
        CUDA_VISIBLE_DEVICES=$GPU \
        $VENV_PYTHON social_laws/run_w_best_case.py \
            task=$TASK \
            algorithm=creppo/continuous/coop_recon \
            algorithm.TRAIN_SEED=$TRAIN_SEED \
            algorithm.EVAL_SEED=$EVAL_SEED \
            algorithm.USE_SAME_SEED=$USE_SAME_SEED \
            algorithm.FIXED_EVAL=true \
            algorithm.ALPHA_VERIFICATION=true \
            NUM_EXPT_AGENTS=$N \
            label=$LABEL \
            logger.project=aht-benchmark \
            logger.entity=jeffreychen287-the-university-of-texas-at-austin \
            logger.mode=online \
            >> $LOGFILE 2>&1 &
    fi
    echo "    PID=$! | log: $LOGFILE"
}

# Run all 4 N-agent sizes in parallel across GPUs 0-3, then wait
run_batch() {
    local TASK_PREFIX=$1
    local LABEL_PREFIX=$2
    local SOCIAL_LAW_LABEL=$3
    local TRAIN_SEED=$4
    local EVAL_SEED=$5
    local USE_SAME_SEED=$6

    run_exp 0 continuous/${TASK_PREFIX}_2_agent 2 ${LABEL_PREFIX}_2_agent $TRAIN_SEED $EVAL_SEED $USE_SAME_SEED "$SOCIAL_LAW_LABEL"
    run_exp 1 continuous/${TASK_PREFIX}_3_agent 3 ${LABEL_PREFIX}_3_agent $TRAIN_SEED $EVAL_SEED $USE_SAME_SEED "$SOCIAL_LAW_LABEL"
    run_exp 2 continuous/${TASK_PREFIX}_4_agent 4 ${LABEL_PREFIX}_4_agent $TRAIN_SEED $EVAL_SEED $USE_SAME_SEED "$SOCIAL_LAW_LABEL"
    run_exp 3 continuous/${TASK_PREFIX}_5_agent 5 ${LABEL_PREFIX}_5_agent $TRAIN_SEED $EVAL_SEED $USE_SAME_SEED "$SOCIAL_LAW_LABEL"
    wait
    echo "  Batch done: $LABEL_PREFIX (train=$TRAIN_SEED, eval=$EVAL_SEED)"
}

# ---------------------------------------------------------------------------
# Seed pairing: same-seed eval uses EVAL_SEED for both
# diff-seed eval uses PI's DIFF_TRAIN_SEED as train, EVAL_SEED as eval
# ---------------------------------------------------------------------------
EVAL_SEEDS=(72128 721280 721281 721282 721283)
DIFF_TRAIN_SEEDS=(174464134 1744641340 1744641341 1744641342 1744641343)

echo "================================================="
echo "CREPPO Alpha Estimation Sweep — aaronson A100s"
echo "================================================="

for i in "${!EVAL_SEEDS[@]}"; do
    EVAL_SEED=${EVAL_SEEDS[$i]}
    DIFF_TRAIN_SEED=${DIFF_TRAIN_SEEDS[$i]}

    echo ""
    echo "======================================================"
    echo "=== SEED INDEX $i (eval=$EVAL_SEED, diff_train=$DIFF_TRAIN_SEED) ==="
    echo "======================================================"

    # ---------- NO LAW ----------
    echo "--- NO LAW | Same Seed ---"
    run_batch "coop_recon_compare_no_law" "creppo_no_law" "none" $EVAL_SEED $EVAL_SEED "true"

    echo "--- NO LAW | Diff Seed (generalization) ---"
    run_batch "coop_recon_compare_no_law" "creppo_no_law_gen" "none" $DIFF_TRAIN_SEED $EVAL_SEED "false"

    # ---------- LAW 0.0 ----------
    echo "--- LAW 0.0 | Same Seed ---"
    run_batch "coop_recon_compare_law_0.0" "creppo_law_0.0" "law_0.0" $EVAL_SEED $EVAL_SEED "true"

    echo "--- LAW 0.0 | Diff Seed (generalization) ---"
    run_batch "coop_recon_compare_law_0.0" "creppo_law_0.0_gen" "law_0.0" $DIFF_TRAIN_SEED $EVAL_SEED "false"

    # ---------- LAW 0.1 ----------
    echo "--- LAW 0.1 | Same Seed ---"
    run_batch "coop_recon_compare_law_0.1" "creppo_law_0.1" "law_0.1" $EVAL_SEED $EVAL_SEED "true"

    echo "--- LAW 0.1 | Diff Seed (generalization) ---"
    run_batch "coop_recon_compare_law_0.1" "creppo_law_0.1_gen" "law_0.1" $DIFF_TRAIN_SEED $EVAL_SEED "false"

    # ---------- LAW 0.2 (task is coop_recon_compare_law, no 0.2 suffix) ----------
    echo "--- LAW 0.2 | Same Seed ---"
    run_batch "coop_recon_compare_law" "creppo_law_0.2" "law_0.2" $EVAL_SEED $EVAL_SEED "true"

    echo "--- LAW 0.2 | Diff Seed (generalization) ---"
    run_batch "coop_recon_compare_law" "creppo_law_0.2_gen" "law_0.2" $DIFF_TRAIN_SEED $EVAL_SEED "false"

done

echo ""
echo "All CREPPO Alpha Estimation runs finished."
