#!/bin/bash
#----------------------------------------------------
# Phase E: Alpha Estimation CREPPO — Social Law Sweep — H100 (Stampede3)
#
# Contains both SAME SEED (baseline) and DIFF SEED (generalization) runs
# required for empirical alpha bounds computation.
#----------------------------------------------------
#SBATCH -J creppo_alpha_sweep
#SBATCH -o logs/creppo_alpha_sweep_%j.out
#SBATCH -e logs/creppo_alpha_sweep_%j.err
#SBATCH -p h100
#SBATCH -A IRI26004
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=96
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

run_exp() {
    local GPU=$1
    local TASK=$2
    local N=$3
    local LABEL=$4
    local TRAIN_SEED=$5
    local EVAL_SEED=$6
    local USE_SAME_SEED=$7
    local SOCIAL_LAW_LABEL=$8

    echo ">>> Starting: $LABEL (train_seed=$TRAIN_SEED, eval_seed=$EVAL_SEED, N=$N) on GPU $GPU"
    
    # Construct the command
    local CMD="$VENV_PYTHON social_laws/run_w_best_case.py \
        task=$TASK \
        algorithm=creppo/continuous/coop_recon \
        algorithm.TRAIN_SEED=$TRAIN_SEED \
        algorithm.EVAL_SEED=$EVAL_SEED \
        algorithm.USE_SAME_SEED=$USE_SAME_SEED \
        algorithm.FIXED_EVAL=true \
        NUM_EXPT_AGENTS=$N \
        label=$LABEL \
        logger.project=aht-benchmark \
        logger.entity=jeffreychen287-the-university-of-texas-at-austin \
        logger.mode=online \
        algorithm.ALPHA_VERIFICATION=true"

    if [ "$SOCIAL_LAW_LABEL" != "none" ]; then
        CMD="$CMD social_law_label=\"$SOCIAL_LAW_LABEL\""
    fi

    CUDA_VISIBLE_DEVICES=$GPU $CMD >> logs/${LABEL}_train${TRAIN_SEED}_eval${EVAL_SEED}_N${N}_${SLURM_JOB_ID}.out 2>&1 &
}

run_batch() {
    local TASK_PREFIX=$1
    local LABEL_PREFIX=$2
    local SOCIAL_LAW_LABEL=$3
    local TRAIN_SEED=$4
    local EVAL_SEED=$5
    local USE_SAME_SEED=$6

    # H100 nodes have 4 GPUs, so we can run 4 experiments in parallel perfectly!
    run_exp 0 continuous/${TASK_PREFIX}_2_agent 2 ${LABEL_PREFIX}_2_agent $TRAIN_SEED $EVAL_SEED $USE_SAME_SEED "$SOCIAL_LAW_LABEL"
    run_exp 1 continuous/${TASK_PREFIX}_3_agent 3 ${LABEL_PREFIX}_3_agent $TRAIN_SEED $EVAL_SEED $USE_SAME_SEED "$SOCIAL_LAW_LABEL"
    run_exp 2 continuous/${TASK_PREFIX}_4_agent 4 ${LABEL_PREFIX}_4_agent $TRAIN_SEED $EVAL_SEED $USE_SAME_SEED "$SOCIAL_LAW_LABEL"
    run_exp 3 continuous/${TASK_PREFIX}_5_agent 5 ${LABEL_PREFIX}_5_agent $TRAIN_SEED $EVAL_SEED $USE_SAME_SEED "$SOCIAL_LAW_LABEL"
    wait
}

echo "================================================="
echo "CREPPO Alpha Estimation Sweep — H100 TACC"
echo "================================================="

EVAL_SEEDS=(72128 721280 721281 721282 721283)
DIFF_TRAIN_SEEDS=(174464134 1744641340 1744641341 1744641342 1744641343)

for i in ${!EVAL_SEEDS[@]}; do
    EVAL_SEED=${EVAL_SEEDS[$i]}
    DIFF_TRAIN_SEED=${DIFF_TRAIN_SEEDS[$i]}

    echo "======================================================"
    echo "=== Running SEED INDEX $i (Eval Seed: $EVAL_SEED) ==="
    echo "======================================================"

    # --- NO LAW ---
    # Same Seed (Baseline)
    echo "--- Condition: NO LAW (Same Seed) ---"
    run_batch "coop_recon_compare_no_law" "creppo_no_law" "none" $EVAL_SEED $EVAL_SEED "true"
    # Diff Seed (Generalization)
    echo "--- Condition: NO LAW (Diff Seed) ---"
    run_batch "coop_recon_compare_no_law" "creppo_no_law_gen" "none" $DIFF_TRAIN_SEED $EVAL_SEED "false"

    # --- LAW 0.0 ---
    echo "--- Condition: LAW 0.0 (Same Seed) ---"
    run_batch "coop_recon_compare_law_0.0" "creppo_law_0.0" "law_0.0" $EVAL_SEED $EVAL_SEED "true"
    echo "--- Condition: LAW 0.0 (Diff Seed) ---"
    run_batch "coop_recon_compare_law_0.0" "creppo_law_0.0_gen" "law_0.0" $DIFF_TRAIN_SEED $EVAL_SEED "false"

    # --- LAW 0.1 ---
    echo "--- Condition: LAW 0.1 (Same Seed) ---"
    run_batch "coop_recon_compare_law_0.1" "creppo_law_0.1" "law_0.1" $EVAL_SEED $EVAL_SEED "true"
    echo "--- Condition: LAW 0.1 (Diff Seed) ---"
    run_batch "coop_recon_compare_law_0.1" "creppo_law_0.1_gen" "law_0.1" $DIFF_TRAIN_SEED $EVAL_SEED "false"

    # --- LAW 0.2 ---
    echo "--- Condition: LAW 0.2 (Same Seed) ---"
    run_batch "coop_recon_compare_law" "creppo_law_0.2" "law_0.2" $EVAL_SEED $EVAL_SEED "true"
    echo "--- Condition: LAW 0.2 (Diff Seed) ---"
    run_batch "coop_recon_compare_law" "creppo_law_0.2_gen" "law_0.2" $DIFF_TRAIN_SEED $EVAL_SEED "false"

done

echo ""
echo "All CREPPO Alpha Estimation runs complete!"
