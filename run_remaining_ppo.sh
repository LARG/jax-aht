#!/bin/bash
# Script to run the remaining PPO jobs for SEED 721283 sequentially on a single GPU.

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

mkdir -p /scratch/cluster/jeffrey9/wandb_cache
export WANDB_DIR=/scratch/cluster/jeffrey9/wandb_cache
export WANDB_CACHE_DIR=/scratch/cluster/jeffrey9/wandb_cache

# We remove the '&' at the end to make it run synchronously (one after the other)
run_exp_seq() {
    local GPU=$1
    local TASK=$2
    local N=$3
    local LABEL=$4
    local SEED=$5

    echo ">>> Running PPO $LABEL sequentially on GPU $GPU..."
    CUDA_VISIBLE_DEVICES=$GPU \
    $VENV_PYTHON social_laws/run.py \
        task=$TASK \
        algorithm=ppo/continuous/coop_recon \
        value_function=dqnppo/continuous/coop_recon \
        algorithm.TRAIN_SEED=$SEED \
        algorithm.USE_SAME_SEED=true \
        value_function.USE_SAME_SEED=true \
        algorithm.FIXED_EVAL=true \
        value_function.FIXED_EVAL=true \
        NUM_EXPT_AGENTS=$N \
        label=$LABEL \
        logger.project=aht-benchmark \
        logger.entity=jeffreychen287-the-university-of-texas-at-austin \
        logger.mode=online \
        algorithm.ALPHA_VERIFICATION=false \
        >> logs/${LABEL}_seed${SEED}_remaining.out 2>&1
}

# The user requested GPU 1
GPU=1
SEED=721283

echo "Starting remaining PPO runs for seed 721283 sequentially on GPU $GPU..."

# Law 0.0
# run_exp_seq $GPU continuous/coop_recon_compare_law_0.0_2_agent 2 ppo_law_0.0_2_agent $SEED
run_exp_seq $GPU continuous/coop_recon_compare_law_0.0_3_agent 3 ppo_law_0.0_3_agent $SEED
run_exp_seq $GPU continuous/coop_recon_compare_law_0.0_4_agent 4 ppo_law_0.0_4_agent $SEED
run_exp_seq $GPU continuous/coop_recon_compare_law_0.0_5_agent 5 ppo_law_0.0_5_agent $SEED

# Law 0.1
run_exp_seq $GPU continuous/coop_recon_compare_law_0.1_2_agent 2 ppo_law_0.1_2_agent $SEED
run_exp_seq $GPU continuous/coop_recon_compare_law_0.1_3_agent 3 ppo_law_0.1_3_agent $SEED
run_exp_seq $GPU continuous/coop_recon_compare_law_0.1_4_agent 4 ppo_law_0.1_4_agent $SEED
run_exp_seq $GPU continuous/coop_recon_compare_law_0.1_5_agent 5 ppo_law_0.1_5_agent $SEED

echo "All remaining PPO runs finished!"
