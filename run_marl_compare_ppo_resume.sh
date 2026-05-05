#!/bin/bash
# PPO Resume Script — picks up from where seed 721281 N=5 no_law crashed
# Seeds remaining: 721281 (partial), 721282, 721283
# All 4 GPUs available — N=5 gets dedicated GPU 3 (no more serialization)

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

run_ppo() {
    local GPU=$1
    local TASK=$2
    local N=$3
    local LABEL=$4
    local SEED=$5

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
        +algorithm.ALPHA_VERIFICATION=false \
        >> logs/${LABEL}_seed${SEED}.out 2>&1 &
}

echo "Starting PPO Phase E Multi-Seed Resume..."

# ---------------------------------------------------------------
# Seed 721281 — no_law N=2,3,4 already done; only N=5 failed.
# Resume: no_law N=5, then full law batch (all 4 N in parallel).
# ---------------------------------------------------------------
echo "=== Seed 721281: resuming no_law_5, then law batch ==="
SEED=721281

# No-law N=5 on GPU 3 (only missing piece)
run_ppo 3 continuous/coop_recon_compare_no_law_5_agent 5 ppo_no_law_5_agent $SEED
wait
echo "Seed 721281 no_law_5 done!"

# Law batch — all 4 N in parallel across all 4 GPUs
run_ppo 0 continuous/coop_recon_compare_law_2_agent 2 ppo_law_2_agent $SEED
run_ppo 1 continuous/coop_recon_compare_law_3_agent 3 ppo_law_3_agent $SEED
run_ppo 2 continuous/coop_recon_compare_law_4_agent 4 ppo_law_4_agent $SEED
run_ppo 3 continuous/coop_recon_compare_law_5_agent 5 ppo_law_5_agent $SEED
wait
echo "Seed 721281 fully complete!"

# ---------------------------------------------------------------
# Seed 721282 — full run, all 4 N in parallel per batch
# ---------------------------------------------------------------
echo "=== Seed 721282: full run ==="
SEED=721282

run_ppo 0 continuous/coop_recon_compare_no_law_2_agent 2 ppo_no_law_2_agent $SEED
run_ppo 1 continuous/coop_recon_compare_no_law_3_agent 3 ppo_no_law_3_agent $SEED
run_ppo 2 continuous/coop_recon_compare_no_law_4_agent 4 ppo_no_law_4_agent $SEED
run_ppo 3 continuous/coop_recon_compare_no_law_5_agent 5 ppo_no_law_5_agent $SEED
wait

run_ppo 0 continuous/coop_recon_compare_law_2_agent 2 ppo_law_2_agent $SEED
run_ppo 1 continuous/coop_recon_compare_law_3_agent 3 ppo_law_3_agent $SEED
run_ppo 2 continuous/coop_recon_compare_law_4_agent 4 ppo_law_4_agent $SEED
run_ppo 3 continuous/coop_recon_compare_law_5_agent 5 ppo_law_5_agent $SEED
wait
echo "Seed 721282 fully complete!"

# ---------------------------------------------------------------
# Seed 721283 — full run, all 4 N in parallel per batch
# ---------------------------------------------------------------
echo "=== Seed 721283: full run ==="
SEED=721283

run_ppo 0 continuous/coop_recon_compare_no_law_2_agent 2 ppo_no_law_2_agent $SEED
run_ppo 1 continuous/coop_recon_compare_no_law_3_agent 3 ppo_no_law_3_agent $SEED
run_ppo 2 continuous/coop_recon_compare_no_law_4_agent 4 ppo_no_law_4_agent $SEED
run_ppo 3 continuous/coop_recon_compare_no_law_5_agent 5 ppo_no_law_5_agent $SEED
wait

run_ppo 0 continuous/coop_recon_compare_law_2_agent 2 ppo_law_2_agent $SEED
run_ppo 1 continuous/coop_recon_compare_law_3_agent 3 ppo_law_3_agent $SEED
run_ppo 2 continuous/coop_recon_compare_law_4_agent 4 ppo_law_4_agent $SEED
run_ppo 3 continuous/coop_recon_compare_law_5_agent 5 ppo_law_5_agent $SEED
wait
echo "Seed 721283 fully complete!"

echo "All remaining PPO runs complete!"
