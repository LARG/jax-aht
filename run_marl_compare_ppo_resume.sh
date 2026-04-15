#!/bin/bash
# PPO Resume Script — picks up from where seed 721281 N=5 no_law crashed
# Seeds remaining: 721281 (partial), 721282, 721283
# All 4 GPUs available (Isaac Sim gone from GPU 0)

mkdir -p logs

export PYTHONPATH=$PYTHONPATH:$PWD
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_DEFAULT_MATMUL_PRECISION=highest

# Use venv_aaronson if present, else freshly built venv
if [ -f "venv_aaronson/bin/python" ]; then
    VENV_PYTHON="$PWD/venv_aaronson/bin/python"
elif [ -f "venv/bin/python" ]; then
    VENV_PYTHON="$PWD/venv/bin/python"
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

echo "=== PPO RESUME: Completing seed 721281 from N=5 no_law ==="

# Seed 721281 — only no_law_5 failed; skip N=2,3,4 (already done)
SEED=721281
run_ppo 0 continuous/coop_recon_compare_no_law_5_agent 5 ppo_no_law_5_agent $SEED
wait
echo "Seed 721281 no_law_5 done! Starting law batch..."

run_ppo 0 continuous/coop_recon_compare_law_2_agent 2 ppo_law_2_agent $SEED
run_ppo 1 continuous/coop_recon_compare_law_3_agent 3 ppo_law_3_agent $SEED
run_ppo 2 continuous/coop_recon_compare_law_4_agent 4 ppo_law_4_agent $SEED
wait
run_ppo 0 continuous/coop_recon_compare_law_5_agent 5 ppo_law_5_agent $SEED
wait
echo "Seed 721281 fully complete!"

echo "=== PPO RESUME: Full run for seed 721282 ==="
SEED=721282
run_ppo 0 continuous/coop_recon_compare_no_law_2_agent 2 ppo_no_law_2_agent $SEED
run_ppo 1 continuous/coop_recon_compare_no_law_3_agent 3 ppo_no_law_3_agent $SEED
run_ppo 2 continuous/coop_recon_compare_no_law_4_agent 4 ppo_no_law_4_agent $SEED
wait
run_ppo 0 continuous/coop_recon_compare_no_law_5_agent 5 ppo_no_law_5_agent $SEED
wait

run_ppo 0 continuous/coop_recon_compare_law_2_agent 2 ppo_law_2_agent $SEED
run_ppo 1 continuous/coop_recon_compare_law_3_agent 3 ppo_law_3_agent $SEED
run_ppo 2 continuous/coop_recon_compare_law_4_agent 4 ppo_law_4_agent $SEED
wait
run_ppo 0 continuous/coop_recon_compare_law_5_agent 5 ppo_law_5_agent $SEED
wait
echo "Seed 721282 fully complete!"

echo "=== PPO RESUME: Full run for seed 721283 ==="
SEED=721283
run_ppo 0 continuous/coop_recon_compare_no_law_2_agent 2 ppo_no_law_2_agent $SEED
run_ppo 1 continuous/coop_recon_compare_no_law_3_agent 3 ppo_no_law_3_agent $SEED
run_ppo 2 continuous/coop_recon_compare_no_law_4_agent 4 ppo_no_law_4_agent $SEED
wait
run_ppo 0 continuous/coop_recon_compare_no_law_5_agent 5 ppo_no_law_5_agent $SEED
wait

run_ppo 0 continuous/coop_recon_compare_law_2_agent 2 ppo_law_2_agent $SEED
run_ppo 1 continuous/coop_recon_compare_law_3_agent 3 ppo_law_3_agent $SEED
run_ppo 2 continuous/coop_recon_compare_law_4_agent 4 ppo_law_4_agent $SEED
wait
run_ppo 0 continuous/coop_recon_compare_law_5_agent 5 ppo_law_5_agent $SEED
wait
echo "Seed 721283 fully complete!"

echo "All remaining PPO runs complete!"
