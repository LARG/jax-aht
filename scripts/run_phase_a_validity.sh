#!/bin/bash
# run_phase_a_validity.sh -- DSSE JAX AHT Phase A validity ablation.
#
# Question: is the LIAM > PPO gap on the 7x7 coordination variant a real
# coordination signal, or is the benchmark exploration-bound?
#
# Three conditions, all on coordination (n_drones_to_rescue=2), 5 ego seeds
# each, 10M env steps, 32 parallel envs:
#
#   1. PPO ego                          (no teammate model)
#   2. LIAM ego, trained encoder        (current setup)
#   3. LIAM ego, FROZEN encoder/decoder (random init, never updated)
#
# Decision rule:
#   - if (3) >= (2) within noise -> exploration-bound, redesign env
#   - if (3) << (2)              -> real teammate-modelling signal,
#                                   proceed to Phase B (10 seeds, table)
#
# Conditions 1 and 2 are reproductions of the existing writeup numbers
# at 5 seeds instead of 3 so they share statistical strength with (3).
#
# GPU layout: PPO on GPU 0, both LIAM runs on GPU 1 (LIAM is small).
# Runtime estimate: ~15 min wall time on a single RTX 6000 Ada.
#
# Usage:
#   bash scripts/run_phase_a_validity.sh

set -eo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

PARTNER_GLOB="results/dsse/ippo/marl_ippo_7x7_2drone_staghunt_seed42/*/saved_train_run"
PARTNER_PATH="$(ls -1dt $PARTNER_GLOB 2>/dev/null | head -n 1)"
if [ -z "$PARTNER_PATH" ] || [ ! -d "$PARTNER_PATH" ]; then
    # Fall back to the stable fixture symlink populated by
    # scripts/regenerate_partner_fixtures.sh.
    if [ -d "evaluation/fixtures/dsse_ippo_seed42_pop3/saved_train_run" ]; then
        PARTNER_PATH="evaluation/fixtures/dsse_ippo_seed42_pop3/saved_train_run"
    else
        echo "ERROR: no seed-42 IPPO partner checkpoint found under $PARTNER_GLOB"
        echo "Run: bash scripts/regenerate_partner_fixtures.sh   to create one."
        exit 1
    fi
fi
echo "Phase A: using partner checkpoint $PARTNER_PATH"

LOG_DIR="/tmp/dsse_phase_a"
mkdir -p "$LOG_DIR"

COMMON=(
    task=dsse
    task.ENV_KWARGS.grid_size=7
    task.ENV_KWARGS.n_targets=1
    +task.ENV_KWARGS.target_cluster_radius=1
    +task.ENV_KWARGS.n_drones_to_rescue=2
    algorithm.partner_agent.ippo.path="$PARTNER_PATH"
    algorithm.partner_agent.ippo.actor_type=mlp
    algorithm.partner_agent.ippo.ckpt_key=final_params
    algorithm.partner_agent.ippo.idx_list=[0]
    algorithm.NUM_EGO_TRAIN_SEEDS=5
    algorithm.TOTAL_TIMESTEPS=10000000
    algorithm.NUM_ENVS=32
    run_heldout_eval=false
    logger.mode=offline
)

echo "=== launching Phase A (3 conditions, 5 seeds each) ==="

# 1. PPO ego on GPU 0
CUDA_VISIBLE_DEVICES=0 .venv/bin/python ego_agent_training/run.py \
    algorithm=ppo_ego/dsse \
    "${COMMON[@]}" \
    label=phase_a_ppo_ego_5seed \
    > "$LOG_DIR/ppo_ego.log" 2>&1 &
PPO_PID=$!
echo "  [GPU0] PPO ego pid=$PPO_PID  log=$LOG_DIR/ppo_ego.log"

# 2. LIAM ego (trained encoder) on GPU 1
CUDA_VISIBLE_DEVICES=1 .venv/bin/python ego_agent_training/run.py \
    algorithm=liam_ego/dsse \
    "${COMMON[@]}" \
    label=phase_a_liam_trained_5seed \
    > "$LOG_DIR/liam_trained.log" 2>&1 &
LIAM_TRAINED_PID=$!
echo "  [GPU1] LIAM trained pid=$LIAM_TRAINED_PID  log=$LOG_DIR/liam_trained.log"

wait $PPO_PID && echo "  [GPU0] PPO ego DONE" || echo "  [GPU0] PPO ego FAILED"
wait $LIAM_TRAINED_PID && echo "  [GPU1] LIAM trained DONE" || echo "  [GPU1] LIAM trained FAILED"

# 3. LIAM ego (FROZEN encoder/decoder) on GPU 1 (sequential after liam_trained
#    so we don't OOM if both run at once; LIAM is fast)
CUDA_VISIBLE_DEVICES=1 .venv/bin/python ego_agent_training/run.py \
    algorithm=liam_ego/dsse \
    +algorithm.FREEZE_ENCODER_DECODER=true \
    "${COMMON[@]}" \
    label=phase_a_liam_frozen_5seed \
    > "$LOG_DIR/liam_frozen.log" 2>&1
echo "  [GPU1] LIAM frozen DONE"

echo
echo "=== ALL DONE ==="
echo "Logs: $LOG_DIR/{ppo_ego,liam_trained,liam_frozen}.log"
echo "Run dirs are under: results/dsse/{ppo_ego_s5,liam_ego_mlp}/phase_a_*"
