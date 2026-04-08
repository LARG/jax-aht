#!/bin/bash
# run_phase_c_population.sh -- DSSE JAX AHT Phase C: real partner population.
#
# The IPPO partner checkpoint at the path below was trained with NUM_SEEDS=3,
# so it actually contains 3 partners (leaf shape (3, ...)). Phase A/B used
# only partner index 0 via idx_list=[0]. Phase C lets the ego agent train
# against ALL 3 partners by setting idx_list=null.
#
# Hypothesis: with a real partner population (>1 partner) the LIAM encoder
# has something to disambiguate, so the LIAM > PPO gap should be larger
# and more consistent than in Phase A/B. If the gap stays the same, we
# need a more diverse population (Phase C2).
#
# Runs PPO ego (control) and LIAM ego on coordination with 10 ego seeds and
# all 3 partners. Reuses the existing partner checkpoint, no new partner
# training needed.

set -eo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

PARTNER_GLOB="results/dsse/ippo/marl_ippo_7x7_2drone_staghunt_seed42/*/saved_train_run"
PARTNER_PATH="$(ls -1dt $PARTNER_GLOB 2>/dev/null | head -n 1)"
if [ -z "$PARTNER_PATH" ] || [ ! -d "$PARTNER_PATH" ]; then
    if [ -d "evaluation/fixtures/dsse_ippo_seed42_pop3/saved_train_run" ]; then
        PARTNER_PATH="evaluation/fixtures/dsse_ippo_seed42_pop3/saved_train_run"
    else
        echo "ERROR: no seed-42 IPPO partner checkpoint found under $PARTNER_GLOB"
        echo "Run: bash scripts/regenerate_partner_fixtures.sh   to create one."
        exit 1
    fi
fi
echo "Phase C: using partner checkpoint $PARTNER_PATH"

LOG_DIR="/tmp/dsse_phase_c"
mkdir -p "$LOG_DIR"

# idx_list=null loads ALL checkpoints (3 partners)
COMMON=(
    task=dsse
    task.ENV_KWARGS.grid_size=7
    task.ENV_KWARGS.n_targets=1
    +task.ENV_KWARGS.target_cluster_radius=1
    +task.ENV_KWARGS.n_drones_to_rescue=2
    algorithm.partner_agent.ippo.path="$PARTNER_PATH"
    algorithm.partner_agent.ippo.actor_type=mlp
    algorithm.partner_agent.ippo.ckpt_key=final_params
    ~algorithm.partner_agent.ippo.idx_list
    +algorithm.partner_agent.ippo.idx_list=null
    algorithm.NUM_EGO_TRAIN_SEEDS=10
    algorithm.TOTAL_TIMESTEPS=10000000
    algorithm.NUM_ENVS=32
    run_heldout_eval=false
    logger.mode=offline
)

echo "=== Phase C: 3-partner population, 10 seeds ==="

CUDA_VISIBLE_DEVICES=0 .venv/bin/python ego_agent_training/run.py \
    algorithm=ppo_ego/dsse \
    "${COMMON[@]}" \
    label=phase_c_ppo_pop3_10seed > "$LOG_DIR/ppo_pop3.log" 2>&1 &
P1=$!
echo "  [GPU0] PPO pop3 pid=$P1"

CUDA_VISIBLE_DEVICES=1 .venv/bin/python ego_agent_training/run.py \
    algorithm=liam_ego/dsse \
    "${COMMON[@]}" \
    label=phase_c_liam_pop3_10seed > "$LOG_DIR/liam_pop3.log" 2>&1 &
P2=$!
echo "  [GPU1] LIAM pop3 pid=$P2"

wait $P1 && echo "  [GPU0] PPO pop3 DONE"
wait $P2 && echo "  [GPU1] LIAM pop3 DONE"

echo "=== Phase C DONE ==="
echo "Logs: $LOG_DIR/{ppo,liam}_pop3.log"
