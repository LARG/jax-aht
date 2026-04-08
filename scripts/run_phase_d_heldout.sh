#!/bin/bash
# run_phase_d_heldout.sh -- DSSE JAX AHT Phase D: held-out generalization.
#
# After Phase B/C confirm that the LIAM > PPO gap is real on the training
# partners, Phase D tests whether the gap survives held-out partners. The
# held-out set is defined in evaluation/configs/global_heldout_settings.yaml
# under heldout_set.dsse: ippo_seed123_pop3 (3 IPPO partners trained from a
# different seed than the ego-training partners).
#
# We re-run PPO and LIAM ego on the same 7x7 coordination env with
# run_heldout_eval=true so that the held-out generalization metric is
# computed and logged at every checkpoint, then dump
# .../ego_train_run/heldout_eval_metrics.pkl into JSON via the existing
# extract_ego_metrics.py.
#
# This is the publishable headline number: trained-partner return AND
# held-out-partner return for both methods.
#
# Runtime: ~25 min wall time per condition; PPO and LIAM run in parallel
# across the two GPUs.

set -eo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

PARTNER_GLOB="results/dsse/ippo/marl_ippo_7x7_2drone_staghunt_seed42/*/saved_train_run"
PARTNER_PATH="$(ls -1dt $PARTNER_GLOB 2>/dev/null | head -n 1)"
if [ -z "$PARTNER_PATH" ] || [ ! -d "$PARTNER_PATH" ]; then
    if [ -d "evaluation/fixtures/dsse_ippo_seed42_pop3/saved_train_run" ]; then
        PARTNER_PATH="evaluation/fixtures/dsse_ippo_seed42_pop3/saved_train_run"
    else
        echo "ERROR: no seed-42 IPPO training partner found under $PARTNER_GLOB"
        echo "Run: bash scripts/regenerate_partner_fixtures.sh   to create one."
        exit 1
    fi
fi

HELDOUT_GLOB="results/dsse/ippo/marl_ippo_7x7_2drone_staghunt_seed123/*/saved_train_run"
HELDOUT_PATH="$(ls -1dt $HELDOUT_GLOB 2>/dev/null | head -n 1)"
if [ -z "$HELDOUT_PATH" ] || [ ! -d "$HELDOUT_PATH" ]; then
    if [ -d "evaluation/fixtures/dsse_ippo_seed123_pop3/saved_train_run" ]; then
        HELDOUT_PATH="evaluation/fixtures/dsse_ippo_seed123_pop3/saved_train_run"
    else
        echo "ERROR: no seed-123 IPPO held-out partner found under $HELDOUT_GLOB"
        echo "Run: bash scripts/regenerate_partner_fixtures.sh   to create one."
        exit 1
    fi
fi
echo "Phase D: training partner=$PARTNER_PATH"
echo "Phase D: held-out partner=$HELDOUT_PATH"

LOG_DIR="/tmp/dsse_phase_d"
mkdir -p "$LOG_DIR"

# Same env / training settings as Phase B but with run_heldout_eval=true.
# Phase D uses the SAME 3-partner training population as Phase C so that
# the trained-partner return is directly comparable.
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
    run_heldout_eval=true
    logger.mode=offline
)

echo "=== Phase D: trained vs held-out, 10 seeds each ==="

CUDA_VISIBLE_DEVICES=0 .venv/bin/python ego_agent_training/run.py \
    algorithm=ppo_ego/dsse \
    "${COMMON[@]}" \
    label=phase_d_ppo_pop3_heldout > "$LOG_DIR/ppo.log" 2>&1 &
P1=$!
echo "  [GPU0] PPO + heldout pid=$P1"

CUDA_VISIBLE_DEVICES=1 .venv/bin/python ego_agent_training/run.py \
    algorithm=liam_ego/dsse \
    "${COMMON[@]}" \
    label=phase_d_liam_pop3_heldout > "$LOG_DIR/liam.log" 2>&1 &
P2=$!
echo "  [GPU1] LIAM + heldout pid=$P2"

wait $P1 && echo "  [GPU0] PPO DONE"
wait $P2 && echo "  [GPU1] LIAM DONE"

echo "=== Phase D DONE ==="
echo "Logs: $LOG_DIR/{ppo,liam}.log"
echo "Run dirs: results/dsse/{ppo_ego_s5,liam_ego_mlp}/phase_d_*"
