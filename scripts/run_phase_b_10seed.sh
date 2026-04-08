#!/bin/bash
# run_phase_b_10seed.sh -- DSSE JAX AHT Phase B: 10-seed final table.
#
# Runs PPO ego and LIAM ego on baseline (ndr=1) and coordination (ndr=2)
# with 10 ego seeds each (vs 3 in Phase 0 and 5 in Phase A). Same single
# IPPO partner as the original writeup so the result is directly
# comparable to writeup section 1.
#
# Phase B is only launched if Phase A confirmed the LIAM > PPO gap is
# real (not exploration-bound). See scripts/run_phase_a_validity.sh.
#
# Runtime: ~25 min wall time on a single RTX 6000 Ada per condition.
# Total ~50 min if PPO and LIAM run concurrently across the two GPUs.

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
echo "Phase B: using partner checkpoint $PARTNER_PATH"

LOG_DIR="/tmp/dsse_phase_b"
mkdir -p "$LOG_DIR"

COMMON=(
    task=dsse
    task.ENV_KWARGS.grid_size=7
    task.ENV_KWARGS.n_targets=1
    +task.ENV_KWARGS.target_cluster_radius=1
    algorithm.partner_agent.ippo.path="$PARTNER_PATH"
    algorithm.partner_agent.ippo.actor_type=mlp
    algorithm.partner_agent.ippo.ckpt_key=final_params
    algorithm.partner_agent.ippo.idx_list=[0]
    algorithm.NUM_EGO_TRAIN_SEEDS=10
    algorithm.TOTAL_TIMESTEPS=10000000
    algorithm.NUM_ENVS=32
    run_heldout_eval=false
    logger.mode=offline
)

run_one() {
    local gpu="$1"
    local algo="$2"
    local label="$3"
    local extra="$4"
    local logfile="$LOG_DIR/${label}.log"
    echo "  [GPU${gpu}] ${label} -> ${logfile}"
    CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python ego_agent_training/run.py \
        algorithm="$algo" \
        "${COMMON[@]}" \
        $extra \
        label="${label}" > "$logfile" 2>&1
}

echo "=== Phase B: 10-seed table (4 runs) ==="

# Coordination variant first (the headline) - GPU0=PPO, GPU1=LIAM, in parallel
run_one 0 ppo_ego/dsse  phase_b_ppo_ndr2_10seed  "+task.ENV_KWARGS.n_drones_to_rescue=2" &
P1=$!
run_one 1 liam_ego/dsse phase_b_liam_ndr2_10seed "+task.ENV_KWARGS.n_drones_to_rescue=2" &
P2=$!
wait $P1 && echo "  [GPU0] PPO ndr2 DONE"
wait $P2 && echo "  [GPU1] LIAM ndr2 DONE"

# Baseline second - parallel across GPUs
run_one 0 ppo_ego/dsse  phase_b_ppo_ndr1_10seed  "" &
P3=$!
run_one 1 liam_ego/dsse phase_b_liam_ndr1_10seed "" &
P4=$!
wait $P3 && echo "  [GPU0] PPO ndr1 DONE"
wait $P4 && echo "  [GPU1] LIAM ndr1 DONE"

echo "=== Phase B DONE ==="
echo "Logs: $LOG_DIR/phase_b_*.log"
