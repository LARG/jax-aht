#!/bin/bash
# Phase E: train a Fictitious Co-Play (FCP) partner population on DSSE,
# then train PPO ego and LIAM ego separately against that population.
#
# Phase E is the headline validity experiment: the partner pool for the
# AHT ego runs comes from a real teammate_generation method, not just
# IPPO self-play checkpoints. Total wall-clock is ~20 min on 2 x RTX 6000 Ada
# (FCP 396 s on one GPU, then PPO ego 437 s and LIAM ego 287 s in parallel).
#
# Outputs:
#   results/dsse/fcp/fcp_dsse_2drone_v1/<ts>/saved_train_run        (FCP partner pool)
#   results/dsse/fcp/fcp_dsse_2drone_v1/<ts>/ego_train_run          (FCP-internal ego)
#   results/dsse/ppo_ego_s5/ego_ppo_fcp_v1/<ts>/ego_train_run       (PPO against FCP)
#   results/dsse/liam_ego_mlp/ego_liam_fcp_v1/<ts>/ego_train_run    (LIAM against FCP)
#
# Metrics JSONs are produced by:
#   writeup/extract_ego_metrics.py
#
# Usage (from repo root):
#   PYTHONPATH=. bash scripts/run_phase_e_fcp.sh
set -uo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-.}"
PY="${PY:-.venv/bin/python}"

LOG_DIR="${LOG_DIR:-/tmp/dsse_phase_e}"
mkdir -p "$LOG_DIR"

FCP_LABEL="fcp_dsse_2drone_v1"

echo "[$(date +%H:%M:%S)] Phase E step 1/3: training FCP partner population"
CUDA_VISIBLE_DEVICES=${FCP_GPU:-0} timeout 5400 \
    $PY teammate_generation/run.py \
        algorithm=fcp/dsse task=dsse label=$FCP_LABEL \
        algorithm.NUM_SEEDS=3 \
        algorithm.TOTAL_TIMESTEPS=10000000 \
        algorithm.NUM_ENVS=64 \
        algorithm.ego_train_algorithm.TOTAL_TIMESTEPS=10000000 \
        algorithm.ego_train_algorithm.NUM_ENVS=32 \
        algorithm.ego_train_algorithm.NUM_EGO_TRAIN_SEEDS=3 \
        run_heldout_eval=false \
        > "$LOG_DIR/${FCP_LABEL}.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
    echo "[$(date +%H:%M:%S)] FCP partner training FAILED (rc=$rc). tail:"
    tail -n 20 "$LOG_DIR/${FCP_LABEL}.log" | sed 's/^/  /'
    exit $rc
fi

# Find the most recent FCP timestamp dir.
FCP_RUN_DIR=$(ls -1dt results/dsse/fcp/${FCP_LABEL}/*/ 2>/dev/null | head -n 1)
if [ -z "$FCP_RUN_DIR" ]; then
    echo "[$(date +%H:%M:%S)] FCP run dir not found under results/dsse/fcp/${FCP_LABEL}/"
    exit 1
fi
FCP_PATH="${FCP_RUN_DIR}saved_train_run"
echo "[$(date +%H:%M:%S)] FCP partner pool at: $FCP_PATH"

run_ego() {
    local name="$1"
    local algo="$2"
    local gpu="$3"
    local log="$LOG_DIR/${name}.log"
    echo "[$(date +%H:%M:%S)] START $name (algo=$algo) on gpu $gpu"
    local t0
    t0=$(date +%s)
    if CUDA_VISIBLE_DEVICES=$gpu timeout 7200 \
        $PY ego_agent_training/run.py \
            algorithm=$algo task=dsse label=$name \
            algorithm.NUM_EGO_TRAIN_SEEDS=3 \
            algorithm.TOTAL_TIMESTEPS=10000000 \
            algorithm.NUM_ENVS=32 \
            algorithm.partner_agent.ippo.path=$FCP_PATH \
            algorithm.partner_agent.ippo.ckpt_key=checkpoints \
            algorithm.partner_agent.ippo.idx_list=null \
            "+algorithm.partner_agent.ippo.custom_loader={name: fcp}" \
            run_heldout_eval=false \
            > "$log" 2>&1; then
        echo "[$(date +%H:%M:%S)] PASS $name ($(( $(date +%s) - t0 ))s)"
    else
        rc=$?
        echo "[$(date +%H:%M:%S)] FAIL $name rc=$rc ($(( $(date +%s) - t0 ))s) tail:"
        tail -n 30 "$log" | sed 's/^/  /'
        return $rc
    fi
}

echo "[$(date +%H:%M:%S)] Phase E steps 2 and 3: standalone PPO and LIAM ego vs FCP pool"
run_ego ego_ppo_fcp_v1 ppo_ego/dsse ${PPO_GPU:-0} &
PID0=$!
run_ego ego_liam_fcp_v1 liam_ego/dsse ${LIAM_GPU:-1} &
PID1=$!

wait $PID0
rc0=$?
wait $PID1
rc1=$?

if [ $rc0 -ne 0 ] || [ $rc1 -ne 0 ]; then
    echo "[$(date +%H:%M:%S)] Phase E FAILED (ppo rc=$rc0 liam rc=$rc1)"
    exit 1
fi

echo "[$(date +%H:%M:%S)] Phase E complete. Extracting metrics:"
$PY writeup/extract_ego_metrics.py
