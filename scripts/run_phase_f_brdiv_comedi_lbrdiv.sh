#!/bin/bash
# Phase F: integration check for BRDiv, CoMeDi, and LBRDiv on DSSE.
#
# Phase F is NOT a competitive comparison against the FCP baseline; it
# verifies that all three teammate generation pipelines integrate
# end-to-end with the DSSE task and the held-out evaluation harness.
# Each method runs at 5M partner steps / 2.5M ego steps / 1 seed,
# which is 4x below the paper-grade default of 20M / 10M and is below
# the noise floor of the DSSE sparse-reward coordination task. The output of
# this script is "the configs are valid, the loops compile, the
# checkpoints serialize, and the held-out eval consumes them via the
# same code path used by FCP in Phase E."
#
# To run a paper-scale Phase F, set TOTAL_TS=20000000, EGO_TS=10000000,
# NUM_SEEDS=3 (and budget several hours per method).
#
# Outputs:
#   results/dsse/brdiv/pr_long/<ts>/{saved_train_run,ego_train_run,heldout_eval_metrics}
#   results/dsse/comedi/pr_long/<ts>/{saved_train_run,ego_train_run,heldout_eval_metrics}
#   results/dsse/lbrdiv/pr_long/<ts>/{saved_train_run,ego_train_run,heldout_eval_metrics}
#
# Wall-clock on 2 x RTX 6000 Ada at the integration-check budget below
# is roughly 2 minutes for BRDiv, 4 minutes for CoMeDi, and 2 minutes
# for LBRDiv. BRDiv and CoMeDi run in parallel on GPU 0 and GPU 1;
# LBRDiv runs on GPU 0 after BRDiv finishes.
#
# Usage (from repo root):
#   PYTHONPATH=. bash scripts/run_phase_f_brdiv_comedi_lbrdiv.sh
set -uo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-.}"
PY="${PY:-.venv/bin/python}"

LOG_DIR="${LOG_DIR:-/tmp/dsse_phase_f}"
mkdir -p "$LOG_DIR"

LABEL="${LABEL:-pr_long}"
TOTAL_TS="${TOTAL_TS:-5000000}"
EGO_TS="${EGO_TS:-2500000}"
NUM_SEEDS="${NUM_SEEDS:-1}"
NUM_ENVS="${NUM_ENVS:-32}"

run_method() {
    local name="$1"
    local algo="$2"
    local ts_key="$3"   # TOTAL_TIMESTEPS or TOTAL_TIMESTEPS_PER_ITERATION
    local gpu="$4"
    local log="$LOG_DIR/${name}_${LABEL}.log"
    echo "[$(date +%H:%M:%S)] START $name on gpu $gpu (algo=$algo, $TOTAL_TS partner steps, $EGO_TS ego steps, $NUM_SEEDS seed)"
    local t0
    t0=$(date +%s)
    if CUDA_VISIBLE_DEVICES=$gpu timeout 7200 \
        $PY teammate_generation/run.py \
            algorithm=$algo task=dsse label=$LABEL \
            run_heldout_eval=true \
            algorithm.${ts_key}=$TOTAL_TS \
            algorithm.NUM_ENVS=$NUM_ENVS \
            algorithm.ego_train_algorithm.TOTAL_TIMESTEPS=$EGO_TS \
            algorithm.ego_train_algorithm.NUM_ENVS=$NUM_ENVS \
            algorithm.NUM_SEEDS=$NUM_SEEDS \
            > "$log" 2>&1; then
        echo "[$(date +%H:%M:%S)] PASS $name ($(( $(date +%s) - t0 ))s)"
        return 0
    else
        local rc=$?
        echo "[$(date +%H:%M:%S)] FAIL $name rc=$rc ($(( $(date +%s) - t0 ))s) tail:"
        tail -n 30 "$log" | sed 's/^/  /'
        return $rc
    fi
}

echo "[$(date +%H:%M:%S)] Phase F: BRDiv (gpu 0) and CoMeDi (gpu 1) in parallel"
run_method brdiv  brdiv/dsse  TOTAL_TIMESTEPS                ${BRDIV_GPU:-0}  &
PID_BR=$!
run_method comedi comedi/dsse TOTAL_TIMESTEPS_PER_ITERATION  ${COMEDI_GPU:-1} &
PID_CM=$!

wait $PID_BR
rc_br=$?
wait $PID_CM
rc_cm=$?

if [ $rc_br -ne 0 ] || [ $rc_cm -ne 0 ]; then
    echo "[$(date +%H:%M:%S)] Phase F FAILED at parallel step (brdiv rc=$rc_br comedi rc=$rc_cm)"
    exit 1
fi

echo "[$(date +%H:%M:%S)] Phase F: LBRDiv on gpu 0"
run_method lbrdiv lbrdiv/dsse TOTAL_TIMESTEPS ${LBRDIV_GPU:-0}
rc_lb=$?
if [ $rc_lb -ne 0 ]; then
    echo "[$(date +%H:%M:%S)] Phase F FAILED at LBRDiv (rc=$rc_lb)"
    exit 1
fi

echo "[$(date +%H:%M:%S)] Phase F complete. Extracting heldout metrics:"
$PY writeup/extract_phase_f_metrics.py
