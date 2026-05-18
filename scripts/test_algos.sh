#!/usr/bin/env bash
# =============================================================================
# test_algos.sh — Smoke-test all algorithms in parallel across GPUs.
#
# PURPOSE
#   Runs a short training pass for every algorithm to verify that the code
#   executes end-to-end without errors.  All runs use tiny timestep budgets,
#   offline logging, and skip artifact saving so they finish quickly.
#
# USAGE
#   bash scripts/test_algos.sh
#   Run from anywhere; the script locates the repo root automatically.
#
# OUTPUT
#   Each job's stdout/stderr is written to:
#     results/test_algos_<timestamp>/<job_name>.log
#   A running status log and final summary are written to:
#     results/test_algos_<timestamp>/summary.log
#   The script exits 0 if all jobs passed, 1 if any failed.
#
# GPU REQUIREMENTS
#   Any GPU reported by nvidia-smi with >= 20 GB of free memory is used.
#   Jobs are distributed across available GPUs; up to N jobs run in parallel
#   where N = number of qualifying GPUs.  The script aborts if none qualify.
#
# SELECTING WHICH ALGORITHMS TO RUN
#   Edit the RUN_JOBS array (just below the job definitions).
#   List the job names you want to run, e.g.:
#     RUN_JOBS=(ippo brdiv liam_ego)
#   Leave it empty to run every defined job:
#     RUN_JOBS=()
#
# ADDING A NEW TEST
#   1. Append a name to JOB_NAMES and the matching command to JOB_CMDS in the
#      "Job definitions" section below.  The two arrays must stay in sync.
#   2. Add the name to RUN_JOBS (or leave RUN_JOBS empty to run all).
#   3. Each JOB_CMDS entry is a plain shell command string (no special quoting
#      needed; use backslash-newline for readability as shown below).
#   4. Use the shared COMMON_FLAGS variable for the standard set of flags that
#      disable logging and artifact saving.
#
# PREREQUISITES
#   nvidia-smi must be on PATH (standard on any machine with an Nvidia driver).
#   The conda/venv environment that can run the project must be active.
# =============================================================================
set -uo pipefail

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$REPO_ROOT/results/test_algos_$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/summary.log"

# ─── Shared flags ─────────────────────────────────────────────────────────────
# Appended to every command: disables W&B uploads and local artifact saving.
COMMON_FLAGS="logger.mode=offline logger.log_train_out=false logger.log_eval_out=false local_logger.save_train_out=false local_logger.save_eval_out=false"

# Path to a pre-trained IPPO partner checkpoint (required by ego-training algos).
PARTNER_PATH="eval_teammates/lbf/ippo/ippo-lbf-7-levels/saved_train_run/"

# =============================================================================
# ─── Job definitions ──────────────────────────────────────────────────────────
# To add a new test:
#   JOB_NAMES+=("my_algo_name")
#   JOB_CMDS+=("python my_module/run.py algorithm=my_algo/lbf ... $COMMON_FLAGS")
#
# The two arrays are parallel: JOB_NAMES[i] is the label for JOB_CMDS[i].
# Use backslash-newline for multi-line readability (bash strips them).
# =============================================================================
JOB_NAMES=()
JOB_CMDS=()

# ── MARL ──────────────────────────────────────────────────────────────────────
JOB_NAMES+=("ippo")
JOB_CMDS+=("python marl/run.py \
    algorithm=ippo/lbf/lbf_7x7_nolevels task=lbf/lbf_7x7_nolevels label=test_ippo \
    algorithm.NUM_SEEDS=1 \
    $COMMON_FLAGS")

# ── Teammate generation ───────────────────────────────────────────────────────
JOB_NAMES+=("brdiv")
JOB_CMDS+=("python teammate_generation/run.py \
    algorithm=brdiv/lbf/lbf_7x7_nolevels task=lbf/lbf_7x7_nolevels label=test_brdiv \
    run_heldout_eval=false train_ego=false \
    algorithm.TOTAL_TIMESTEPS=2e5 algorithm.PARTNER_POP_SIZE=2 algorithm.NUM_SEEDS=1 \
    $COMMON_FLAGS")

JOB_NAMES+=("lbrdiv")
JOB_CMDS+=("python teammate_generation/run.py \
    algorithm=lbrdiv/lbf/lbf_7x7_nolevels task=lbf/lbf_7x7_nolevels label=test_lbrdiv \
    run_heldout_eval=false train_ego=false \
    algorithm.TOTAL_TIMESTEPS=2e5 algorithm.PARTNER_POP_SIZE=2 algorithm.NUM_SEEDS=1 \
    $COMMON_FLAGS")

JOB_NAMES+=("comedi")
JOB_CMDS+=("python teammate_generation/run.py \
    algorithm=comedi/lbf/lbf_7x7_nolevels task=lbf/lbf_7x7_nolevels label=test_comedi \
    run_heldout_eval=false train_ego=false \
    algorithm.TOTAL_TIMESTEPS_PER_ITERATION=2e5 algorithm.PARTNER_POP_SIZE=2 algorithm.NUM_SEEDS=1 \
    $COMMON_FLAGS")

# test training ego agent
JOB_NAMES+=("fcp")
JOB_CMDS+=("python teammate_generation/run.py \
    algorithm=fcp/lbf/lbf_7x7_nolevels task=lbf/lbf_7x7_nolevels label=test_fcp \
    run_heldout_eval=false train_ego=true \
    algorithm.TOTAL_TIMESTEPS=1e5 algorithm.NUM_CHECKPOINTS=2 \
    algorithm.ego_train_algorithm.TOTAL_TIMESTEPS=1e5 \
    algorithm.PARTNER_POP_SIZE=2 algorithm.NUM_SEEDS=1 \
    $COMMON_FLAGS")

# ── Ego training ──────────────────────────────────────────────────────────────
# Run heldout eval for ppo ego only
JOB_NAMES+=("ppo_ego")
JOB_CMDS+=("python ego_agent_training/run.py \
    algorithm=ppo_ego/lbf/lbf_7x7_nolevels task=lbf/lbf_7x7_nolevels label=test_ppo_ego \
    algorithm.TOTAL_TIMESTEPS=1e5 algorithm.NUM_EGO_TRAIN_SEEDS=1 \
    algorithm.partner_agent.ippo.path=$PARTNER_PATH \
    run_heldout_eval=true \
    $COMMON_FLAGS")

JOB_NAMES+=("liam_ego")
JOB_CMDS+=("python ego_agent_training/run.py \
    algorithm=liam_ego/lbf/lbf_7x7_nolevels task=lbf/lbf_7x7_nolevels label=test_liam_ego \
    algorithm.TOTAL_TIMESTEPS=1e5 algorithm.NUM_EGO_TRAIN_SEEDS=1 \
    algorithm.partner_agent.ippo.path=$PARTNER_PATH \
    run_heldout_eval=false \
    $COMMON_FLAGS")

JOB_NAMES+=("meliba_ego")
JOB_CMDS+=("python ego_agent_training/run.py \
    algorithm=meliba_ego/lbf/lbf_7x7_nolevels task=lbf/lbf_7x7_nolevels label=test_meliba_ego \
    algorithm.TOTAL_TIMESTEPS=1e5 algorithm.NUM_EGO_TRAIN_SEEDS=1 \
    algorithm.partner_agent.ippo.path=$PARTNER_PATH \
    run_heldout_eval=false \
    $COMMON_FLAGS")

# ── Open-ended / unified training ─────────────────────────────────────────────
JOB_NAMES+=("rotate")
JOB_CMDS+=("python open_ended_training/run.py \
    algorithm=rotate/lbf/lbf_7x7_nolevels task=lbf/lbf_7x7_nolevels label=test_rotate \
    algorithm.NUM_OPEN_ENDED_ITERS=1 \
    algorithm.TIMESTEPS_PER_ITER_PARTNER=1e5 algorithm.TIMESTEPS_PER_ITER_EGO=1e5 \
    algorithm.NUM_SEEDS=1 \
    run_heldout_eval=false \
    $COMMON_FLAGS")

JOB_NAMES+=("cole")
JOB_CMDS+=("python open_ended_training/run.py \
    algorithm=cole/lbf/lbf_7x7_nolevels task=lbf/lbf_7x7_nolevels label=test_cole \
    algorithm.TOTAL_TIMESTEPS_PER_ITERATION=2e5 algorithm.PARTNER_POP_SIZE=2 algorithm.NUM_SEEDS=1 \
    run_heldout_eval=false \
    $COMMON_FLAGS")

JOB_NAMES+=("trajedi")
JOB_CMDS+=("python open_ended_training/run.py \
    algorithm=trajedi/lbf/lbf_7x7_nolevels task=lbf/lbf_7x7_nolevels label=test_trajedi \
    algorithm.TOTAL_TIMESTEPS=2e5 algorithm.PARTNER_POP_SIZE=2 algorithm.NUM_SEEDS=1 \
    run_heldout_eval=false \
    $COMMON_FLAGS")

# =============================================================================
# ─── Selection — which jobs to run ────────────────────────────────────────────
# List the job names you want to run.  Empty = run all defined jobs.
# Example (run a subset):
#   RUN_JOBS=(ippo brdiv liam_ego)
# =============================================================================
RUN_JOBS=(
    ippo
    brdiv
    lbrdiv
    comedi
    fcp
    liam_ego
    meliba_ego
    ppo_ego
    rotate
    cole
    trajedi
)

# =============================================================================
# ─── Implementation — no edits needed below this line ─────────────────────────
# =============================================================================

echo "Logs → $RESULTS_DIR" | tee "$LOG"

# ─── GPU detection ────────────────────────────────────────────────────────────
mapfile -t GPUS < <(
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',' '{ gsub(/ /,"",$1); gsub(/ /,"",$2); if ($2+0 >= 20000) print $1 }'
)

if [[ ${#GPUS[@]} -eq 0 ]]; then
    echo "ERROR: no GPUs with >= 20 GB free memory found. Aborting." | tee -a "$LOG" >&2
    exit 1
fi
echo "GPUs with >= 20 GB free: ${GPUS[*]}" | tee -a "$LOG"

# ─── Parallel job manager ─────────────────────────────────────────────────────
declare -a FREE_GPUS=("${GPUS[@]}")
declare -A PID_GPU    # pid -> gpu index
declare -A PID_NAME   # pid -> job name
SUCCEEDED=()
FAILED=()

# Reap any finished jobs and return their GPUs to the pool.
reap() {
    local -a done_pids=()
    for pid in "${!PID_GPU[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            done_pids+=("$pid")
        fi
    done
    for pid in "${done_pids[@]}"; do
        local name="${PID_NAME[$pid]}"
        local gpu="${PID_GPU[$pid]}"
        if wait "$pid"; then
            SUCCEEDED+=("$name")
            echo "[$(date '+%H:%M:%S')] [OK]   $name  (GPU $gpu)" | tee -a "$LOG"
        else
            FAILED+=("$name")
            echo "[$(date '+%H:%M:%S')] [FAIL] $name  (GPU $gpu) — see ${name}.log" | tee -a "$LOG"
        fi
        FREE_GPUS+=("$gpu")
        unset 'PID_GPU[$pid]'
        unset 'PID_NAME[$pid]'
    done
}

# Block until a GPU slot is free; sets FREE_GPU to the acquired GPU index.
acquire_gpu() {
    while [[ ${#FREE_GPUS[@]} -eq 0 ]]; do
        reap
        [[ ${#FREE_GPUS[@]} -eq 0 ]] && sleep 2
    done
    FREE_GPU="${FREE_GPUS[0]}"
    FREE_GPUS=("${FREE_GPUS[@]:1}")
}

# Launch a job: launch <name> <cmd...>
launch() {
    local name="$1"; shift
    acquire_gpu
    local gpu="$FREE_GPU"
    local logfile="$RESULTS_DIR/${name}.log"
    echo "[$(date '+%H:%M:%S')] [START] $name  (GPU $gpu)" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES="$gpu" "$@" >"$logfile" 2>&1 &
    local pid=$!
    PID_GPU[$pid]="$gpu"
    PID_NAME[$pid]="$name"
}

# Wait for all running jobs to finish.
wait_all() {
    while [[ ${#PID_GPU[@]} -gt 0 ]]; do
        reap
        [[ ${#PID_GPU[@]} -gt 0 ]] && sleep 2
    done
}

# ─── Dispatch selected jobs ───────────────────────────────────────────────────
cd "$REPO_ROOT"

# Build a lookup set from RUN_JOBS (empty RUN_JOBS = run everything).
declare -A _RUN_SET
if [[ ${#RUN_JOBS[@]} -gt 0 ]]; then
    for name in "${RUN_JOBS[@]}"; do _RUN_SET["$name"]=1; done
fi

for i in "${!JOB_NAMES[@]}"; do
    name="${JOB_NAMES[$i]}"
    if [[ ${#RUN_JOBS[@]} -eq 0 || -n "${_RUN_SET[$name]+x}" ]]; then
        launch "$name" bash -c "${JOB_CMDS[$i]}"
    else
        echo "[SKIP]  $name" | tee -a "$LOG"
    fi
done

wait_all

# ─── Summary ──────────────────────────────────────────────────────────────────
{
echo ""
echo "══════════════════════════════════════════════════════"
echo "  RESULTS"
echo "══════════════════════════════════════════════════════"
printf "  PASSED (%d):" "${#SUCCEEDED[@]}"; printf " %s" "${SUCCEEDED[@]:-}"; echo ""
printf "  FAILED (%d):" "${#FAILED[@]}";   printf " %s" "${FAILED[@]:-}";   echo ""
echo "══════════════════════════════════════════════════════"
echo "  Full logs: $RESULTS_DIR"
} | tee -a "$LOG"

[[ ${#FAILED[@]} -eq 0 ]] && exit 0 || exit 1
