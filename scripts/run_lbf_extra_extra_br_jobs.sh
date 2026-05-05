#!/usr/bin/env bash
set -euo pipefail

# Runs BR jobs for the 2 new LBF heldout teammates (greedy_closest_teammates and entitled).
# Example:
#   GPU_LIST=1,2 PARALLEL_JOBS=2 TOTAL_TIMESTEPS=10000000 bash scripts/run_lbf_extra_br_jobs.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GPU_LIST="${GPU_LIST:-1,2}"
PARALLEL_JOBS="${PARALLEL_JOBS:-0}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-10000000}"

RUN_LOG="lbf_extra_br_runs.log"
CHECKPOINT_LOG_PREFIX="lbf_extra_br_checkpoints"

mkdir -p results

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

log_msg() {
  local msg="$1"
  echo "[$(timestamp)] $msg" | tee -a "$RUN_LOG"
}

record_checkpoint() {
  local checkpoint_log="$1"
  local task_name="$2"
  local job_key="$3"
  local label="$4"
  local checkpoint_path="$5"

  if [[ -z "$checkpoint_path" ]]; then
    checkpoint_path="NOT_FOUND"
  fi

  {
    echo "[$(timestamp)] task_name=$task_name job_key=$job_key label=$label"
    echo "checkpoint_path=$checkpoint_path"
    echo
  } >> "$checkpoint_log"

  echo "checkpoint_path=$checkpoint_path"
}

already_logged() {
  local job_key="$1"
  local f
  for f in ${CHECKPOINT_LOG_PREFIX}_*.txt; do
    if [[ -f "$f" ]] && grep -q "job_key=$job_key " "$f" 2>/dev/null; then
      return 0
    fi
  done
  return 1
}

run_job() {
  local task_name="$1"
  local job_key="$2"
  local label="$3"
  local partner_override="$4"
  local gpu_id="$5"
  local checkpoint_log="${CHECKPOINT_LOG_PREFIX}_gpu${gpu_id}.txt"

  log_msg "START $job_key on GPU $gpu_id"

  CUDA_VISIBLE_DEVICES="$gpu_id" XLA_PYTHON_CLIENT_PREALLOCATE=false python3 ego_agent_training/run.py \
    task="$task_name" \
    algorithm="ppo_br/$task_name" \
    label="$label" \
    run_heldout_eval=false \
    algorithm.TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS" \
    logger.mode=online \
    ~algorithm.partner_agent \
    +algorithm.partner_agent="$partner_override"

  local ckpt_path
  ckpt_path="$(find results -path "*${label}*ego_train_run" | sort | tail -n 1 || true)"

  log_msg "DONE $job_key on GPU $gpu_id"
  record_checkpoint "$checkpoint_log" "$task_name" "$job_key" "$label" "$ckpt_path"
}

declare -a JOBS
add_job() {
  local task_name="$1"
  local job_key="$2"
  local label="$3"
  local partner_override="$4"
  JOBS+=("$task_name|$job_key|$label|$partner_override")
}

parse_gpu_list() {
  local raw="$1"
  local old_ifs="$IFS"
  IFS=',' read -r -a GPU_IDS <<< "$raw"
  IFS="$old_ifs"

  if [[ ${#GPU_IDS[@]} -eq 0 ]]; then
    echo "GPU_LIST is empty" >&2
    exit 1
  fi

  local i
  for i in "${!GPU_IDS[@]}"; do
    GPU_IDS[$i]="${GPU_IDS[$i]//[[:space:]]/}"
    if [[ -z "${GPU_IDS[$i]}" ]]; then
      echo "Invalid GPU id in GPU_LIST: '$raw'" >&2
      exit 1
    fi
  done
}

run_all_jobs() {
  parse_gpu_list "$GPU_LIST"

  local num_gpus="${#GPU_IDS[@]}"
  local max_parallel
  if [[ "$PARALLEL_JOBS" -le 0 ]]; then
    max_parallel="$num_gpus"
  else
    max_parallel="$PARALLEL_JOBS"
  fi
  if [[ "$max_parallel" -gt "$num_gpus" ]]; then
    max_parallel="$num_gpus"
  fi

  log_msg "LBF extra BR batch started"
  log_msg "GPU_LIST=$GPU_LIST PARALLEL_JOBS=$max_parallel"
  log_msg "TOTAL_TIMESTEPS=$TOTAL_TIMESTEPS"
  log_msg "Checkpoint logs: ${CHECKPOINT_LOG_PREFIX}_gpu<id>.txt"

  local running=0
  local idx=0
  local spec task_name job_key label partner_override gpu_id

  for spec in "${JOBS[@]}"; do
    IFS='|' read -r task_name job_key label partner_override <<< "$spec"

    if already_logged "$job_key"; then
      log_msg "SKIP $job_key (already present in ${CHECKPOINT_LOG_PREFIX}_*.txt)"
      continue
    fi

    while [[ "$running" -ge "$max_parallel" ]]; do
      wait -n
      running=$((running - 1))
    done

    gpu_id="${GPU_IDS[$((idx % num_gpus))]}"

    run_job "$task_name" "$job_key" "$label" "$partner_override" "$gpu_id" &
    running=$((running + 1))
    idx=$((idx + 1))
  done

  while [[ "$running" -gt 0 ]]; do
    wait -n
    running=$((running - 1))
  done

  log_msg "LBF extra BR batch completed"
}

# lbf_7x7_nolevels extra BR jobs (2 jobs)
add_job "lbf/lbf_7x7_nolevels" "lbf/lbf_7x7_nolevels.br_for_entitled_agent" "entitled_agent_serious" "{entitled_agent:{actor_type:entitled_agent}}"
add_job "lbf/lbf_7x7_nolevels" "lbf/lbf_7x7_nolevels.br_for_greedy_closest_teammate" "greedy_closest_teammate_serious" "{greedy_closest_teammate:{actor_type:greedy_agent,heuristic:closest_teammate}}"

run_all_jobs
