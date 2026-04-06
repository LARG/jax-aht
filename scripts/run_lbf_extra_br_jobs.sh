#!/usr/bin/env bash
set -euo pipefail

# Runs BR jobs for the extra LBF heldout teammates (LBRDiv + CoMeDi).
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
  local job_key="$2"
  local label="$3"
  local checkpoint_path="$4"

  if [[ -z "$checkpoint_path" ]]; then
    checkpoint_path="NOT_FOUND"
  fi

  {
    echo "[$(timestamp)] job_key=$job_key label=$label"
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
  local job_key="$1"
  local label="$2"
  local partner_override="$3"
  local gpu_id="$4"
  local checkpoint_log="${CHECKPOINT_LOG_PREFIX}_gpu${gpu_id}.txt"

  log_msg "START $job_key on GPU $gpu_id"

  CUDA_VISIBLE_DEVICES="$gpu_id" XLA_PYTHON_CLIENT_PREALLOCATE=false python3 ego_agent_training/run.py \
    task=lbf \
    algorithm=ppo_br/lbf \
    label="$label" \
    run_heldout_eval=false \
    algorithm.TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS" \
    logger.mode=online \
    ~algorithm.partner_agent \
    +algorithm.partner_agent="$partner_override"

  local ckpt_path
  ckpt_path="$(find results -path "*${label}*ego_train_run" | sort | tail -n 1 || true)"

  log_msg "DONE $job_key on GPU $gpu_id"
  record_checkpoint "$checkpoint_log" "$job_key" "$label" "$ckpt_path"
}

declare -a JOBS
add_job() {
  local job_key="$1"
  local label="$2"
  local partner_override="$3"
  JOBS+=("$job_key|$label|$partner_override")
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
  local spec job_key label partner_override gpu_id

  for spec in "${JOBS[@]}"; do
    IFS='|' read -r job_key label partner_override <<< "$spec"

    if already_logged "$job_key"; then
      log_msg "SKIP $job_key (already present in ${CHECKPOINT_LOG_PREFIX}_*.txt)"
      continue
    fi

    while [[ "$running" -ge "$max_parallel" ]]; do
      wait -n
      running=$((running - 1))
    done

    gpu_id="${GPU_IDS[$((idx % num_gpus))]}"

    run_job "$job_key" "$label" "$partner_override" "$gpu_id" &
    running=$((running + 1))
    idx=$((idx + 1))
  done

  while [[ "$running" -gt 0 ]]; do
    wait -n
    running=$((running - 1))
  done

  log_msg "LBF extra BR batch completed"
}

# LBF extra BR jobs (8 jobs)
add_job "lbf.br_for_lbrdiv_conf_1_0" "lbrdiv_conf_1_0_serious" "{lbrdiv-conf:{path:val_teammates/lbf/lbrdiv/2026-03-02_13-01-20/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:3,idx_list:[[1,0]],test_mode:false}}"
add_job "lbf.br_for_lbrdiv_conf_1_1" "lbrdiv_conf_1_1_serious" "{lbrdiv-conf:{path:val_teammates/lbf/lbrdiv/2026-03-02_13-01-20/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:3,idx_list:[[1,1]],test_mode:false}}"
add_job "lbf.br_for_lbrdiv_conf_1_2" "lbrdiv_conf_1_2_serious" "{lbrdiv-conf:{path:val_teammates/lbf/lbrdiv/2026-03-02_13-01-20/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:3,idx_list:[[1,2]],test_mode:false}}"
add_job "lbf.br_for_comedi_1_0" "comedi_1_0_serious" "{comedi:{path:val_teammates/lbf/comedi/2026-03-02_00-58-19/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:10,idx_list:[[1,0]],test_mode:false}}"
add_job "lbf.br_for_comedi_1_1" "comedi_1_1_serious" "{comedi:{path:val_teammates/lbf/comedi/2026-03-02_00-58-19/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:10,idx_list:[[1,1]],test_mode:false}}"
add_job "lbf.br_for_comedi_1_2" "comedi_1_2_serious" "{comedi:{path:val_teammates/lbf/comedi/2026-03-02_00-58-19/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:10,idx_list:[[1,2]],test_mode:false}}"
add_job "lbf.br_for_comedi_1_3" "comedi_1_3_serious" "{comedi:{path:val_teammates/lbf/comedi/2026-03-02_00-58-19/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:10,idx_list:[[1,3]],test_mode:false}}"
add_job "lbf.br_for_comedi_1_4" "comedi_1_4_serious" "{comedi:{path:val_teammates/lbf/comedi/2026-03-02_00-58-19/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:10,idx_list:[[1,4]],test_mode:false}}"

run_all_jobs