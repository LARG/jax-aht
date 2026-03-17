#!/usr/bin/env bash
set -euo pipefail

# Runs LBF BR jobs, supports parallel GPU workers, and records checkpoints persistently.

# GPU_LIST=1,2 PARALLEL_JOBS=2 bash scripts/run_lbf_br_jobs.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GPU_LIST="${GPU_LIST:-1,2}"
PARALLEL_JOBS="${PARALLEL_JOBS:-0}"
# Training budget per job. Override with TOTAL_TIMESTEPS=... at launch.
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-10000000}"

RUN_LOG="lbf_br_runs.log"
CHECKPOINT_LOG_PREFIX="lbf_br_checkpoints"

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

  if already_logged "$job_key"; then
    log_msg "SKIP $job_key (already present in ${CHECKPOINT_LOG_PREFIX}_*.txt)"
    return
  fi

  log_msg "START $job_key on GPU $gpu_id"

  CUDA_VISIBLE_DEVICES="$gpu_id" python3 ego_agent_training/run.py \
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

  log_msg "LBF BR batch started"
  log_msg "GPU_LIST=$GPU_LIST PARALLEL_JOBS=$max_parallel"
  log_msg "TOTAL_TIMESTEPS=$TOTAL_TIMESTEPS"
  log_msg "Checkpoint logs: ${CHECKPOINT_LOG_PREFIX}_gpu<id>.txt"

  local running=0
  local idx=0
  local spec job_key label partner_override gpu_id

  for spec in "${JOBS[@]}"; do
    while [[ "$running" -ge "$max_parallel" ]]; do
      wait -n
      running=$((running - 1))
    done

    IFS='|' read -r job_key label partner_override <<< "$spec"
    gpu_id="${GPU_IDS[$((idx % num_gpus))]}"

    run_job "$job_key" "$label" "$partner_override" "$gpu_id" &
    running=$((running + 1))
    idx=$((idx + 1))
  done

  while [[ "$running" -gt 0 ]]; do
    wait -n
    running=$((running - 1))
  done

  log_msg "LBF BR batch completed"
}

# LBF BR jobs (13 jobs)
add_job "lbf.br_for_ippo_mlp_0" "ippo_mlp_0_serious" "{ippo_mlp:{path:eval_teammates/lbf/ippo/2025-04-21_23-41-17/saved_train_run,actor_type:mlp,ckpt_key:final_params,idx_list:[0],test_mode:false}}"
add_job "lbf.br_for_ippo_mlp_s2c0_2_0" "ippo_mlp_s2c0_serious" "{ippo_mlp_s2c0:{path:eval_teammates/lbf/ippo/2025-04-21_23-41-17/saved_train_run,actor_type:mlp,idx_list:[[2,0]],test_mode:false}}"
add_job "lbf.br_for_brdiv_conf1_0" "brdiv_conf1_0_serious" "{brdiv-conf1:{path:eval_teammates/lbf/brdiv/2025-04-16/11-32-07/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:5,idx_list:[0],test_mode:false}}"
add_job "lbf.br_for_brdiv_conf1_1" "brdiv_conf1_1_serious" "{brdiv-conf1:{path:eval_teammates/lbf/brdiv/2025-04-16/11-32-07/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:5,idx_list:[1],test_mode:false}}"
add_job "lbf.br_for_brdiv_conf1_2" "brdiv_conf1_2_serious" "{brdiv-conf1:{path:eval_teammates/lbf/brdiv/2025-04-16/11-32-07/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:5,idx_list:[2],test_mode:false}}"
add_job "lbf.br_for_brdiv_conf2_0" "brdiv_conf2_0_serious" "{brdiv-conf2:{path:eval_teammates/lbf/brdiv/2025-04-23/13-48-47/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:3,idx_list:[0],test_mode:false}}"
add_job "lbf.br_for_brdiv_conf2_1" "brdiv_conf2_1_serious" "{brdiv-conf2:{path:eval_teammates/lbf/brdiv/2025-04-23/13-48-47/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:3,idx_list:[1],test_mode:false}}"
add_job "lbf.br_for_seq_agent_lexi" "seq_agent_lexi_serious" "{seq_agent_lexi:{actor_type:seq_agent,ordering_strategy:lexicographic}}"
add_job "lbf.br_for_seq_agent_rlexi" "seq_agent_rlexi_serious" "{seq_agent_rlexi:{actor_type:seq_agent,ordering_strategy:reverse_lexicographic}}"
add_job "lbf.br_for_seq_agent_col" "seq_agent_col_serious" "{seq_agent_col:{actor_type:seq_agent,ordering_strategy:column_major}}"
add_job "lbf.br_for_seq_agent_rcol" "seq_agent_rcol_serious" "{seq_agent_rcol:{actor_type:seq_agent,ordering_strategy:reverse_column_major}}"
add_job "lbf.br_for_seq_agent_nearest" "seq_agent_nearest_serious" "{seq_agent_nearest:{actor_type:seq_agent,ordering_strategy:nearest_agent}}"
add_job "lbf.br_for_seq_agent_farthest" "seq_agent_farthest_serious" "{seq_agent_farthest:{actor_type:seq_agent,ordering_strategy:farthest_agent}}"

run_all_jobs
