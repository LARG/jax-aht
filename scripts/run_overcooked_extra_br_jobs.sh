#!/usr/bin/env bash
set -euo pipefail

# Runs BR jobs for extra Overcooked heldout teammates (LBRDiv + CoMeDi).
# Example:
#   GPU_LIST=1,2 PARALLEL_JOBS=2 TOTAL_TIMESTEPS=10000000 bash scripts/run_overcooked_extra_br_jobs.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GPU_LIST="${GPU_LIST:-1,2}"
PARALLEL_JOBS="${PARALLEL_JOBS:-0}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-10000000}"

RUN_LOG="overcooked_extra_br_runs.log"
CHECKPOINT_LOG_PREFIX="overcooked_extra_br_checkpoints"

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
  local layout="$2"
  local job_key="$3"
  local label="$4"
  local checkpoint_path="$5"

  if [[ -z "$checkpoint_path" ]]; then
    checkpoint_path="NOT_FOUND"
  fi

  {
    echo "[$(timestamp)] layout=$layout job_key=$job_key label=$label"
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
  local layout="$1"
  local job_key="$2"
  local label="$3"
  local partner_override="$4"
  local gpu_id="$5"
  local checkpoint_log="${CHECKPOINT_LOG_PREFIX}_gpu${gpu_id}.txt"

  log_msg "START $job_key on GPU $gpu_id"

  CUDA_VISIBLE_DEVICES="$gpu_id" XLA_PYTHON_CLIENT_PREALLOCATE=false python3 ego_agent_training/run.py \
    task="overcooked-v1/$layout" \
    algorithm="ppo_br/overcooked-v1/$layout" \
    label="$label" \
    run_heldout_eval=false \
    algorithm.TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS" \
    logger.mode=online \
    ~algorithm.partner_agent \
    +algorithm.partner_agent="$partner_override"

  local ckpt_path
  ckpt_path="$(find results -path "*${label}*ego_train_run" | sort | tail -n 1 || true)"

  log_msg "DONE $job_key on GPU $gpu_id"
  record_checkpoint "$checkpoint_log" "$layout" "$job_key" "$label" "$ckpt_path"
}

declare -a JOBS
add_job() {
  local layout="$1"
  local job_key="$2"
  local label="$3"
  local partner_override="$4"
  JOBS+=("$layout|$job_key|$label|$partner_override")
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

  log_msg "Overcooked extra BR batch started"
  log_msg "GPU_LIST=$GPU_LIST PARALLEL_JOBS=$max_parallel"
  log_msg "TOTAL_TIMESTEPS=$TOTAL_TIMESTEPS"
  log_msg "Checkpoint logs: ${CHECKPOINT_LOG_PREFIX}_gpu<id>.txt"

  local running=0
  local idx=0
  local spec layout job_key label partner_override gpu_id

  for spec in "${JOBS[@]}"; do
    IFS='|' read -r layout job_key label partner_override <<< "$spec"

    if already_logged "$job_key"; then
      log_msg "SKIP $job_key (already present in ${CHECKPOINT_LOG_PREFIX}_*.txt)"
      continue
    fi

    while [[ "$running" -ge "$max_parallel" ]]; do
      wait -n
      running=$((running - 1))
    done

    gpu_id="${GPU_IDS[$((idx % num_gpus))]}"

    run_job "$layout" "$job_key" "$label" "$partner_override" "$gpu_id" &
    running=$((running + 1))
    idx=$((idx + 1))
  done

  while [[ "$running" -gt 0 ]]; do
    wait -n
    running=$((running - 1))
  done

  log_msg "Overcooked extra BR batch completed"
}

# cramped_room extra teammates (8 jobs)
add_job "cramped_room" "cramped_room.br_for_lbrdiv_conf_1_0" "cramped_room_lbrdiv_conf_1_0_serious" "{lbrdiv-conf:{path:val_teammates/overcooked-v1/cramped_room/lbrdiv/2026-03-03_19-04-46/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:3,idx_list:[[1,0]],test_mode:false}}"
add_job "cramped_room" "cramped_room.br_for_lbrdiv_conf_1_1" "cramped_room_lbrdiv_conf_1_1_serious" "{lbrdiv-conf:{path:val_teammates/overcooked-v1/cramped_room/lbrdiv/2026-03-03_19-04-46/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:3,idx_list:[[1,1]],test_mode:false}}"
add_job "cramped_room" "cramped_room.br_for_lbrdiv_conf_1_2" "cramped_room_lbrdiv_conf_1_2_serious" "{lbrdiv-conf:{path:val_teammates/overcooked-v1/cramped_room/lbrdiv/2026-03-03_19-04-46/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:3,idx_list:[[1,2]],test_mode:false}}"
add_job "cramped_room" "cramped_room.br_for_comedi_1_0" "cramped_room_comedi_1_0_serious" "{comedi:{path:val_teammates/overcooked-v1/cramped_room/comedi/2026-03-03_22-28-24/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:10,idx_list:[[1,0]],test_mode:false}}"
add_job "cramped_room" "cramped_room.br_for_comedi_1_1" "cramped_room_comedi_1_1_serious" "{comedi:{path:val_teammates/overcooked-v1/cramped_room/comedi/2026-03-03_22-28-24/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:10,idx_list:[[1,1]],test_mode:false}}"
add_job "cramped_room" "cramped_room.br_for_comedi_1_2" "cramped_room_comedi_1_2_serious" "{comedi:{path:val_teammates/overcooked-v1/cramped_room/comedi/2026-03-03_22-28-24/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:10,idx_list:[[1,2]],test_mode:false}}"
add_job "cramped_room" "cramped_room.br_for_comedi_1_3" "cramped_room_comedi_1_3_serious" "{comedi:{path:val_teammates/overcooked-v1/cramped_room/comedi/2026-03-03_22-28-24/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:10,idx_list:[[1,3]],test_mode:false}}"
add_job "cramped_room" "cramped_room.br_for_comedi_1_4" "cramped_room_comedi_1_4_serious" "{comedi:{path:val_teammates/overcooked-v1/cramped_room/comedi/2026-03-03_22-28-24/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:10,idx_list:[[1,4]],test_mode:false}}"

# asymm_advantages extra teammates (8 jobs)
add_job "asymm_advantages" "asymm_advantages.br_for_lbrdiv_conf_1_0" "asymm_advantages_lbrdiv_conf_1_0_serious" "{lbrdiv-conf:{path:val_teammates/overcooked-v1/asymm_advantages/lbrdiv/2026-03-03_17-02-53/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:3,idx_list:[[1,0]],test_mode:false}}"
add_job "asymm_advantages" "asymm_advantages.br_for_lbrdiv_conf_1_1" "asymm_advantages_lbrdiv_conf_1_1_serious" "{lbrdiv-conf:{path:val_teammates/overcooked-v1/asymm_advantages/lbrdiv/2026-03-03_17-02-53/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:3,idx_list:[[1,1]],test_mode:false}}"
add_job "asymm_advantages" "asymm_advantages.br_for_lbrdiv_conf_1_2" "asymm_advantages_lbrdiv_conf_1_2_serious" "{lbrdiv-conf:{path:val_teammates/overcooked-v1/asymm_advantages/lbrdiv/2026-03-03_17-02-53/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:3,idx_list:[[1,2]],test_mode:false}}"
add_job "asymm_advantages" "asymm_advantages.br_for_comedi_1_0" "asymm_advantages_comedi_1_0_serious" "{comedi:{path:val_teammates/overcooked-v1/asymm_advantages/comedi/2026-03-03_17-06-25/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:10,idx_list:[[1,0]],test_mode:false}}"
add_job "asymm_advantages" "asymm_advantages.br_for_comedi_1_1" "asymm_advantages_comedi_1_1_serious" "{comedi:{path:val_teammates/overcooked-v1/asymm_advantages/comedi/2026-03-03_17-06-25/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:10,idx_list:[[1,1]],test_mode:false}}"
add_job "asymm_advantages" "asymm_advantages.br_for_comedi_1_2" "asymm_advantages_comedi_1_2_serious" "{comedi:{path:val_teammates/overcooked-v1/asymm_advantages/comedi/2026-03-03_17-06-25/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:10,idx_list:[[1,2]],test_mode:false}}"
add_job "asymm_advantages" "asymm_advantages.br_for_comedi_1_3" "asymm_advantages_comedi_1_3_serious" "{comedi:{path:val_teammates/overcooked-v1/asymm_advantages/comedi/2026-03-03_17-06-25/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:10,idx_list:[[1,3]],test_mode:false}}"
add_job "asymm_advantages" "asymm_advantages.br_for_comedi_1_4" "asymm_advantages_comedi_1_4_serious" "{comedi:{path:val_teammates/overcooked-v1/asymm_advantages/comedi/2026-03-03_17-06-25/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:10,idx_list:[[1,4]],test_mode:false}}"

# counter_circuit extra teammates (8 jobs)
add_job "counter_circuit" "counter_circuit.br_for_lbrdiv_conf_1_0" "counter_circuit_lbrdiv_conf_1_0_serious" "{lbrdiv-conf:{path:val_teammates/overcooked-v1/counter_circuit/lbrdiv/2026-03-03_17-49-14/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:3,idx_list:[[1,0]],test_mode:false}}"
add_job "counter_circuit" "counter_circuit.br_for_lbrdiv_conf_1_1" "counter_circuit_lbrdiv_conf_1_1_serious" "{lbrdiv-conf:{path:val_teammates/overcooked-v1/counter_circuit/lbrdiv/2026-03-03_17-49-14/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:3,idx_list:[[1,1]],test_mode:false}}"
add_job "counter_circuit" "counter_circuit.br_for_lbrdiv_conf_1_2" "counter_circuit_lbrdiv_conf_1_2_serious" "{lbrdiv-conf:{path:val_teammates/overcooked-v1/counter_circuit/lbrdiv/2026-03-03_17-49-14/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:3,idx_list:[[1,2]],test_mode:false}}"
add_job "counter_circuit" "counter_circuit.br_for_comedi_1_0" "counter_circuit_comedi_1_0_serious" "{comedi:{path:val_teammates/overcooked-v1/counter_circuit/comedi/2026-03-03_19-57-43/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:10,idx_list:[[1,0]],test_mode:false}}"
add_job "counter_circuit" "counter_circuit.br_for_comedi_1_1" "counter_circuit_comedi_1_1_serious" "{comedi:{path:val_teammates/overcooked-v1/counter_circuit/comedi/2026-03-03_19-57-43/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:10,idx_list:[[1,1]],test_mode:false}}"
add_job "counter_circuit" "counter_circuit.br_for_comedi_1_2" "counter_circuit_comedi_1_2_serious" "{comedi:{path:val_teammates/overcooked-v1/counter_circuit/comedi/2026-03-03_19-57-43/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:10,idx_list:[[1,2]],test_mode:false}}"
add_job "counter_circuit" "counter_circuit.br_for_comedi_1_3" "counter_circuit_comedi_1_3_serious" "{comedi:{path:val_teammates/overcooked-v1/counter_circuit/comedi/2026-03-03_19-57-43/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:10,idx_list:[[1,3]],test_mode:false}}"
add_job "counter_circuit" "counter_circuit.br_for_comedi_1_4" "counter_circuit_comedi_1_4_serious" "{comedi:{path:val_teammates/overcooked-v1/counter_circuit/comedi/2026-03-03_19-57-43/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:10,idx_list:[[1,4]],test_mode:false}}"

# coord_ring extra teammates (8 jobs)
add_job "coord_ring" "coord_ring.br_for_lbrdiv_conf_1_0" "coord_ring_lbrdiv_conf_1_0_serious" "{lbrdiv-conf:{path:val_teammates/overcooked-v1/coord_ring/lbrdiv/2026-03-02_14-56-18/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:3,idx_list:[[1,0]],test_mode:false}}"
add_job "coord_ring" "coord_ring.br_for_lbrdiv_conf_1_1" "coord_ring_lbrdiv_conf_1_1_serious" "{lbrdiv-conf:{path:val_teammates/overcooked-v1/coord_ring/lbrdiv/2026-03-02_14-56-18/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:3,idx_list:[[1,1]],test_mode:false}}"
add_job "coord_ring" "coord_ring.br_for_lbrdiv_conf_1_2" "coord_ring_lbrdiv_conf_1_2_serious" "{lbrdiv-conf:{path:val_teammates/overcooked-v1/coord_ring/lbrdiv/2026-03-02_14-56-18/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:3,idx_list:[[1,2]],test_mode:false}}"
add_job "coord_ring" "coord_ring.br_for_comedi_1_0" "coord_ring_comedi_1_0_serious" "{comedi:{path:val_teammates/overcooked-v1/coord_ring/comedi/2026-03-01_22-32-32/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:10,idx_list:[[1,0]],test_mode:false}}"
add_job "coord_ring" "coord_ring.br_for_comedi_1_1" "coord_ring_comedi_1_1_serious" "{comedi:{path:val_teammates/overcooked-v1/coord_ring/comedi/2026-03-01_22-32-32/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:10,idx_list:[[1,1]],test_mode:false}}"
add_job "coord_ring" "coord_ring.br_for_comedi_1_2" "coord_ring_comedi_1_2_serious" "{comedi:{path:val_teammates/overcooked-v1/coord_ring/comedi/2026-03-01_22-32-32/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:10,idx_list:[[1,2]],test_mode:false}}"
add_job "coord_ring" "coord_ring.br_for_comedi_1_3" "coord_ring_comedi_1_3_serious" "{comedi:{path:val_teammates/overcooked-v1/coord_ring/comedi/2026-03-01_22-32-32/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:10,idx_list:[[1,3]],test_mode:false}}"
add_job "coord_ring" "coord_ring.br_for_comedi_1_4" "coord_ring_comedi_1_4_serious" "{comedi:{path:val_teammates/overcooked-v1/coord_ring/comedi/2026-03-01_22-32-32/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:10,idx_list:[[1,4]],test_mode:false}}"

# forced_coord extra teammates (8 jobs)
add_job "forced_coord" "forced_coord.br_for_lbrdiv_conf_1_0" "forced_coord_lbrdiv_conf_1_0_serious" "{lbrdiv-conf:{path:val_teammates/overcooked-v1/forced_coord/lbrdiv/2026-03-03_19-30-49/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:3,idx_list:[[1,0]],test_mode:false}}"
add_job "forced_coord" "forced_coord.br_for_lbrdiv_conf_1_1" "forced_coord_lbrdiv_conf_1_1_serious" "{lbrdiv-conf:{path:val_teammates/overcooked-v1/forced_coord/lbrdiv/2026-03-03_19-30-49/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:3,idx_list:[[1,1]],test_mode:false}}"
add_job "forced_coord" "forced_coord.br_for_lbrdiv_conf_1_2" "forced_coord_lbrdiv_conf_1_2_serious" "{lbrdiv-conf:{path:val_teammates/overcooked-v1/forced_coord/lbrdiv/2026-03-03_19-30-49/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:3,idx_list:[[1,2]],test_mode:false}}"
add_job "forced_coord" "forced_coord.br_for_comedi_1_0" "forced_coord_comedi_1_0_serious" "{comedi:{path:val_teammates/overcooked-v1/forced_coord/comedi/2026-03-04_00-28-31/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:10,idx_list:[[1,0]],test_mode:false}}"
add_job "forced_coord" "forced_coord.br_for_comedi_1_1" "forced_coord_comedi_1_1_serious" "{comedi:{path:val_teammates/overcooked-v1/forced_coord/comedi/2026-03-04_00-28-31/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:10,idx_list:[[1,1]],test_mode:false}}"
add_job "forced_coord" "forced_coord.br_for_comedi_1_2" "forced_coord_comedi_1_2_serious" "{comedi:{path:val_teammates/overcooked-v1/forced_coord/comedi/2026-03-04_00-28-31/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:10,idx_list:[[1,2]],test_mode:false}}"
add_job "forced_coord" "forced_coord.br_for_comedi_1_3" "forced_coord_comedi_1_3_serious" "{comedi:{path:val_teammates/overcooked-v1/forced_coord/comedi/2026-03-04_00-28-31/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:10,idx_list:[[1,3]],test_mode:false}}"
add_job "forced_coord" "forced_coord.br_for_comedi_1_4" "forced_coord_comedi_1_4_serious" "{comedi:{path:val_teammates/overcooked-v1/forced_coord/comedi/2026-03-04_00-28-31/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:10,idx_list:[[1,4]],test_mode:false}}"

run_all_jobs
