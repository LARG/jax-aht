#!/usr/bin/env bash
set -euo pipefail

# Runs Overcooked BR jobs, supports parallel GPU workers, and records checkpoints persistently.

# GPU_LIST=1,2 PARALLEL_JOBS=2 TOTAL_TIMESTEPS=10000000 bash scripts/run_overcooked_br_jobs.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GPU_LIST="${GPU_LIST:-1,2}"
PARALLEL_JOBS="${PARALLEL_JOBS:-0}"
# Training budget per job. Override with TOTAL_TIMESTEPS=... at launch.
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-10000000}"

RUN_LOG="overcooked_br_runs.log"
CHECKPOINT_LOG_PREFIX="overcooked_br_checkpoints"

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

  if already_logged "$job_key"; then
    log_msg "SKIP $job_key (already present in ${CHECKPOINT_LOG_PREFIX}_*.txt)"
    return
  fi

  log_msg "START $job_key on GPU $gpu_id"

  CUDA_VISIBLE_DEVICES="$gpu_id" python3 ego_agent_training/run.py \
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

  log_msg "Overcooked BR batch started"
  log_msg "GPU_LIST=$GPU_LIST PARALLEL_JOBS=$max_parallel"
  log_msg "TOTAL_TIMESTEPS=$TOTAL_TIMESTEPS"
  log_msg "Checkpoint logs: ${CHECKPOINT_LOG_PREFIX}_gpu<id>.txt"

  local running=0
  local idx=0
  local spec layout job_key label partner_override gpu_id

  for spec in "${JOBS[@]}"; do
    while [[ "$running" -ge "$max_parallel" ]]; do
      wait -n
      running=$((running - 1))
    done

    IFS='|' read -r layout job_key label partner_override <<< "$spec"
    gpu_id="${GPU_IDS[$((idx % num_gpus))]}"

    run_job "$layout" "$job_key" "$label" "$partner_override" "$gpu_id" &
    running=$((running + 1))
    idx=$((idx + 1))
  done

  while [[ "$running" -gt 0 ]]; do
    wait -n
    running=$((running - 1))
  done

  log_msg "Overcooked BR batch completed"
}

# cramped_room (9 jobs)
add_job "cramped_room" "cramped_room.br_for_ippo_mlp_0" "cramped_room_ippo_mlp_0_serious" "{ippo_mlp:{path:eval_teammates/overcooked-v1/cramped_room/ippo/2025-04-21_22-53-04/saved_train_run,actor_type:mlp,ckpt_key:final_params,idx_list:[0],test_mode:true}}"
add_job "cramped_room" "cramped_room.br_for_ippo_mlp_1" "cramped_room_ippo_mlp_1_serious" "{ippo_mlp:{path:eval_teammates/overcooked-v1/cramped_room/ippo/2025-04-21_22-53-04/saved_train_run,actor_type:mlp,ckpt_key:final_params,idx_list:[1],test_mode:true}}"
add_job "cramped_room" "cramped_room.br_for_ippo_mlp_2" "cramped_room_ippo_mlp_2_serious" "{ippo_mlp:{path:eval_teammates/overcooked-v1/cramped_room/ippo/2025-04-21_22-53-04/saved_train_run,actor_type:mlp,ckpt_key:final_params,idx_list:[2],test_mode:true}}"
add_job "cramped_room" "cramped_room.br_for_brdiv_conf_0" "cramped_room_brdiv_conf_0_serious" "{brdiv-conf:{path:eval_teammates/overcooked-v1/cramped_room/brdiv/2025-04-23-12-55-47/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:4,idx_list:[0]}}"
add_job "cramped_room" "cramped_room.br_for_brdiv_conf_1" "cramped_room_brdiv_conf_1_serious" "{brdiv-conf:{path:eval_teammates/overcooked-v1/cramped_room/brdiv/2025-04-23-12-55-47/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:4,idx_list:[1]}}"
add_job "cramped_room" "cramped_room.br_for_independent_agent_0_4" "cramped_room_independent_agent_0_4_serious" "{independent_agent_0.4:{actor_type:independent_agent,p_onion_on_counter:0.4,p_plate_on_counter:0.4}}"
add_job "cramped_room" "cramped_room.br_for_independent_agent_0" "cramped_room_independent_agent_0_serious" "{independent_agent_0:{actor_type:independent_agent,p_onion_on_counter:0.0,p_plate_on_counter:0.0}}"
add_job "cramped_room" "cramped_room.br_for_onion_agent_0_1" "cramped_room_onion_agent_0_1_serious" "{onion_agent_0.1:{actor_type:onion_agent,p_onion_on_counter:0.1}}"
add_job "cramped_room" "cramped_room.br_for_plate_agent_0_1" "cramped_room_plate_agent_0_1_serious" "{plate_agent_0.1:{actor_type:plate_agent,p_plate_on_counter:0.1}}"

# asymm_advantages (9 jobs)
add_job "asymm_advantages" "asymm_advantages.br_for_ippo_mlp_0" "asymm_advantages_ippo_mlp_0_serious" "{ippo_mlp:{path:eval_teammates/overcooked-v1/asymm_advantages/ippo/2025-04-21_22-53-56/saved_train_run,actor_type:mlp,ckpt_key:final_params,idx_list:[0],test_mode:true}}"
add_job "asymm_advantages" "asymm_advantages.br_for_ippo_mlp_1" "asymm_advantages_ippo_mlp_1_serious" "{ippo_mlp:{path:eval_teammates/overcooked-v1/asymm_advantages/ippo/2025-04-21_22-53-56/saved_train_run,actor_type:mlp,ckpt_key:final_params,idx_list:[1],test_mode:true}}"
add_job "asymm_advantages" "asymm_advantages.br_for_ippo_mlp_2" "asymm_advantages_ippo_mlp_2_serious" "{ippo_mlp:{path:eval_teammates/overcooked-v1/asymm_advantages/ippo/2025-04-21_22-53-56/saved_train_run,actor_type:mlp,ckpt_key:final_params,idx_list:[2],test_mode:true}}"
add_job "asymm_advantages" "asymm_advantages.br_for_brdiv_conf_0" "asymm_advantages_brdiv_conf_0_serious" "{brdiv-conf:{path:eval_teammates/overcooked-v1/asymm_advantages/brdiv/2025-04-23/13-16-34/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:4,idx_list:[0]}}"
add_job "asymm_advantages" "asymm_advantages.br_for_brdiv_conf_1" "asymm_advantages_brdiv_conf_1_serious" "{brdiv-conf:{path:eval_teammates/overcooked-v1/asymm_advantages/brdiv/2025-04-23/13-16-34/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:4,idx_list:[1]}}"
add_job "asymm_advantages" "asymm_advantages.br_for_brdiv_conf_2" "asymm_advantages_brdiv_conf_2_serious" "{brdiv-conf:{path:eval_teammates/overcooked-v1/asymm_advantages/brdiv/2025-04-23/13-16-34/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:4,idx_list:[2]}}"
add_job "asymm_advantages" "asymm_advantages.br_for_independent_agent_0" "asymm_advantages_independent_agent_0_serious" "{independent_agent_0:{actor_type:independent_agent,p_onion_on_counter:0.0,p_plate_on_counter:0.0}}"
add_job "asymm_advantages" "asymm_advantages.br_for_onion_agent_0" "asymm_advantages_onion_agent_0_serious" "{onion_agent_0:{actor_type:onion_agent,p_onion_on_counter:0.0}}"
add_job "asymm_advantages" "asymm_advantages.br_for_plate_agent_0" "asymm_advantages_plate_agent_0_serious" "{plate_agent_0:{actor_type:plate_agent,p_plate_on_counter:0.0}}"

# counter_circuit (11 jobs)
add_job "counter_circuit" "counter_circuit.br_for_ippo_mlp_cc_0" "counter_circuit_ippo_mlp_cc_0_serious" "{ippo_mlp_cc:{path:eval_teammates/overcooked-v1/counter_circuit/ippo/2025-04-21_22-55-42/saved_train_run,actor_type:mlp,ckpt_key:final_params,idx_list:[0],test_mode:true}}"
add_job "counter_circuit" "counter_circuit.br_for_ippo_mlp_cc_1" "counter_circuit_ippo_mlp_cc_1_serious" "{ippo_mlp_cc:{path:eval_teammates/overcooked-v1/counter_circuit/ippo/2025-04-21_22-55-42/saved_train_run,actor_type:mlp,ckpt_key:final_params,idx_list:[1],test_mode:true}}"
add_job "counter_circuit" "counter_circuit.br_for_ippo_mlp_cc_2" "counter_circuit_ippo_mlp_cc_2_serious" "{ippo_mlp_cc:{path:eval_teammates/overcooked-v1/counter_circuit/ippo/2025-04-21_22-55-42/saved_train_run,actor_type:mlp,ckpt_key:final_params,idx_list:[2],test_mode:true}}"
add_job "counter_circuit" "counter_circuit.br_for_ippo_mlp_pass_0" "counter_circuit_ippo_mlp_pass_0_serious" "{ippo_mlp_pass:{path:eval_teammates/overcooked-v1/counter_circuit/ippo/2025-04-23_15-28-23/saved_train_run,actor_type:mlp,ckpt_key:final_params,idx_list:[0],test_mode:true}}"
add_job "counter_circuit" "counter_circuit.br_for_ippo_mlp_pass_1" "counter_circuit_ippo_mlp_pass_1_serious" "{ippo_mlp_pass:{path:eval_teammates/overcooked-v1/counter_circuit/ippo/2025-04-23_15-28-23/saved_train_run,actor_type:mlp,ckpt_key:final_params,idx_list:[1],test_mode:true}}"
add_job "counter_circuit" "counter_circuit.br_for_ippo_mlp_pass_2" "counter_circuit_ippo_mlp_pass_2_serious" "{ippo_mlp_pass:{path:eval_teammates/overcooked-v1/counter_circuit/ippo/2025-04-23_15-28-23/saved_train_run,actor_type:mlp,ckpt_key:final_params,idx_list:[2],test_mode:true}}"
add_job "counter_circuit" "counter_circuit.br_for_independent_agent_0" "counter_circuit_independent_agent_0_serious" "{independent_agent_0:{actor_type:independent_agent,p_onion_on_counter:0.0,p_plate_on_counter:0.0}}"
add_job "counter_circuit" "counter_circuit.br_for_onion_agent_0" "counter_circuit_onion_agent_0_serious" "{onion_agent_0:{actor_type:onion_agent,p_onion_on_counter:0.0}}"
add_job "counter_circuit" "counter_circuit.br_for_plate_agent_0" "counter_circuit_plate_agent_0_serious" "{plate_agent_0:{actor_type:plate_agent,p_plate_on_counter:0.0}}"
add_job "counter_circuit" "counter_circuit.br_for_onion_agent_0_9" "counter_circuit_onion_agent_0_9_serious" "{onion_agent_0.9:{actor_type:onion_agent,p_onion_on_counter:0.9}}"
add_job "counter_circuit" "counter_circuit.br_for_plate_agent_0_9" "counter_circuit_plate_agent_0_9_serious" "{plate_agent_0.9:{actor_type:plate_agent,p_plate_on_counter:0.9}}"

# coord_ring (9 jobs)
add_job "coord_ring" "coord_ring.br_for_ippo_mlp_1" "coord_ring_ippo_mlp_1_serious" "{ippo_mlp:{path:eval_teammates/overcooked-v1/coord_ring/ippo/2025-04-21_22-58-26/saved_train_run,actor_type:mlp,ckpt_key:final_params,idx_list:[1],test_mode:true}}"
add_job "coord_ring" "coord_ring.br_for_ippo_mlp_2" "coord_ring_ippo_mlp_2_serious" "{ippo_mlp:{path:eval_teammates/overcooked-v1/coord_ring/ippo/2025-04-21_22-58-26/saved_train_run,actor_type:mlp,ckpt_key:final_params,idx_list:[2],test_mode:true}}"
add_job "coord_ring" "coord_ring.br_for_ippo_mlp_3" "coord_ring_ippo_mlp_3_serious" "{ippo_mlp:{path:eval_teammates/overcooked-v1/coord_ring/ippo/2025-04-21_22-58-26/saved_train_run,actor_type:mlp,ckpt_key:final_params,idx_list:[3],test_mode:true}}"
add_job "coord_ring" "coord_ring.br_for_brdiv_conf1_1" "coord_ring_brdiv_conf1_1_serious" "{brdiv-conf1:{path:eval_teammates/overcooked-v1/coord_ring/brdiv/2025_04-23_15-00-27/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:3,idx_list:[1]}}"
add_job "coord_ring" "coord_ring.br_for_brdiv_conf1_2" "coord_ring_brdiv_conf1_2_serious" "{brdiv-conf1:{path:eval_teammates/overcooked-v1/coord_ring/brdiv/2025_04-23_15-00-27/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:3,idx_list:[2]}}"
add_job "coord_ring" "coord_ring.br_for_brdiv_conf2_0" "coord_ring_brdiv_conf2_0_serious" "{brdiv-conf2:{path:eval_teammates/overcooked-v1/coord_ring/brdiv/2025_04-23_14-17-36/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:3,idx_list:[0]}}"
add_job "coord_ring" "coord_ring.br_for_independent_agent_0" "coord_ring_independent_agent_0_serious" "{independent_agent_0:{actor_type:independent_agent,p_onion_on_counter:0.0,p_plate_on_counter:0.0}}"
add_job "coord_ring" "coord_ring.br_for_onion_agent_0" "coord_ring_onion_agent_0_serious" "{onion_agent_0:{actor_type:onion_agent,p_onion_on_counter:0.0}}"
add_job "coord_ring" "coord_ring.br_for_plate_agent_0" "coord_ring_plate_agent_0_serious" "{plate_agent_0:{actor_type:plate_agent,p_plate_on_counter:0.0}}"

# forced_coord (9 jobs)
add_job "forced_coord" "forced_coord.br_for_ippo_mlp_0" "forced_coord_ippo_mlp_0_serious" "{ippo_mlp:{path:eval_teammates/overcooked-v1/forced_coord/ippo/2025-04-21_23-00-17/saved_train_run,actor_type:mlp,ckpt_key:final_params,idx_list:[0],test_mode:true}}"
add_job "forced_coord" "forced_coord.br_for_ippo_mlp_1" "forced_coord_ippo_mlp_1_serious" "{ippo_mlp:{path:eval_teammates/overcooked-v1/forced_coord/ippo/2025-04-21_23-00-17/saved_train_run,actor_type:mlp,ckpt_key:final_params,idx_list:[1],test_mode:true}}"
add_job "forced_coord" "forced_coord.br_for_ippo_mlp_2" "forced_coord_ippo_mlp_2_serious" "{ippo_mlp:{path:eval_teammates/overcooked-v1/forced_coord/ippo/2025-04-21_23-00-17/saved_train_run,actor_type:mlp,ckpt_key:final_params,idx_list:[2],test_mode:true}}"
add_job "forced_coord" "forced_coord.br_for_brdiv_conf1_0" "forced_coord_brdiv_conf1_0_serious" "{brdiv-conf1:{path:eval_teammates/overcooked-v1/forced_coord/brdiv/2025-04-23_19-44-30/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:3,idx_list:[0]}}"
add_job "forced_coord" "forced_coord.br_for_brdiv_conf1_2" "forced_coord_brdiv_conf1_2_serious" "{brdiv-conf1:{path:eval_teammates/overcooked-v1/forced_coord/brdiv/2025-04-23_19-44-30/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:3,idx_list:[2]}}"
add_job "forced_coord" "forced_coord.br_for_brdiv_conf2_1" "forced_coord_brdiv_conf2_1_serious" "{brdiv-conf2:{path:eval_teammates/overcooked-v1/forced_coord/brdiv/2025-04-23_20-25-28/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:3,idx_list:[1]}}"
add_job "forced_coord" "forced_coord.br_for_brdiv_conf3_0" "forced_coord_brdiv_conf3_0_serious" "{brdiv-conf3:{path:eval_teammates/overcooked-v1/forced_coord/brdiv/2025-04-23_21-06-05/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:3,idx_list:[0]}}"
add_job "forced_coord" "forced_coord.br_for_brdiv_conf3_2" "forced_coord_brdiv_conf3_2_serious" "{brdiv-conf3:{path:eval_teammates/overcooked-v1/forced_coord/brdiv/2025-04-23_21-06-05/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,POP_SIZE:3,idx_list:[2]}}"
add_job "forced_coord" "forced_coord.br_for_independent_agent_0_6" "forced_coord_independent_agent_0_6_serious" "{independent_agent_0.6:{actor_type:independent_agent,p_onion_on_counter:0.6,p_plate_on_counter:0.6}}"

run_all_jobs
