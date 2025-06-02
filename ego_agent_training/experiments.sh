#!/bin/bash

# Algorithm to run
algo="ppo_ego"
label="paper-v0:1reg-ego-v-pop"
num_seeds=3

# Declare an associative array for partner paths
declare -A partner_paths
partner_paths=(
    ["overcooked-v1/cramped_room"]="results/overcooked-v1/cramped_room/oe_persistent/paper-v0:1reg/2025-05-10_16-28-55/saved_train_run"
    ["overcooked-v1/asymm_advantages"]="results/overcooked-v1/asymm_advantages/oe_persistent/paper-v0:1reg/2025-05-10_01-14-02/saved_train_run/"
    ["overcooked-v1/counter_circuit"]="results/overcooked-v1/counter_circuit/oe_persistent/paper-v0:1reg/2025-05-10_10-48-53/saved_train_run/"
    ["overcooked-v1/forced_coord"]="results/overcooked-v1/forced_coord/oe_persistent/paper-v0:1reg/2025-05-10_18-22-48/saved_train_run/"
    ["overcooked-v1/coord_ring"]="results/overcooked-v1/coord_ring/oe_persistent/paper-v0:1reg/2025-05-10_05-18-25/saved_train_run/"
    ["lbf"]="results/lbf/oe_persistent/paper-v0:1reg/2025-05-10_19-26-02/saved_train_run/"
)

# Create log directory if it doesn't exist
mkdir -p results/ego_agent_training_logs/${algo}/${label}

# Get current timestamp for log file
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="results/ego_agent_training_logs/${algo}/${label}/experiment_${timestamp}.log"

# Tasks to run
tasks=(
    "overcooked-v1/asymm_advantages"
    "overcooked-v1/coord_ring"
    "overcooked-v1/counter_circuit"
    "overcooked-v1/cramped_room"
    "overcooked-v1/forced_coord"
    "lbf"
)

# Function to log messages
log() {
    local message="$1"
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[${timestamp}] ${message}" | tee -a "${log_file}"
}

# Initialize counters
success_count=0
failure_count=0

# Run experiments for each task
for task in "${tasks[@]}"; do
    log "Starting task: ${algo}/${task}"
    
    # Get the partner path for the current task
    partner_path=${partner_paths[$task]}

    if [ -z "$partner_path" ]; then
        log "❌ Error: No partner path found for task: ${task}"
        ((failure_count++))
        continue # Skip to the next task
    fi
    
    log "Using partner path: ${partner_path}"

    if python ego_agent_training/run.py algorithm="${algo}/${task}" task="${task}" \
        label="${label}" algorithm.NUM_EGO_TRAIN_SEEDS="${num_seeds}" \
        algorithm.partner_agent.path="${partner_path}" 2>> "${log_file}"; then
        log "✅ Successfully completed task: ${algo}/${task}"
        ((success_count++))
    else
        log "❌ Failed to complete task: ${algo}/${task}"
        ((failure_count++))
    fi
done

# Print final summary
log "Experiment Summary:"
log "Total tasks attempted: ${#tasks[@]}"
log "Successful tasks: ${success_count}"
log "Failed tasks: ${failure_count}"

if [ ${failure_count} -gt 0 ]; then
    log "Warning: Some tasks failed. Check the log file for details: ${log_file}"
    exit 1
else
    log "All tasks completed successfully!"
    exit 0
fi

