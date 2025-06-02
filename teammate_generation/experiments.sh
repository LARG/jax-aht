#!/bin/bash

# Algorithm to run
algo="comedi"
label="jax-aht:test"
num_seeds=3

# Create log directory if it doesn't exist
mkdir -p results/teammate_generation_logs/${algo}/${label}

# Get current timestamp for log file
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="results/teammate_generation_logs/${algo}/${label}/experiment_${timestamp}.log"

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
    
    if python teammate_generation/run.py algorithm="${algo}/${task}" task="${task}" label="${label}" algorithm.NUM_SEEDS="${num_seeds}" 2>> "${log_file}"; then
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

