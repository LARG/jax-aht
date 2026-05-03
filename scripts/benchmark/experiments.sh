#!/bin/bash
# Unified benchmark experiment runner.
#
# Validates that all algorithm configs match expected timestep targets before
# running anything. Cancels with a diagnostic if any configs are out of date.
#
# Usage:
#   bash scripts/benchmark/experiments.sh
#
# Configure the algorithms, tasks, and label in the section below.

# === Configuration ===
algos=("brdiv" "lbrdiv" "comedi")
label="neurips:benchmark"
num_seeds=5
num_checkpoints=1   # used for all algorithms except FCP (see below)

# Target total training timestep budgets per difficulty tier.
# These must match the values used when running update_timesteps.py.
easy_target_teammate="195M"   # teammate_generation easy tasks
hard_target_teammate="390M"   # teammate_generation hard tasks
easy_target_oe="195M"         # open_ended_training easy tasks
hard_target_oe="390M"         # open_ended_training hard tasks
easy_target_ego="30M"         # ego_agent_training easy tasks
hard_target_ego="60M"         # ego_agent_training hard tasks

# Algorithms excluded from the timestep sanity check (not part of the benchmark).
skip_algos_oe="open_ended_minimax paired"
skip_algos_ego="ppo_br"
skip_algos_teammate=""

tasks=(
    # "lbf/lbf_7x7_nolevels"
    "lbf/lbf_12x12"
    # "overcooked-v1/coord_ring"
    # "overcooked-v1/asymm_advantages"
    # "overcooked-v1/counter_circuit"
    # "overcooked-v1/cramped_room"
    # "overcooked-v1/forced_coord"
)

# === Algorithm → entry point mapping ===
get_entry_point() {
    case "$1" in
        fcp|brdiv|lbrdiv|comedi)   echo "teammate_generation" ;;
        rotate|cole|trajedi)        echo "open_ended_training" ;;
        ppo_ego|liam|meliba)        echo "ego_agent_training" ;;
        *)                          echo "" ;;
    esac
}

# === Logging ===
log_dir="results/benchmark_logs/${label}"
mkdir -p "${log_dir}"
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="${log_dir}/experiment_${timestamp}.log"

log() {
    local ts
    ts=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[${ts}] $1" | tee -a "${log_file}"
}

# ============================================================
# Step 1: Timestep sanity check (all entry points)
# ============================================================

echo ""
echo "========================================================"
echo "Step 1: Validating timestep configs (all entry points)"
echo "========================================================"
echo ""

check_output=$(
    # shellcheck disable=SC2086  # intentional word-splitting for skip lists
    PYTHONPATH=. python scripts/manage_configs/update_timesteps.py \
        open_ended_training/ \
        --easy-target "${easy_target_oe}" --hard-target "${hard_target_oe}" \
        ${skip_algos_oe:+--skip-algos $skip_algos_oe} \
        --dry-run 2>&1
    echo "---"
    PYTHONPATH=. python scripts/manage_configs/update_timesteps.py \
        ego_agent_training/ \
        --easy-target "${easy_target_ego}" --hard-target "${hard_target_ego}" \
        ${skip_algos_ego:+--skip-algos $skip_algos_ego} \
        --dry-run 2>&1
    echo "---"
    PYTHONPATH=. python scripts/manage_configs/update_timesteps.py \
        teammate_generation/ \
        --easy-target "${easy_target_teammate}" --hard-target "${hard_target_teammate}" \
        ${skip_algos_teammate:+--skip-algos $skip_algos_teammate} \
        --dry-run 2>&1
)

echo "${check_output}"
echo ""

if echo "${check_output}" | grep -q "DRY RUN"; then
    echo "========================================================"
    echo "ERROR: One or more algorithm configs have out-of-date"
    echo "timesteps. Lines marked 'DRY RUN' above show what would"
    echo "change. Notify colleagues that the benchmark configs"
    echo "are out of date and fix them by running"
    echo "the corresponding update_timesteps.py scripts,",
    echo "then re-run this script and push updated configs."
    echo "========================================================"
    exit 1
fi

echo "All timestep configs are up to date."
echo ""

# ============================================================
# Step 2: Run experiments
# ============================================================

log "========================================================"
log "Step 2: Running experiments"
log "Algorithms: ${algos[*]}"
log "Tasks:      ${tasks[*]}"
log "Log file:   ${log_file}"
log "========================================================"

success_count=0
failure_count=0

for algo in "${algos[@]}"; do
    entry_point=$(get_entry_point "${algo}")
    if [ -z "${entry_point}" ]; then
        log "ERROR: unknown algorithm '${algo}' — skipping."
        ((failure_count++))
        continue
    fi

    for task in "${tasks[@]}"; do
        log "Starting: ${algo}/${task}  [${entry_point}]"

        # NUM_CHECKPOINTS is a core hyperparameter for FCP (controls how many
        # partner checkpoints are collected), so don't override it there.
        checkpoint_arg=""
        if [ "${algo}" != "fcp" ]; then
            checkpoint_arg="algorithm.NUM_CHECKPOINTS=${num_checkpoints}"
        fi

        if XLA_FLAGS=--xla_disable_hlo_passes=fusion XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=. python "${entry_point}/run.py" \
            algorithm="${algo}/${task}" \
            task="${task}" \
            label="${label}" \
            algorithm.NUM_SEEDS="${num_seeds}" \
            ${checkpoint_arg} \
            logger.mode="online" \
            2>> "${log_file}"; then
            log "✅ Completed: ${algo}/${task}"
            ((success_count++))
        else
            log "❌ Failed:    ${algo}/${task}"
            ((failure_count++))
        fi
    done
done

# ============================================================
# Summary
# ============================================================

echo ""
log "========================================================"
log "Summary"
log "  Attempted:  $((success_count + failure_count))"
log "  Successful: ${success_count}"
log "  Failed:     ${failure_count}"
log "  Log file:   ${log_file}"
log "========================================================"

if [ "${failure_count}" -gt 0 ]; then
    exit 1
fi
