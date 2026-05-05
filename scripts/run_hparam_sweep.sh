#!/bin/bash
# usage: bash scripts/run_hparam_sweep.sh <sweep_id>
# example: bash scripts/run_hparam_sweep.sh dqsezvy1

if [ -z "$1" ]; then
    echo "Usage: $0 <sweep_id>"
    echo "Example: $0 dqsezvy1"
    exit 1
fi
SWEEP_ID="aht-project/aht-parameter-sweep/$1"

conda activate bench311
cd $SCRATCH/jax-aht

# Detect GPUs available on this node and select ones with >50GB free VRAM
MIN_FREE_MB=$((50 * 1024))
mapfile -t FREE_MEM < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
echo "Found ${#FREE_MEM[@]} GPUs on node $(hostname); requiring >${MIN_FREE_MB} MiB free"

ELIGIBLE_GPUS=()
for i in "${!FREE_MEM[@]}"; do
    free_mb=$(echo "${FREE_MEM[$i]}" | tr -d '[:space:]')
    if [ "$free_mb" -gt "$MIN_FREE_MB" ]; then
        ELIGIBLE_GPUS+=("$i")
        echo "  GPU $i: ${free_mb} MiB free -> eligible"
    else
        echo "  GPU $i: ${free_mb} MiB free -> skipped"
    fi
done

if [ "${#ELIGIBLE_GPUS[@]}" -eq 0 ]; then
    echo "No GPUs with >${MIN_FREE_MB} MiB free VRAM. Exiting."
    exit 1
fi

echo "Launching ${#ELIGIBLE_GPUS[@]} agent(s) on GPUs: ${ELIGIBLE_GPUS[*]}"

# Launch one agent per eligible GPU
# ORBAX RACE FIX: cap to 1 agent/node to avoid shared-savedir collisions.
# Original line: for i in "${ELIGIBLE_GPUS[@]}"; do
for i in "${ELIGIBLE_GPUS[@]:0:1}"; do
    # Original line: CUDA_VISIBLE_DEVICES=$i PYTHONPATH=. wandb agent "$SWEEP_ID" --count 30 &
    CUDA_VISIBLE_DEVICES=$i PYTHONPATH=. wandb agent "$SWEEP_ID" --count 200 &
done

wait
