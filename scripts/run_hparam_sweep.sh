#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <sweep_id>"
    echo "Example: $0 dqsezvy1"
    exit 1
fi
SWEEP_ID="aht-project/aht-parameter-sweep/$1"

conda activate bench311
cd $SCRATCH/jax-aht

# Detect GPUs available on this node
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Found $NUM_GPUS GPUs on node $(hostname)"

# Launch one agent per GPU
for i in $(seq 0 $((NUM_GPUS - 1))); do
    CUDA_VISIBLE_DEVICES=$i PYTHONPATH=. wandb agent "$SWEEP_ID" --count 30 &
done

wait
