#!/bin/bash
#SBATCH -J wandb_sweep
#SBATCH -o wandb_sweep_%j.out
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -t 48:00:00
#SBATCH -A IRI26004

# module load nvidia
conda activate bench311
cd $SCRATCH/jax-aht

SWEEP_ID="aht-project/aht-parameter-sweep/dqsezvy1"

# Detect GPUs available on this node
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Found $NUM_GPUS GPUs on node $(hostname)"

# Launch one agent per GPU
for i in $(seq 0 $((NUM_GPUS - 1))); do
    CUDA_VISIBLE_DEVICES=$i PYTHONPATH=. wandb agent "$SWEEP_ID" --count 40 &
done

wait