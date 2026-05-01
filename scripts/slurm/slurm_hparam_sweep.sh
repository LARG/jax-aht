#!/bin/bash
#SBATCH -J wandb_sweep
#SBATCH -o results/slurm_logs/wandb_sweep_%j.out
#SBATCH -e results/slurm_logs/wandb_sweep_%j.err
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -t 48:00:00
#SBATCH -A IRI26004

bash $SCRATCH/jax-aht/scripts/run_hparam_sweep.sh "$SWEEP_SHORT_ID"
sleep infinity # keeps node alive so you can ssh in after job ends.
