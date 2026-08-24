#!/bin/bash
#SBATCH -J multi_gpu_job
#SBATCH -o multi_gpu_job.out
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -t 48:00:00
#SBATCH -A IRI26004

conda activate bench311
cd $SCRATCH/jax-aht

sleep infinity
