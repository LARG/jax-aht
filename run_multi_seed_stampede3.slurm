#!/bin/bash
#----------------------------------------------------
# Multi-Seed Scaling Evaluation — amd-rtx (RTX PRO 6000 Blackwell)
#
# Launches IPPO, PPO, and CREPPO multi-seed arrays.
# Each array batches 4 experiments and cycles through 3 seeds.
# 
# Texas Robotics Priority Queue
#----------------------------------------------------
#SBATCH -J aht_multi_seed
#SBATCH -p amd-rtx
#SBATCH -A IRI26004
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 24:00:00
#SBATCH -o logs/aht_multi_seed_%j.out
#SBATCH -e logs/aht_multi_seed_%j.err
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=jeffreychen287@utexas.edu

mkdir -p logs

module purge
module load gcc/13.2.0 cuda/12.8 python

# cd to the JAX-AHT repository (assumes user cloned to TACC scratch)
cd /scratch/11079/jeffreychen287/jax-aht

# Execute the array scripts sequentially
echo "Starting Full IPPO Multi-Seed Array..."
bash run_marl_compare_ippo.sh

echo "Starting Full PPO Multi-Seed Array..."
bash run_marl_compare_ppo.sh

echo "Starting Full CREPPO Multi-Seed Array..."
bash run_marl_compare_creppo.sh

echo "All Multi-Seed Scaling Evaluations Complete!"
