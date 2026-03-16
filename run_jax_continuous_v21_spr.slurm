#!/bin/bash
#SBATCH -J jax_coop_recon_creppo_v21
#SBATCH -o jax_coop_recon_creppo_v21_%j.out
#SBATCH -e jax_coop_recon_creppo_v21_%j.err
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=112
#SBATCH -t 06:00:00
#SBATCH --mail-user=jeffreychen287@gmail.com
#SBATCH --mail-type=all

module load python
source /scratch/11079/jeffreychen287/jax-aht/venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# v21 ablation: larger collision radius (0.05 -> 0.10)
# Hypothesis: bigger detection zone gives agents earlier warning -> easier avoidance
/scratch/11079/jeffreychen287/jax-aht/venv/bin/python social_laws/run.py \
    task=continuous/coop_recon \
    algorithm=creppo/continuous/coop_recon \
    logger.mode=offline \
    algorithm.TOTAL_TIMESTEPS=20000000 \
    algorithm.FIXED_EVAL=true \
    algorithm.SAP_DOMAIN_RANDOMIZE_PARTNER=true \
    'task.ENV_KWARGS.collision_radius=0.10' \
    label=jax_coop_recon_v21_phaseA_cr10_creppo

echo "Training complete."
