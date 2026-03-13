#!/bin/bash
#SBATCH -J jax_coop_recon_creppo_v22
#SBATCH -o jax_coop_recon_creppo_v22_%j.out
#SBATCH -e jax_coop_recon_creppo_v22_%j.err
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

# v22 ablation: ego-centric observations (absolute -> relative)
# Hypothesis: relative position obs gives stronger signal for collision avoidance
/scratch/11079/jeffreychen287/jax-aht/venv/bin/python social_laws/run.py \
    task=continuous/coop_recon \
    algorithm=creppo/continuous/coop_recon \
    logger.mode=offline \
    algorithm.TOTAL_TIMESTEPS=20000000 \
    algorithm.FIXED_EVAL=true \
    algorithm.SAP_DOMAIN_RANDOMIZE_PARTNER=true \
    'task.ENV_KWARGS.ego_centric_obs=true' \
    label=jax_coop_recon_v22_phaseA_ego_creppo

echo "Training complete."
