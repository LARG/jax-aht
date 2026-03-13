#!/bin/bash
#SBATCH -J jax_coop_recon_creppo_v23
#SBATCH -o jax_coop_recon_creppo_v23_%j.out
#SBATCH -e jax_coop_recon_creppo_v23_%j.err
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

# v23 ablation: NO step penalty (step_penalty=0.0)
# Hypothesis: without negative step returns on collision episodes,
# does alpha improve? Addresses team member concern about negative returns.
# Collision dead-end is still active.
/scratch/11079/jeffreychen287/jax-aht/venv/bin/python social_laws/run.py \
    task=continuous/coop_recon \
    algorithm=creppo/continuous/coop_recon \
    logger.mode=offline \
    algorithm.TOTAL_TIMESTEPS=20000000 \
    algorithm.FIXED_EVAL=true \
    algorithm.SAP_DOMAIN_RANDOMIZE_PARTNER=true \
    'task.ENV_KWARGS.step_penalty=0.0' \
    label=jax_coop_recon_v23_phaseA_nostep_creppo

echo "Training complete."
