#!/bin/bash
#SBATCH -J jax_coop_recon_creppo_v26
#SBATCH -o jax_coop_recon_creppo_v26_%j.out
#SBATCH -e jax_coop_recon_creppo_v26_%j.err
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

# v26 DIAGNOSTIC: Reproduces v20 exactly (40M steps, default min_sep_goal=0.30, 50 eval episodes)
# Purpose: isolate whether weird alpha in v24/v25 is due to min_sep_goal=0.5
# OR due to the team's new joint eval code now running correctly.
# If v26 alpha ≈ 0.997 (like v20), the new code is fine and min_sep_goal=0.5 is causing the issue.
# If v26 alpha is also low/weird, the new joint eval code is affecting results.
/scratch/11079/jeffreychen287/jax-aht/venv/bin/python social_laws/run.py \
    task=continuous/coop_recon \
    algorithm=creppo/continuous/coop_recon \
    logger.mode=offline \
    algorithm.TOTAL_TIMESTEPS=40000000 \
    algorithm.FIXED_EVAL=true \
    algorithm.SAP_DOMAIN_RANDOMIZE_PARTNER=true \
    algorithm.NUM_EVAL_EPISODES=50 \
    label=jax_coop_recon_v26_phaseA_diag_creppo

echo "Training complete."
