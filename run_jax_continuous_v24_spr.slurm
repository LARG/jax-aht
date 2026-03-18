#!/bin/bash
#SBATCH -J jax_coop_recon_creppo_v24
#SBATCH -o jax_coop_recon_creppo_v24_%j.out
#SBATCH -e jax_coop_recon_creppo_v24_%j.err
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

# v24 ablation: larger goal separation (min_sep_goal 0.30 -> 0.50)
# Hypothesis: bigger separation forces agents to diverge naturally, improving worst-case alpha.
# Also bumped NUM_EVAL_EPISODES to 50 for more stable alpha estimates.
/scratch/11079/jeffreychen287/jax-aht/venv/bin/python social_laws/run.py \
    task=continuous/coop_recon \
    algorithm=creppo/continuous/coop_recon \
    logger.mode=offline \
    algorithm.TOTAL_TIMESTEPS=40000000 \
    algorithm.FIXED_EVAL=true \
    algorithm.SAP_DOMAIN_RANDOMIZE_PARTNER=true \
    algorithm.NUM_EVAL_EPISODES=50 \
    '+task.ENV_KWARGS.min_sep_goal=0.5' \
    label=jax_coop_recon_v24_phaseA_sep50_creppo

echo "Training complete."
