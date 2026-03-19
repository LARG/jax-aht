#!/bin/bash
#SBATCH -J jax_coop_recon_creppo_v28
#SBATCH -o jax_coop_recon_creppo_v28_%j.out
#SBATCH -e jax_coop_recon_creppo_v28_%j.err
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

# v28: TRUE ego-centric obs.
# v25 had a bug where the agent's goal observations were still absolute [0,1] coordinates, 
# forcing the agent to learn an internal global map from relative positions just to navigate.
# v28 fixes this by making goal observations relative vectors (goal - agent_pos).
# With proper relative goal vectors, navigation is much easier, so 40M steps should be enough.
/scratch/11079/jeffreychen287/jax-aht/venv/bin/python social_laws/run.py \
    task=continuous/coop_recon \
    algorithm=creppo/continuous/coop_recon \
    logger.mode=offline \
    algorithm.TOTAL_TIMESTEPS=40000000 \
    algorithm.FIXED_EVAL=true \
    algorithm.SAP_DOMAIN_RANDOMIZE_PARTNER=true \
    algorithm.NUM_EVAL_EPISODES=50 \
    task.ENV_KWARGS.ego_centric_obs=true \
    label=jax_coop_recon_v28_phaseA_true_ego_creppo

echo "Training complete."
