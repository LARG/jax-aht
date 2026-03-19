#!/bin/bash
#SBATCH -J creppo_v29_3agent
#SBATCH -o creppo_v29_3agent_%j.out
#SBATCH -e creppo_v29_3agent_%j.err
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=112
#SBATCH -t 10:00:00
#SBATCH --mail-user=jeffreychen287@gmail.com
#SBATCH --mail-type=end,fail

# ==============================================================
# v29: Phase B — N=3 agent coop_recon (ego-centric, 40M steps)
# Grid: 1x1, Horizon: 80
# Baseline config from v28 (ego-centric wins in Phase A)
# ==============================================================

module load python
source /scratch/11079/jeffreychen287/jax-aht/venv/bin/activate
cd /scratch/11079/jeffreychen287/jax-aht
export PYTHONPATH=$PYTHONPATH:$PWD
export XLA_PYTHON_CLIENT_PREALLOCATE=false

/scratch/11079/jeffreychen287/jax-aht/venv/bin/python social_laws/run.py \
    task=continuous/coop_recon_3_agent \
    algorithm=creppo/continuous/coop_recon \
    logger.mode=offline \
    NUM_EXPT_AGENTS=3 \
    algorithm.TOTAL_TIMESTEPS=40000000 \
    algorithm.FIXED_EVAL=true \
    algorithm.SAP_DOMAIN_RANDOMIZE_PARTNER=true \
    algorithm.NUM_EVAL_EPISODES=50 \
    task.ENV_KWARGS.ego_centric_obs=true \
    label=jax_coop_recon_v29_3agent_ego_creppo

echo "v29 training complete"
