#!/bin/bash
#SBATCH -J creppo_v31_3agent_h100
#SBATCH -o creppo_v31_3agent_h100_%j.out
#SBATCH -e creppo_v31_3agent_h100_%j.err
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH -t 4:00:00
#SBATCH --mail-user=jeffreychen287@gmail.com
#SBATCH --mail-type=end,fail

# ==============================================================
# v31: Phase B — N=3 agent coop_recon (ego-centric, 40M steps)
# Fixes applied: grid 1.5x1.5, min_sep_goal 0.5, NOOP velocity reset 
# ==============================================================

module load python
module load cuda/12.3
source /scratch/11079/jeffreychen287/jax-aht/venv/bin/activate
cd /scratch/11079/jeffreychen287/jax-aht
export PYTHONPATH=$PYTHONPATH:$PWD
export XLA_PYTHON_CLIENT_PREALLOCATE=false

/scratch/11079/jeffreychen287/jax-aht/venv/bin/python social_laws/run.py \
    task=continuous/coop_recon_3_agent_v31 \
    algorithm=creppo/continuous/coop_recon \
    logger.mode=offline \
    NUM_EXPT_AGENTS=3 \
    algorithm.TOTAL_TIMESTEPS=40000000 \
    algorithm.FIXED_EVAL=true \
    algorithm.SAP_DOMAIN_RANDOMIZE_PARTNER=true \
    algorithm.NUM_EVAL_EPISODES=50 \
    task.ENV_KWARGS.ego_centric_obs=true \
    label=jax_coop_recon_v31_3agent_ego_creppo

echo "v31 training complete"
