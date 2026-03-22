#!/bin/bash
#SBATCH -J creppo_v34_3agent_spr
#SBATCH -o creppo_v34_3agent_spr_%j.out
#SBATCH -e creppo_v34_3agent_spr_%j.err
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=112
#SBATCH -t 24:00:00
#SBATCH --mail-user=jeffreychen287@gmail.com
#SBATCH --mail-type=end,fail

# ==============================================================
# v34: Phase B — N=3 agent coop_recon (ego-centric, 100M steps, 1.5 grid)
# Key changes from v32:
#   - Grid expanded back to 1.5 (v31 style) to defeat geometric pincer maneuvers
#   - Kept 100M SAP steps (v32 style) to ensure SAP solves the large 1.5 grid
#   - eps=0.001 (extreme theoretical bound to compare against v33)
# ==============================================================

module load python
source /scratch/11079/jeffreychen287/jax-aht/venv/bin/activate
cd /scratch/11079/jeffreychen287/jax-aht
export PYTHONPATH=$PYTHONPATH:$PWD
export XLA_PYTHON_CLIENT_PREALLOCATE=false

/scratch/11079/jeffreychen287/jax-aht/venv/bin/python social_laws/run.py \
    task=continuous/coop_recon_3_agent_v33 \
    algorithm=creppo/continuous/coop_recon \
    logger.mode=offline \
    NUM_EXPT_AGENTS=3 \
    algorithm.TOTAL_TIMESTEPS=100000000 \
    algorithm.TOTAL_TIMESTEPS_DECAY=80000000 \
    algorithm.FIXED_EVAL=true \
    algorithm.SAP_DOMAIN_RANDOMIZE_PARTNER=true \
    algorithm.NUM_EVAL_EPISODES=50 \
    algorithm.EPSILON_OPTIMAL=0.001 \
    task.ENV_KWARGS.ego_centric_obs=true \
    label=jax_coop_recon_v34_3agent_0.001_eps

echo "v34 training complete"
