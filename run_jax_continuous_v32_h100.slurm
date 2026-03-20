#!/bin/bash
#SBATCH -J creppo_v32_3agent_h100
#SBATCH -o creppo_v32_3agent_h100_%j.out
#SBATCH -e creppo_v32_3agent_h100_%j.err
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=96
#SBATCH -t 12:00:00
#SBATCH --mail-user=jeffreychen287@gmail.com
#SBATCH --mail-type=end,fail

# ==============================================================
# v32: Phase B — N=3 agent coop_recon (ego-centric, 100M steps)
# Key changes from v31:
#   - Back to 1.0 grid, 0.30 min_sep (v31's 1.5/0.5 hurt Agent 3)
#   - 100M SAP steps (2.5x more) for strong Q-values
#   - EPSILON_OPTIMAL=0.001 (10x tighter than v29/v31's 0.01)
#   - Keeps velocity reset fix from v31
# ==============================================================

module load python
module load cuda/12
source /scratch/11079/jeffreychen287/jax-aht/venv/bin/activate
cd /scratch/11079/jeffreychen287/jax-aht
export PYTHONPATH=$PYTHONPATH:$PWD
export XLA_PYTHON_CLIENT_PREALLOCATE=false

/scratch/11079/jeffreychen287/jax-aht/venv/bin/python social_laws/run.py \
    task=continuous/coop_recon_3_agent_v32 \
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
    label=jax_coop_recon_v32_3agent_ego_creppo

echo "v32 training complete"
