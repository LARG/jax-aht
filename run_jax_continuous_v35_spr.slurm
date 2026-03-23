#!/bin/bash
#SBATCH -J creppo_v35_3agent_spr
#SBATCH -o creppo_v35_3agent_spr_%j.out
#SBATCH -e creppo_v35_3agent_spr_%j.err
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=112
#SBATCH -t 24:00:00
#SBATCH --mail-user=jeffreychen287@gmail.com
#SBATCH --mail-type=end,fail

module load python
source /scratch/11079/jeffreychen287/jax-aht/venv/bin/activate
cd /scratch/11079/jeffreychen287/jax-aht
export PYTHONPATH=$PYTHONPATH:$PWD
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Expedited 30M Timestep execution logic to comfortably finish Phase 1 and Phase 2 under 24 hours
/scratch/11079/jeffreychen287/jax-aht/venv/bin/python social_laws/run.py \
    task=continuous/coop_recon_3_agent_v33 \
    algorithm=creppo/continuous/coop_recon \
    logger.mode=offline \
    NUM_EXPT_AGENTS=3 \
    algorithm.TOTAL_TIMESTEPS=40000000 \
    algorithm.TOTAL_TIMESTEPS_DECAY=35000000 \
    algorithm.FIXED_EVAL=true \
    algorithm.SAP_DOMAIN_RANDOMIZE_PARTNER=true \
    algorithm.NUM_EVAL_EPISODES=50 \
    algorithm.EPSILON_OPTIMAL=0.01 \
    task.ENV_KWARGS.ego_centric_obs=true \
    label=jax_coop_recon_v35_3agent_0.01_eps

echo "v35 SPR fast-track training complete"
