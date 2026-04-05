#!/bin/bash
#----------------------------------------------------
# V59 — N=5, Social Min-Distance 0.15, 200 eval ep
# Branch: collision_avoidance_social_law
# Baseline comparison: V50 (N=5, allocation only, 50 ep, α≈0.53)
# N=5 is the highest density — most dramatic collision avoidance test.
#----------------------------------------------------
#SBATCH -J coop_recon_v59_5agent_law
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=96
#SBATCH -t 24:00:00
#SBATCH -o logs/v59_%j.out
#SBATCH -e logs/v59_%j.err
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=jeffreychen287@utexas.edu

mkdir -p logs

module load python
module load cuda/12.3

cd /scratch/11079/jeffreychen287/jax-aht
git pull origin collision_avoidance_social_law

source /scratch/11079/jeffreychen287/jax-aht/venv/bin/activate

export PYTHONPATH=$PYTHONPATH:$PWD
export XLA_PYTHON_CLIENT_PREALLOCATE=false

JAX_DEFAULT_MATMUL_PRECISION=highest \
/scratch/11079/jeffreychen287/jax-aht/venv/bin/python social_laws/run.py \
    task=continuous/coop_recon_5_agent_v59 \
    algorithm=creppo/continuous/coop_recon \
    logger.mode=offline \
    NUM_EXPT_AGENTS=5 \
    algorithm.TOTAL_TIMESTEPS=40000000 \
    algorithm.TOTAL_TIMESTEPS_DECAY=35000000 \
    algorithm.FIXED_EVAL=true \
    algorithm.SAP_DOMAIN_RANDOMIZE_PARTNER=true \
    algorithm.NUM_EVAL_EPISODES=200 \
    algorithm.EPSILON_OPTIMAL=0.01 \
    task.ENV_KWARGS.ego_centric_obs=true \
    label=jax_coop_recon_v59_5agent_social_0.15_200ep

echo "V59 training complete"
