#!/bin/bash
#SBATCH -J creppo_v33_v34_h100
#SBATCH -o creppo_v33_v34_h100_%j.out
#SBATCH -e creppo_v33_v34_h100_%j.err
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=96
#SBATCH -t 12:00:00
#SBATCH --mail-user=jeffreychen287@gmail.com
#SBATCH --mail-type=end,fail

module load python
source /scratch/11079/jeffreychen287/jax-aht/venv/bin/activate
cd /scratch/11079/jeffreychen287/jax-aht
export PYTHONPATH=$PYTHONPATH:$PWD

# CRITICAL: Prevent JAX from instantly hogging 100% of all 4 GPUs' VRAM
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "Spawning v33 on GPU 0..."
# Run v33 (epsilon = 0.01) isolated fully on GPU 0
CUDA_VISIBLE_DEVICES=0 /scratch/11079/jeffreychen287/jax-aht/venv/bin/python social_laws/run.py \
    task=continuous/coop_recon_3_agent_v33 \
    algorithm=creppo/continuous/coop_recon \
    logger.mode=offline \
    NUM_EXPT_AGENTS=3 \
    algorithm.TOTAL_TIMESTEPS=100000000 \
    algorithm.TOTAL_TIMESTEPS_DECAY=80000000 \
    algorithm.FIXED_EVAL=true \
    algorithm.SAP_DOMAIN_RANDOMIZE_PARTNER=true \
    algorithm.NUM_EVAL_EPISODES=50 \
    algorithm.EPSILON_OPTIMAL=0.01 \
    task.ENV_KWARGS.ego_centric_obs=true \
    label=jax_coop_recon_v33_3agent_0.01_eps_control &

echo "Spawning v34 on GPU 1..."
# Run v34 (epsilon = 0.001) isolated fully on GPU 1
CUDA_VISIBLE_DEVICES=1 /scratch/11079/jeffreychen287/jax-aht/venv/bin/python social_laws/run.py \
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
    label=jax_coop_recon_v34_3agent_0.001_eps_control &

# Suspend the shell until both background Python executions physically terminate
wait
echo "v33 & v34 parallel h100 training complete"
