#!/bin/bash
#SBATCH -J creppo_v35_h100
#SBATCH -o creppo_v35_h100_%j.out
#SBATCH -e creppo_v35_h100_%j.err
#SBATCH -p h100
#SBATCH -A IRI26004
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=96
#SBATCH -t 04:00:00
#SBATCH --mail-user=jeffreychen287@gmail.com
#SBATCH --mail-type=end,fail

module load python
source /scratch/11079/jeffreychen287/jax-aht/venv/bin/activate
cd /scratch/11079/jeffreychen287/jax-aht
export PYTHONPATH=$PYTHONPATH:$PWD

# CRITICAL: Prevent JAX from instantly hogging 100% of GPU 0 VRAM
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "Spawning v35 on GPU 0..."
# Run v35 (epsilon = 0.01) isolated fully on GPU 0
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
    label=jax_coop_recon_v35_3agent_0.01_eps &

# Suspend the shell until the background Python execution physically terminates
wait
echo "v35 h100 training complete"
