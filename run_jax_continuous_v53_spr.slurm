#!/bin/bash
#----------------------------------------------------
# V53 — Social Minimum Distance Law, N=3, 1.0 grid
# Branch: collision_avoidance_social_law
# Compares directly to V37 (same config + social_min_dist=0.15)
#----------------------------------------------------
#SBATCH -J coop_recon_v53_3agent_law
#SBATCH -p spr
#SBATCH -A IRI26004
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=112
#SBATCH -t 24:00:00
#SBATCH -o logs/v53_%j.out
#SBATCH -e logs/v53_%j.err
#SBATCH --mail-type=end
#SBATCH --mail-user=jeffreychen287@utexas.edu

mkdir -p logs

cd /scratch/11079/jeffreychen287/jax-aht
git pull origin collision_avoidance_social_law

source /scratch/11079/jeffreychen287/jax-aht/venv/bin/activate

PYTHONPATH=. JAX_DEFAULT_MATMUL_PRECISION=highest \
python social_laws/run.py \
    task=continuous/coop_recon_3_agent_v53 \
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
    label=jax_coop_recon_v53_3agent_social_min_dist_0.15
