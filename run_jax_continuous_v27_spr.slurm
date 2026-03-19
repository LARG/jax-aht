#!/bin/bash
#SBATCH -J jax_coop_recon_creppo_v27
#SBATCH -o jax_coop_recon_creppo_v27_%j.out
#SBATCH -e jax_coop_recon_creppo_v27_%j.err
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=112
#SBATCH -t 10:00:00
#SBATCH --mail-user=jeffreychen287@gmail.com
#SBATCH --mail-type=all

module load python
source /scratch/11079/jeffreychen287/jax-aht/venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# v27: ego-centric obs with bug fix (v25), doubled to 80M steps.
# Rationale: v25 Proj/CheckpointReturn was still climbing at 40M (~4.4 vs optimal ~7.4),
# indicating the ego-centric SAP policy hadn't converged yet. The per-episode alpha
# outlier issue (negative plotted alpha) goes away once the policy is converged and
# per-episode returns are consistently non-zero.
/scratch/11079/jeffreychen287/jax-aht/venv/bin/python social_laws/run.py \
    task=continuous/coop_recon \
    algorithm=creppo/continuous/coop_recon \
    logger.mode=offline \
    algorithm.TOTAL_TIMESTEPS=80000000 \
    algorithm.TOTAL_TIMESTEPS_DECAY=80000000 \
    algorithm.FIXED_EVAL=true \
    algorithm.SAP_DOMAIN_RANDOMIZE_PARTNER=true \
    algorithm.NUM_EVAL_EPISODES=50 \
    task.ENV_KWARGS.ego_centric_obs=true \
    label=jax_coop_recon_v27_phaseA_ego_80M_creppo

echo "Training complete."
