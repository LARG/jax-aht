#!/bin/bash
#SBATCH -J jax_coop_recon_v45
#SBATCH -o logs/v45_out_%j.txt
#SBATCH -p spr
#SBATCH -A IRI26004
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=112
#SBATCH -t 24:00:00

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load miniconda3
source /scratch/11079/jeffreychen287/jax-aht/venv/bin/activate
export WANDB_EXECUTABLE="/scratch/11079/jeffreychen287/jax-aht/venv/bin/wandb"
cd /scratch/11079/jeffreychen287/jax-aht
export PYTHONPATH=.

python social_laws/run.py \
    task=continuous/coop_recon_4_agent_v43 \
    algorithm=creppo/continuous/coop_recon \
    logger.mode=offline \
    NUM_EXPT_AGENTS=4 \
    algorithm.TOTAL_TIMESTEPS=60000000 \
    algorithm.TOTAL_TIMESTEPS_DECAY=55000000 \
    algorithm.FIXED_EVAL=true \
    algorithm.SAP_DOMAIN_RANDOMIZE_PARTNER=true \
    algorithm.NUM_EVAL_EPISODES=50 \
    algorithm.EPSILON_OPTIMAL=0.01 \
    task.ENV_KWARGS.ego_centric_obs=true \
    algorithm.REVERSE_AGENT_TRAIN_ORDER=true \
    label=jax_coop_recon_v45_4agent_1.5grid_reverse

echo "Syncing V45 offline run to WandB..."
/scratch/11079/jeffreychen287/jax-aht/venv/bin/wandb sync \
    --entity jeffreychen287-the-university-of-texas-at-austin \
    --project aht-benchmark \
    /scratch/11079/jeffreychen287/jax-aht/results/continuous/coop_recon_n_agent_4/creppo_mlp/jax_coop_recon_v45_4agent_1.5grid_reverse/*/wandb/offline-run-*
