#!/bin/bash
#SBATCH -J creppo_sweep_non_ego
#SBATCH -o creppo_sweep_non_ego_%A_%a.out
#SBATCH -e creppo_sweep_non_ego_%A_%a.err
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=112
#SBATCH -t 06:00:00
#SBATCH --array=0-11          # 12 combinations (3 LR x 2 GAMMA x 2 HIDDEN_SIZE)
#SBATCH --mail-user=jeffreychen287@gmail.com
#SBATCH --mail-type=end,fail

module load python
source /scratch/11079/jeffreychen287/jax-aht/venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# ==============================================================
# Hyperparameter sweep — ego_centric_obs=FALSE (absolute obs)
# Baseline config: v20/v26 settings, 40M steps
# ==============================================================

# Define sweep axes
LR_VALUES=(0.0001 0.0003 0.0005)           # 3 values
GAMMA_VALUES=(0.95 0.99)                   # 2 values
HIDDEN_SIZE_VALUES=(128 256)               # 2 values

# Total combinations: 3 x 2 x 2 = 12  (matches --array=0-11)
N_LR=${#LR_VALUES[@]}
N_GAMMA=${#GAMMA_VALUES[@]}
N_HIDDEN=${#HIDDEN_SIZE_VALUES[@]}

TASK_ID=${SLURM_ARRAY_TASK_ID}
LR_IDX=$(( TASK_ID / (N_GAMMA * N_HIDDEN) ))
GAMMA_IDX=$(( (TASK_ID / N_HIDDEN) % N_GAMMA ))
HIDDEN_IDX=$(( TASK_ID % N_HIDDEN ))

LR=${LR_VALUES[$LR_IDX]}
GAMMA=${GAMMA_VALUES[$GAMMA_IDX]}
HIDDEN=${HIDDEN_SIZE_VALUES[$HIDDEN_IDX]}

echo "Array task $TASK_ID: LR=$LR, GAMMA=$GAMMA, HIDDEN_SIZE=$HIDDEN"

LABEL="creppo_sweep_non_ego_lr${LR}_g${GAMMA}_h${HIDDEN}"

/scratch/11079/jeffreychen287/jax-aht/venv/bin/python social_laws/run.py \
    task=continuous/coop_recon \
    algorithm=creppo/continuous/coop_recon \
    logger.mode=offline \
    algorithm.TOTAL_TIMESTEPS=40000000 \
    algorithm.FIXED_EVAL=true \
    algorithm.SAP_DOMAIN_RANDOMIZE_PARTNER=true \
    algorithm.NUM_EVAL_EPISODES=50 \
    algorithm.LR=$LR \
    algorithm.GAMMA=$GAMMA \
    algorithm.HIDDEN_SIZE=$HIDDEN \
    task.ENV_KWARGS.ego_centric_obs=false \
    label=$LABEL

echo "Training complete: $LABEL"
