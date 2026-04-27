#!/bin/bash
#----------------------------------------------------
# Pizza V2 CREPPO test run — SPR nodes (Stampede3)
#----------------------------------------------------
#SBATCH -J pizza_v2_creppo
#SBATCH -o logs/pizza_v2_creppo_%j.out
#SBATCH -e logs/pizza_v2_creppo_%j.err
#SBATCH -p spr
#SBATCH -A IRI26004
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=112
#SBATCH -t 24:00:00
#SBATCH --mail-user=jeffreychen287@gmail.com
#SBATCH --mail-type=end,fail

mkdir -p logs

module load python
source /scratch/11079/jeffreychen287/jax-aht/venv/bin/activate

cd /scratch/11079/jeffreychen287/jax-aht

export PYTHONPATH=$PYTHONPATH:$PWD
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_DEFAULT_MATMUL_PRECISION=highest

VENV_PYTHON=/scratch/11079/jeffreychen287/jax-aht/venv/bin/python

echo "================================================="
echo "Pizza V2 CREPPO — SPR test run"
echo "Branch: $(git rev-parse --abbrev-ref HEAD)  $(git rev-parse --short HEAD)"
echo "================================================="

$VENV_PYTHON social_laws/run_w_best_case.py \
    task=rddl/pizza_v2 \
    algorithm=creppo/rddl/pizza_v2 \
    algorithm.ALPHA_VERIFICATION=true \
    algorithm.TRAIN_SEED=72128 \
    algorithm.EVAL_SEED=72128 \
    algorithm.USE_SAME_SEED=true \
    algorithm.FIXED_EVAL=true \
    logger.project=aht-benchmark \
    logger.entity=jeffreychen287-the-university-of-texas-at-austin \
    logger.mode=online

echo "Pizza V2 CREPPO run complete."
