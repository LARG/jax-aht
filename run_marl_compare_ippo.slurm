#!/bin/bash
#----------------------------------------------------
# Phase E: MARL Comparison IPPO
# 1 Node, 8 GPUs (amd-rtx B200)
#----------------------------------------------------
#SBATCH -J e_ippo
#SBATCH -p amd-rtx
#SBATCH -A IRI26004
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH -o logs/e_ippo_%j.out
#SBATCH -e logs/e_ippo_%j.err
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=jeffreychen287@utexas.edu

mkdir -p logs

module purge
module load gcc/13.2.0 cuda/12.8 python

cd /scratch/11079/jeffreychen287/jax-aht
source /scratch/11079/jeffreychen287/jax-aht/venv/bin/activate

export PYTHONPATH=$PYTHONPATH:$PWD
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_DEFAULT_MATMUL_PRECISION=highest

run_exp() {
    local GPU=$1
    local TASK=$2
    local LABEL=$3

    CUDA_VISIBLE_DEVICES=$GPU \
    python social_laws/experiments/run_marl_compare_coop_recon.py \
        task=$TASK \
        algorithm=ippo/continuous/coop_recon \
        algorithm.TRAIN_SEED=72128 \
        algorithm.USE_SAME_SEED=true \
        algorithm.FIXED_EVAL=true \
        label=$LABEL \
        logger.project=NEURIPS-2026 \
        logger.mode=online \
        >> logs/${LABEL}_${SLURM_JOB_ID}.out 2>&1 &
}

echo "Starting IPPO Phase E Comparisons (8 jobs on 8 GPUs)..."

# No Law (Baseline)
run_exp 0 continuous/coop_recon_compare_no_law_2_agent ippo_no_law_2_agent
run_exp 1 continuous/coop_recon_compare_no_law_3_agent ippo_no_law_3_agent
run_exp 2 continuous/coop_recon_compare_no_law_4_agent ippo_no_law_4_agent
run_exp 3 continuous/coop_recon_compare_no_law_5_agent ippo_no_law_5_agent

# Social Law
run_exp 4 continuous/coop_recon_compare_law_2_agent ippo_law_2_agent
run_exp 5 continuous/coop_recon_compare_law_3_agent ippo_law_3_agent
run_exp 6 continuous/coop_recon_compare_law_4_agent ippo_law_4_agent
run_exp 7 continuous/coop_recon_compare_law_5_agent ippo_law_5_agent

wait

echo "All IPPO comparisons finished."
