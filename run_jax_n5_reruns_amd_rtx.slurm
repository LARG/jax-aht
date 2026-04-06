#!/bin/bash
#----------------------------------------------------
# N=5 Reruns — 3-GPU amd-rtx (B200 Blackwell)
# Branch: collision_avoidance_social_law
#
# These three N=5 jobs crashed in the 8-GPU run because 8 concurrent
# JAX LLVM JIT compilations exhausted the node's CPU RAM.
# Running only 3 heavily compiling N=5 scripts on a dedicated node
# preserves enough host memory to successfully JIT compile.
#----------------------------------------------------
#SBATCH -J coop_recon_n5_rerun
#SBATCH -p amd-rtx
#SBATCH -A IRI26004
#SBATCH -N 1
#SBATCH -n 3
#SBATCH --gpus=3
#SBATCH -t 12:00:00
#SBATCH -o logs/n5_rerun_%j.out
#SBATCH -e logs/n5_rerun_%j.err
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=jeffreychen287@utexas.edu

mkdir -p logs

module purge
module load gcc/13.2.0 cuda/12.8 python

cd /scratch/11079/jeffreychen287/jax-aht
source /scratch/11079/jeffreychen287/jax-aht/venv/bin/activate

export PYTHONPATH=$PYTHONPATH:$PWD
export XLA_PYTHON_CLIENT_PREALLOCATE=false

run_exp() {
    local GPU=$1
    local TASK=$2
    local N=$3
    local LABEL=$4

    CUDA_VISIBLE_DEVICES=$GPU JAX_DEFAULT_MATMUL_PRECISION=highest \
    /scratch/11079/jeffreychen287/jax-aht/venv/bin/python social_laws/run.py \
        task=$TASK \
        algorithm=creppo/continuous/coop_recon \
        logger.mode=offline \
        NUM_EXPT_AGENTS=$N \
        algorithm.TOTAL_TIMESTEPS=40000000 \
        algorithm.TOTAL_TIMESTEPS_DECAY=35000000 \
        algorithm.FIXED_EVAL=true \
        algorithm.SAP_DOMAIN_RANDOMIZE_PARTNER=true \
        algorithm.NUM_EVAL_EPISODES=200 \
        algorithm.EPSILON_OPTIMAL=0.01 \
        task.ENV_KWARGS.ego_centric_obs=true \
        label=$LABEL \
        >> logs/${LABEL}_${SLURM_JOB_ID}.out 2>&1 &
    echo "Launched $LABEL on GPU $GPU (PID $!)"
}

# ── Rerun the 3 crashed N=5 jobs ─────────────────────────────────────────────
run_exp 0 continuous/coop_recon_5_agent_v61 5 jax_coop_recon_v61_5agent_baseline_200ep
run_exp 1 continuous/coop_recon_5_agent_v59 5 jax_coop_recon_v59_5agent_social_0.15_200ep
run_exp 2 continuous/coop_recon_5_agent_v65 5 jax_coop_recon_v65_5agent_social_0.20_200ep

echo "N=5 reruns launched. Waiting for completion..."
wait
echo "All N=5 experiments complete."
