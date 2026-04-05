#!/bin/bash
#----------------------------------------------------
# V58 + V59 + V60 + V61 — 4-GPU amd-rtx (B200 Blackwell)
# Branch: collision_avoidance_social_law
#
# Runs 4 experiments in parallel, one per GPU:
#   GPU 0: V60 — N=4, no law, 200 ep      (baseline)
#   GPU 1: V61 — N=5, no law, 200 ep      (baseline)
#   GPU 2: V58 — N=4, law 0.15, 200 ep   (treatment)
#   GPU 3: V59 — N=5, law 0.15, 200 ep   (treatment)
#
# B200 NOTE: JAX must support sm_100. If you see device errors,
# run the fix below before submitting:
#   pip install -U "jax[cuda12]"
#----------------------------------------------------
#SBATCH -J coop_recon_v58_v59_v60_v61_b200
#SBATCH -p amd-rtx
#SBATCH -A IRI26004
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gpus=4
#SBATCH -t 08:00:00
#SBATCH -o logs/v58_v61_%j.out
#SBATCH -e logs/v58_v61_%j.err
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=jeffreychen287@utexas.edu

mkdir -p logs

module purge
module load gcc/13.2.0 cuda/12.8

cd /scratch/11079/jeffreychen287/jax-aht
git pull origin collision_avoidance_social_law

source /scratch/11079/jeffreychen287/jax-aht/venv/bin/activate

export PYTHONPATH=$PYTHONPATH:$PWD
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# ── Helper: run one experiment on a specific GPU in background ──────────────
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

# ── Launch all 4 experiments ─────────────────────────────────────────────────
run_exp 0 continuous/coop_recon_4_agent_v60 4 jax_coop_recon_v60_4agent_baseline_200ep
run_exp 1 continuous/coop_recon_5_agent_v61 5 jax_coop_recon_v61_5agent_baseline_200ep
run_exp 2 continuous/coop_recon_4_agent_v58 4 jax_coop_recon_v58_4agent_social_0.15_200ep
run_exp 3 continuous/coop_recon_5_agent_v59 5 jax_coop_recon_v59_5agent_social_0.15_200ep

# ── Wait for all background jobs ─────────────────────────────────────────────
echo "All 4 experiments launched. Waiting for completion..."
wait
echo "All V58/V59/V60/V61 experiments complete."
