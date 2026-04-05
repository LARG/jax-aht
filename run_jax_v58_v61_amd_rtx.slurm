#!/bin/bash
#----------------------------------------------------
# V58-V65 — 8-GPU amd-rtx (B200 Blackwell)
# Branch: collision_avoidance_social_law
#
# 8 experiments in parallel, one per GPU:
#   GPU 0: V60 — N=4, no law, 200ep       (N=4 baseline)
#   GPU 1: V61 — N=5, no law, 200ep       (N=5 baseline)
#   GPU 2: V58 — N=4, law 0.15, 200ep    (N=4 law)
#   GPU 3: V59 — N=5, law 0.15, 200ep    (N=5 law)
#   GPU 4: V62 — N=2, no law, 200ep       (N=2 baseline, completes sweep)
#   GPU 5: V63 — N=2, law 0.15, 200ep    (N=2 law, confirms no downside)
#   GPU 6: V64 — N=4, law 0.20, 200ep    (law sensitivity at N=4)
#   GPU 7: V65 — N=5, law 0.20, 200ep    (law sensitivity at N=5)
#
# B200 NOTE: verify JAX supports sm_100 first:
#   idev -p amd-rtx -A IRI26004 -m 15
#   python -c "import jax; print(jax.devices())"
#   # If error: pip install -U "jax[cuda12]"
#----------------------------------------------------
#SBATCH -J coop_recon_v58_v65_b200
#SBATCH -p amd-rtx
#SBATCH -A IRI26004
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gpus=8
#SBATCH -t 08:00:00
#SBATCH -o logs/v58_v65_%j.out
#SBATCH -e logs/v58_v65_%j.err
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=jeffreychen287@utexas.edu

mkdir -p logs

module purge
module load gcc/13.2.0 cuda/12.8

cd /scratch/11079/jeffreychen287/jax-aht
git pull origin collision_avoidance_social_law

source /scratch/11079/jeffreychen287/jax-aht/venv/bin/activate

# jaxmarl and its deps are not in the venv on amd-rtx nodes.
# Install full chain, but keep jax[cuda12] pinned at 0.9.2.
pip install jaxmarl chex flax optax distrax matplotlib pygame imageio --upgrade-strategy only-if-needed -q

export PYTHONPATH=$PYTHONPATH:$PWD
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# ── Helper: launch one experiment on a specific GPU in background ────────────
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

# ── Density sweep baselines (GPUs 0-1) ───────────────────────────────────────
run_exp 0 continuous/coop_recon_4_agent_v60 4 jax_coop_recon_v60_4agent_baseline_200ep
run_exp 1 continuous/coop_recon_5_agent_v61 5 jax_coop_recon_v61_5agent_baseline_200ep

# ── Density sweep with law=0.15 (GPUs 2-3) ───────────────────────────────────
run_exp 2 continuous/coop_recon_4_agent_v58 4 jax_coop_recon_v58_4agent_social_0.15_200ep
run_exp 3 continuous/coop_recon_5_agent_v59 5 jax_coop_recon_v59_5agent_social_0.15_200ep

# ── N=2 density point (GPUs 4-5) ─────────────────────────────────────────────
run_exp 4 continuous/coop_recon_2_agent_v62 2 jax_coop_recon_v62_2agent_baseline_200ep
run_exp 5 continuous/coop_recon_2_agent_v63 2 jax_coop_recon_v63_2agent_social_0.15_200ep

# ── Law sensitivity: threshold=0.20 (GPUs 6-7) ───────────────────────────────
run_exp 6 continuous/coop_recon_4_agent_v64 4 jax_coop_recon_v64_4agent_social_0.20_200ep
run_exp 7 continuous/coop_recon_5_agent_v65 5 jax_coop_recon_v65_5agent_social_0.20_200ep

# ── Wait for all 8 background jobs ───────────────────────────────────────────
echo "All 8 experiments launched. Waiting for completion..."
wait
echo "All V58-V65 experiments complete."

