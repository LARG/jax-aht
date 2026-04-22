import os
import stat

base_dir = "/Users/jeffrey/Desktop/AHTMD/jax-aht"

def get_bash_template(algo, script_name, seeds, is_creppo=False, is_ppo=False):
    header = f"""#!/bin/bash
# Phase E: MARL Comparison {algo.upper()} — Social Law Sweep (aaronson/mckennie)
# Seeds: {', '.join(map(str, seeds))}

mkdir -p logs

export PYTHONPATH=$PYTHONPATH:$PWD
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_DEFAULT_MATMUL_PRECISION=highest

"""
    
    if algo.lower() == "mappo":
        header += """# Ensure venv is active - Prioritize 'venv' as it works on debruyne
if [ -f "venv/bin/python" ]; then
    VENV_PYTHON="$PWD/venv/bin/python"
elif [ -f "venv_aaronson/bin/python" ]; then
    VENV_PYTHON="$PWD/venv_aaronson/bin/python"
elif [ -f ".venv/bin/python" ]; then
    VENV_PYTHON="$PWD/.venv/bin/python"
else
    VENV_PYTHON="python"
fi
"""
    else:
        header += """# Ensure venv is active
if [ -f "venv/bin/python" ]; then
    VENV_PYTHON="$PWD/venv/bin/python"
elif [ -f "venv_aaronson/bin/python" ]; then
    VENV_PYTHON="$PWD/venv_aaronson/bin/python"
elif [ -f ".venv/bin/python" ]; then
    VENV_PYTHON="$PWD/.venv/bin/python"
else
    VENV_PYTHON="python"
fi
"""

    header += """
# Redirect WandB artifacts to scratch to save home folder space
mkdir -p /scratch/cluster/jeffrey9/wandb_cache
export WANDB_DIR=/scratch/cluster/jeffrey9/wandb_cache
export WANDB_CACHE_DIR=/scratch/cluster/jeffrey9/wandb_cache

"""

    if is_creppo:
        func = """run_exp() {
    local GPU=$1
    local TASK=$2
    local N=$3
    local LABEL=$4
    local SEED=$5

    echo ">>> Starting: $LABEL (seed=$SEED, N=$N) on GPU $GPU"
    CUDA_VISIBLE_DEVICES=$GPU \\
    $VENV_PYTHON social_laws/run_w_best_case.py \\
        task=$TASK \\
        algorithm=creppo/continuous/coop_recon \\
        algorithm.TRAIN_SEED=$SEED \\
        algorithm.USE_SAME_SEED=true \\
        algorithm.FIXED_EVAL=true \\
        NUM_EXPT_AGENTS=$N \\
        label=$LABEL \\
        logger.project=aht-benchmark \\
        logger.entity=jeffreychen287-the-university-of-texas-at-austin \\
        logger.mode=online \\
        algorithm.ALPHA_VERIFICATION=false \\
        >> logs/${LABEL}_seed${SEED}.out 2>&1 &
    echo "    PID=$! | log: logs/${LABEL}_seed${SEED}.out"
}

run_cond() {
    local TASK_PREFIX=$1   # e.g. "coop_recon_compare_law_0.15"
    local LABEL_PREFIX=$2  # e.g. "creppo_law_0.15"
    local SEED=$3

    # All 4 N-values in parallel across all 4 GPUs
    run_exp 0 continuous/${TASK_PREFIX}_2_agent 2 ${LABEL_PREFIX}_2_agent $SEED
    run_exp 1 continuous/${TASK_PREFIX}_3_agent 3 ${LABEL_PREFIX}_3_agent $SEED
    run_exp 2 continuous/${TASK_PREFIX}_4_agent 4 ${LABEL_PREFIX}_4_agent $SEED
    run_exp 3 continuous/${TASK_PREFIX}_5_agent 5 ${LABEL_PREFIX}_5_agent $SEED
    wait
    echo "  All N done for seed=$SEED"
}
"""
    elif is_ppo:
        func = """run_exp() {
    local GPU=$1
    local TASK=$2
    local N=$3
    local LABEL=$4
    local SEED=$5

    CUDA_VISIBLE_DEVICES=$GPU \\
    $VENV_PYTHON social_laws/run.py \\
        task=$TASK \\
        algorithm=ppo/continuous/coop_recon \\
        value_function=dqnppo/continuous/coop_recon \\
        algorithm.TRAIN_SEED=$SEED \\
        algorithm.USE_SAME_SEED=true \\
        value_function.USE_SAME_SEED=true \\
        algorithm.FIXED_EVAL=true \\
        value_function.FIXED_EVAL=true \\
        NUM_EXPT_AGENTS=$N \\
        label=$LABEL \\
        logger.project=aht-benchmark \\
        logger.entity=jeffreychen287-the-university-of-texas-at-austin \\
        logger.mode=online \\
        algorithm.ALPHA_VERIFICATION=false \\
        >> logs/${LABEL}_seed${SEED}_${SLURM_JOB_ID:+$SLURM_JOB_ID}.out 2>&1 &
}

run_cond() {
    local TASK_PREFIX=$1
    local LABEL_PREFIX=$2
    local SEED=$3

    run_exp 1 continuous/${TASK_PREFIX}_2_agent 2 ${LABEL_PREFIX}_2_agent $SEED
    run_exp 2 continuous/${TASK_PREFIX}_3_agent 3 ${LABEL_PREFIX}_3_agent $SEED
    run_exp 3 continuous/${TASK_PREFIX}_4_agent 4 ${LABEL_PREFIX}_4_agent $SEED
    wait
    run_exp 1 continuous/${TASK_PREFIX}_5_agent 5 ${LABEL_PREFIX}_5_agent $SEED
    wait
}
"""
    else:
        # IPPO or MAPPO
        func = f"""run_exp() {{
    local GPU=$1
    local TASK=$2
    local N=$3
    local LABEL=$4
    local SEED=$5

    CUDA_VISIBLE_DEVICES=$GPU \\
    $VENV_PYTHON social_laws/experiments/run_marl_compare_coop_recon.py \\
        task=$TASK \\
        algorithm={algo.lower()}/continuous/coop_recon \\
        algorithm.TRAIN_SEED=$SEED \\
        algorithm.USE_SAME_SEED=true \\
        algorithm.FIXED_EVAL=true \\
        NUM_EXPT_AGENTS=$N \\
        label=$LABEL \\
        logger.project=aht-benchmark \\
        logger.entity=jeffreychen287-the-university-of-texas-at-austin \\
        logger.mode=online \\
        >> logs/${{LABEL}}_seed${{SEED}}_${{SLURM_JOB_ID:+$SLURM_JOB_ID}}.out 2>&1 &
}}

run_cond() {{
    local TASK_PREFIX=$1
    local LABEL_PREFIX=$2
    local SEED=$3

    run_exp 1 continuous/${{TASK_PREFIX}}_2_agent 2 ${{LABEL_PREFIX}}_2_agent $SEED
    run_exp 2 continuous/${{TASK_PREFIX}}_3_agent 3 ${{LABEL_PREFIX}}_3_agent $SEED
    run_exp 3 continuous/${{TASK_PREFIX}}_4_agent 4 ${{LABEL_PREFIX}}_4_agent $SEED
    wait
    run_exp 1 continuous/${{TASK_PREFIX}}_5_agent 5 ${{LABEL_PREFIX}}_5_agent $SEED
    wait
}}
"""
    
    body = f"\necho \"Starting {algo.upper()} Social Law Sweep Comparisons...\"\n"
    body += f"for SEED in {' '.join(map(str, seeds))}; do\n"
    body += f"    echo \"=== Running {algo.upper()} with SEED=$SEED ===\"\n\n"
    
    for variation in ["0.0", "0.15", "0.3"]:
        body += f"    echo \"--- Variation: Law {variation} ---\"\n"
        body += f"    run_cond \"coop_recon_compare_law_{variation}\" \"{algo.lower()}_law_{variation}\" $SEED\n"
        body += f"    echo \"--- Finished Law {variation} for seed $SEED ---\"\n\n"
        
    body += "done\n\n"
    body += f"echo \"All {algo.upper()} social law sweep comparisons finished.\"\n"
    
    return header + func + body


scripts = [
    ("IPPO", "run_marl_compare_ippo_law_sweep.sh", [72128, 721280, 721281, 721282, 721283], False, False),
    ("MAPPO", "run_marl_compare_mappo_law_sweep.sh", [72128, 721280, 721281, 721282, 721283], False, False),
    ("PPO", "run_marl_compare_ppo_law_sweep.sh", [72128, 721280, 721281, 721282, 721283], False, True),
    ("CREPPO", "run_marl_compare_creppo_mckennie_law_sweep.sh", [72128, 721280, 721281, 721282, 721283], True, False),
]

for algo, name, seeds, is_creppo, is_ppo in scripts:
    content = get_bash_template(algo, name, seeds, is_creppo, is_ppo)
    script_path = os.path.join(base_dir, name)
    with open(script_path, "w") as f:
        f.write(content)
    
    # Make executable
    st = os.stat(script_path)
    os.chmod(script_path, st.st_mode | stat.S_IEXEC)

print("Successfully generated and chmod'ed 4 bash execution scripts.")
