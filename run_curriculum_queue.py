"""
Dynamic GPU-aware queue for IPPO and MAPPO curriculum runs
on the continuous coop_recon environment.

Runs on aaronson (4x A100s). Monitors GPU memory and dispatches
curriculum jobs to idle GPUs. Multi-instance safe.

Usage:
    python run_curriculum_queue.py

The curriculum runs use social_laws/experiments/run_marl_compare.py
with algorithm.CURRICULUM=true. The task configs include both ENV_KWARGS
(law env) and NO_LAW_ENV_KWARGS (unconstrained env) so the agent
first masters the law env before graduating to the harder no-law setting.
"""
import os
import subprocess
import time

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
VENV_PYTHON = os.environ.get("VIRTUAL_ENV", "") + "/bin/python" if os.environ.get("VIRTUAL_ENV") else "python"
if not os.path.exists(VENV_PYTHON):
    VENV_PYTHON = "venv/bin/python" if os.path.exists("venv/bin/python") else "venv_aaronson/bin/python"

SEEDS = [72128, 721280, 721281, 721282, 721283]
ALGORITHMS = ["ippo", "mappo"]
LAWS = ["law_0.0", "law_0.1", "law_0.2"]
N_AGENTS = [2, 3, 4, 5]

os.makedirs("logs", exist_ok=True)
WANDB_CACHE = "/scratch/cluster/jeffrey9/wandb_cache"
os.makedirs(WANDB_CACHE, exist_ok=True)


def build_cmd(algo, law_name, n, seed):
    task = f"continuous/coop_recon_compare_curriculum_{law_name}_{n}_agent"
    label = f"{algo}_curriculum_{law_name}_{n}_agent"

    cmd = [
        VENV_PYTHON, "social_laws/experiments/run_marl_compare.py",
        f"task={task}",
        f"algorithm={algo}/continuous/coop_recon",
        f"algorithm.TRAIN_SEED={seed}",
        "algorithm.USE_SAME_SEED=true",
        "algorithm.FIXED_EVAL=true",
        "algorithm.CURRICULUM=true",
        f"NUM_EXPT_AGENTS={n}",
        f"label={label}",
        "logger.project=aht-benchmark",
        "logger.entity=jeffreychen287-the-university-of-texas-at-austin",
        "logger.mode=online",
    ]
    logfile = f"logs/{label}_seed{seed}.out"
    desc = f"{label} (seed:{seed} N:{n})"
    return cmd, logfile, desc


# ---------------------------------------------------------
# Build the queue
# ---------------------------------------------------------
job_queue = []
for algo in ALGORITHMS:
    for law_name in LAWS:
        for n in N_AGENTS:
            for seed in SEEDS:
                cmd, logfile, desc = build_cmd(algo, law_name, n, seed)
                job_queue.append({"cmd": cmd, "logfile": logfile, "desc": desc})

print(f"Total curriculum jobs: {len(job_queue)}")  # 2 * 3 * 4 * 5 = 120


# ---------------------------------------------------------
# Skip already completed runs
# ---------------------------------------------------------
def should_skip(logfile):
    return os.path.exists(logfile) and os.path.getsize(logfile) > 500


pending_jobs = []
for job in job_queue:
    if should_skip(job["logfile"]):
        print(f"Skipping {job['desc']} - logfile exists and has data")
    else:
        if os.path.exists(job["logfile"]):
            os.remove(job["logfile"])  # remove stale crash log
        pending_jobs.append(job)

print(f"Jobs left to run: {len(pending_jobs)}")


# ---------------------------------------------------------
# Dynamic GPU runner
# ---------------------------------------------------------
def get_free_gpus():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            text=True
        )
        return [
            int(line.split(",")[0])
            for line in output.strip().split("\n")
            if int(line.split(",")[1].strip()) < 1000
        ]
    except Exception as e:
        print(f"Error querying GPUs: {e}")
        return []


def main():
    active_processes = {}  # GPU index -> subprocess

    while pending_jobs or active_processes:
        # Reap finished processes
        for gpu in [g for g, p in active_processes.items() if p.poll() is not None]:
            print(f"[GPU {gpu}] Finished a task.")
            del active_processes[gpu]

        # Dispatch to free GPUs
        free_gpus = [g for g in get_free_gpus() if g not in active_processes]
        while free_gpus and pending_jobs:
            job = pending_jobs[0]
            # Multi-instance safety: re-check if another runner claimed it
            if should_skip(job["logfile"]):
                pending_jobs.pop(0)
                continue

            gpu = free_gpus.pop(0)
            job = pending_jobs.pop(0)
            print(f"[GPU {gpu}] Starting {job['desc']}")

            # Claim the logfile immediately
            open(job["logfile"], "a").close()

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["WANDB_DIR"] = WANDB_CACHE
            env["WANDB_CACHE_DIR"] = WANDB_CACHE
            env["PYTHONPATH"] = env.get("PYTHONPATH", "") + ":" + os.getcwd()
            env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            env["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"

            with open(job["logfile"], "a") as f:
                proc = subprocess.Popen(job["cmd"], env=env, stdout=f, stderr=subprocess.STDOUT)
            active_processes[gpu] = proc
            time.sleep(10)  # stagger launches

        time.sleep(20)

    print("All curriculum tasks complete!")


if __name__ == "__main__":
    main()
