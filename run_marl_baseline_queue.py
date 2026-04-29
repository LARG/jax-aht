import os
import subprocess
import time
import sys

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
VENV_PYTHON = os.environ.get("VIRTUAL_ENV", "") + "/bin/python" if os.environ.get("VIRTUAL_ENV") else "python"
if not os.path.exists(VENV_PYTHON):
    VENV_PYTHON = "venv_aaronson/bin/python" if os.path.exists("venv_aaronson/bin/python") else "venv/bin/python"

SEEDS = [72128, 721280, 721281, 721282, 721283]
ALGORITHMS = ["ippo", "mappo", "ppo"]
LAWS = [
    ("coop_recon_compare_no_law", "no_law"),
    ("coop_recon_compare_law_0.0", "law_0.0"),
    ("coop_recon_compare_law_0.1", "law_0.1")
]
N_AGENTS = [2, 3, 4, 5]

os.makedirs("logs", exist_ok=True)
os.makedirs(os.environ.get("WANDB_DIR", "/scratch/cluster/jeffrey9/wandb_cache"), exist_ok=True)

# ---------------------------------------------------------
# Build the Queue
# ---------------------------------------------------------
job_queue = []

for algo in ALGORITHMS:
    for task_prefix, law_name in LAWS:
        for n in N_AGENTS:
            for seed in SEEDS:
                task = f"continuous/{task_prefix}_{n}_agent"
                label = f"{algo}_{law_name}_{n}_agent"
                logfile = f"logs/{label}_seed{seed}.out"
                
                if algo in ["ippo", "mappo"]:
                    cmd = [
                        VENV_PYTHON, "social_laws/experiments/run_marl_compare_coop_recon.py",
                        f"task={task}",
                        f"algorithm={algo}/continuous/coop_recon",
                        f"algorithm.TRAIN_SEED={seed}",
                        "algorithm.USE_SAME_SEED=true",
                        "algorithm.FIXED_EVAL=true",
                        f"NUM_EXPT_AGENTS={n}",
                        f"label={label}",
                        "logger.project=aht-benchmark",
                        "logger.entity=jeffreychen287-the-university-of-texas-at-austin",
                        "logger.mode=online"
                    ]
                else: # ppo
                    cmd = [
                        VENV_PYTHON, "social_laws/run.py",
                        f"task={task}",
                        "algorithm=ppo/continuous/coop_recon",
                        "value_function=dqnppo/continuous/coop_recon",
                        f"algorithm.TRAIN_SEED={seed}",
                        "algorithm.USE_SAME_SEED=true",
                        "value_function.USE_SAME_SEED=true",
                        "algorithm.FIXED_EVAL=true",
                        "value_function.FIXED_EVAL=true",
                        f"NUM_EXPT_AGENTS={n}",
                        f"label={label}",
                        "logger.project=aht-benchmark",
                        "logger.entity=jeffreychen287-the-university-of-texas-at-austin",
                        "logger.mode=online",
                        "algorithm.ALPHA_VERIFICATION=false"
                    ]

                job_queue.append({
                    "cmd": cmd,
                    "logfile": logfile,
                    "desc": f"{label} (seed:{seed} N:{n})"
                })

# ---------------------------------------------------------
# Dynamic Runner
# ---------------------------------------------------------
def get_free_gpus():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], 
            text=True
        )
        free_gpus = []
        for line in output.strip().split('\n'):
            idx, mem = line.split(',')
            # If memory usage is < 1000 MB, we consider the GPU free
            if int(mem.strip()) < 1000:
                free_gpus.append(int(idx))
        return free_gpus
    except Exception as e:
        print(f"Error querying GPUs: {e}")
        return []

def main():
    print(f"Total baseline jobs generated: {len(job_queue)}")
    
    pending_jobs = []
    for job in job_queue:
        logfile = job["logfile"]
        if not os.path.exists(logfile) or os.path.getsize(logfile) < 500:
            pending_jobs.append(job)
            if os.path.exists(logfile):
                os.remove(logfile)
        else:
            print(f"Skipping {job['desc']} - logfile exists and has data")
            
    print(f"Jobs left to run: {len(pending_jobs)}")
    
    active_processes = {} # map GPU index -> subprocess
    
    while len(pending_jobs) > 0 or len(active_processes) > 0:
        finished_gpus = []
        for gpu, proc in active_processes.items():
            if proc.poll() is not None:
                finished_gpus.append(gpu)
                
        for gpu in finished_gpus:
            print(f"[GPU {gpu}] Finished a task.")
            del active_processes[gpu]
            
        free_gpus = get_free_gpus()
        truly_free = [g for g in free_gpus if g not in active_processes]
        
        while len(truly_free) > 0 and len(pending_jobs) > 0:
            gpu = truly_free.pop(0)
            job = pending_jobs.pop(0)
            
            print(f"[GPU {gpu}] Starting {job['desc']}")
            
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["WANDB_DIR"] = env.get("WANDB_DIR", "/scratch/cluster/jeffrey9/wandb_cache")
            env["WANDB_CACHE_DIR"] = env.get("WANDB_CACHE_DIR", "/scratch/cluster/jeffrey9/wandb_cache")
            env["PYTHONPATH"] = env.get("PYTHONPATH", "") + ":" + os.getcwd()
            env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            env["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"
            
            with open(job["logfile"], "a") as f:
                proc = subprocess.Popen(job["cmd"], env=env, stdout=f, stderr=subprocess.STDOUT)
            
            active_processes[gpu] = proc
            time.sleep(10)
            
        time.sleep(20)
        
    print("All baseline tasks complete!")

if __name__ == "__main__":
    main()
