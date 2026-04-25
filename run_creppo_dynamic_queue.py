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

EVAL_SEEDS = [72128, 721280, 721281, 721282, 721283]
DIFF_TRAIN_SEEDS = [174464134, 1744641340, 1744641341, 1744641342, 1744641343]

# Create logs directory
os.makedirs("logs", exist_ok=True)
os.makedirs(os.environ.get("WANDB_DIR", "/scratch/cluster/jeffrey9/wandb_cache"), exist_ok=True)

# ---------------------------------------------------------
# Build the Queue
# ---------------------------------------------------------
job_queue = []

def add_batch(task_prefix, label_prefix, social_law_label, train_seed, eval_seed, use_same_seed):
    for n in [2, 3, 4, 5]:
        task = f"continuous/{task_prefix}_{n}_agent"
        label = f"{label_prefix}_{n}_agent"
        
        logfile = f"logs/{label}_tr{train_seed}_ev{eval_seed}_N{n}.out"
        
        # Build command
        cmd = [
            VENV_PYTHON, "social_laws/run_w_best_case.py",
            f"task={task}",
            "algorithm=creppo/continuous/coop_recon",
            f"algorithm.TRAIN_SEED={train_seed}",
            f"algorithm.EVAL_SEED={eval_seed}",
            f"algorithm.USE_SAME_SEED={use_same_seed}",
            "algorithm.FIXED_EVAL=true",
            "algorithm.ALPHA_VERIFICATION=true",
            f"NUM_EXPT_AGENTS={n}",
            f"label={label}",
            "logger.project=aht-benchmark",
            "logger.entity=jeffreychen287-the-university-of-texas-at-austin",
            "logger.mode=online"
        ]
        
        if social_law_label != "none":
            cmd.append(f"social_law_label={social_law_label}")
            
        job_queue.append({
            "cmd": cmd,
            "logfile": logfile,
            "desc": f"{label} (tr:{train_seed} ev:{eval_seed} N:{n})"
        })

# Generate all 160 jobs
for i in range(len(EVAL_SEEDS)):
    ev = EVAL_SEEDS[i]
    tr_diff = DIFF_TRAIN_SEEDS[i]
    
    # NO LAW
    add_batch("coop_recon_compare_no_law", "creppo_no_law", "none", ev, ev, "true")
    add_batch("coop_recon_compare_no_law", "creppo_no_law_gen", "none", tr_diff, ev, "false")
    
    # LAW 0.0
    add_batch("coop_recon_compare_law_0.0", "creppo_law_0.0", "law_0.0", ev, ev, "true")
    add_batch("coop_recon_compare_law_0.0", "creppo_law_0.0_gen", "law_0.0", tr_diff, ev, "false")
    
    # LAW 0.1
    add_batch("coop_recon_compare_law_0.1", "creppo_law_0.1", "law_0.1", ev, ev, "true")
    add_batch("coop_recon_compare_law_0.1", "creppo_law_0.1_gen", "law_0.1", tr_diff, ev, "false")
    
    # LAW 0.2
    add_batch("coop_recon_compare_law", "creppo_law_0.2", "law_0.2", ev, ev, "true")
    add_batch("coop_recon_compare_law", "creppo_law_0.2_gen", "law_0.2", tr_diff, ev, "false")

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
    print(f"Total jobs generated: {len(job_queue)}")
    
    # Filter out jobs that already have a log file
    # This prevents rerunning things the bash script already started
    pending_jobs = []
    for job in job_queue:
        if not os.path.exists(job["logfile"]):
            pending_jobs.append(job)
        else:
            print(f"Skipping {job['desc']} - logfile exists")
            
    print(f"Jobs left to run: {len(pending_jobs)}")
    
    active_processes = {} # map GPU index -> subprocess
    
    while len(pending_jobs) > 0 or len(active_processes) > 0:
        # Check for finished processes
        finished_gpus = []
        for gpu, proc in active_processes.items():
            if proc.poll() is not None:
                finished_gpus.append(gpu)
                
        for gpu in finished_gpus:
            print(f"[GPU {gpu}] Finished a task.")
            del active_processes[gpu]
            
        # Get free GPUs
        free_gpus = get_free_gpus()
        
        # Don't assign a task to a free GPU if we supposedly have a process running on it
        # (This is a safety check in case memory drops temporarily during setup)
        truly_free = [g for g in free_gpus if g not in active_processes]
        
        # Dispatch jobs to truly free GPUs
        while len(truly_free) > 0 and len(pending_jobs) > 0:
            gpu = truly_free.pop(0)
            job = pending_jobs.pop(0)
            
            print(f"[GPU {gpu}] Starting {job['desc']}")
            
            # Setup environment
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["WANDB_DIR"] = env.get("WANDB_DIR", "/scratch/cluster/jeffrey9/wandb_cache")
            env["WANDB_CACHE_DIR"] = env.get("WANDB_CACHE_DIR", "/scratch/cluster/jeffrey9/wandb_cache")
            env["PYTHONPATH"] = env.get("PYTHONPATH", "") + ":" + os.getcwd()
            env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            env["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"
            
            # Launch
            with open(job["logfile"], "a") as f:
                proc = subprocess.Popen(job["cmd"], env=env, stdout=f, stderr=subprocess.STDOUT)
            
            active_processes[gpu] = proc
            
            # Give JAX a few seconds to preallocate memory before launching next
            time.sleep(10)
            
        time.sleep(20) # Polling interval
        
    print("All tasks complete!")

if __name__ == "__main__":
    main()
