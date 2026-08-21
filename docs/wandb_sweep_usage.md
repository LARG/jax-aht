# WandB Sweep Usage

## Overview
This guide explains how to run hyperparameter sweeps using Weights & Biases.

## Usage

### Step 1: Initialize the Sweep
We use the teammate generation entry point and the FCP algorithm / LBF environment as an example.

From the repository root, run:

```
PYTHONPATH=<PATH_TO_REPO_DIR>/jax-aht XLA_PYTHON_CLIENT_PREALLOCATE="false" wandb sweep teammate_generation/param_sweep/fcp/lbf/param_sweep.yml
```

This will create a sweep and output a sweep ID like: `entity/project/sweep_id`

### Step 2: Run Sweep Agents
Launch one or more agents to execute the sweep runs:

```
PYTHONPATH=<PATH_TO_REPO_DIR>/jax-aht XLA_PYTHON_CLIENT_PREALLOCATE="false" wandb agent <sweep_id>
```

If using the Bayesian or random sweep, you can limit the total number of runs by adding the `--count <num_runs>` flag:

```
PYTHONPATH=<PATH_TO_REPO_DIR>/jax-aht XLA_PYTHON_CLIENT_PREALLOCATE="false" wandb agent <sweep_id> --count 100
```

### Step 3: Monitor Results
View your sweep results at:
```
https://wandb.ai/<entity>/<project>/sweeps/<sweep_id>
```

## Seeding a Sweep With the Default Hyperparameters

The workflow above uses wandb's cloud controller, which does not guarantee that an
algorithm's current default hyperparameters are ever tried.
[scripts/hparam_search/sweep_controller.py](../scripts/hparam_search/sweep_controller.py)
replaces it with a local controller that schedules the defaults (read from the Hydra
configs) as the sweep's first run, then falls back to the sweep yaml's own search method.

Agents are unaffected: `wandb agent` polls the backend for scheduled runs regardless of
who scheduled them, so `scripts/hparam_search/run_hparam_sweep.sh` works unchanged.

### Installation

Needs the `wandb[sweeps]` extra:

```bash
pip install 'wandb[sweeps]==0.19.9' 'narwhals==1.33.0'
```

The `narwhals` pin matters: without it pip resolves scikit-learn 1.9, which upgrades
`narwhals` out from under `plotly`. With it the install is purely additive.

### Usage

```bash
# 1. Create the sweep. This reads the committed sweep yaml but does not modify it.
PYTHONPATH=. python scripts/hparam_search/sweep_controller.py create \
    <param_sweep_config>.yml

# 2. Run the controller. It prints the exact command to use, including
#    --entity/--project when the sweep is not in aht-project/aht-parameter-sweep.
screen -S sweepctl
PYTHONPATH=. python scripts/hparam_search/sweep_controller.py run <sweep_id> --max-pending 8
# detach with C-a d; reattach with `screen -r sweepctl`

# 3. Launch agents, one per node, exactly as for a cloud-controlled sweep.
bash scripts/hparam_search/run_hparam_sweep.sh <sweep_id>
```

The controller **must** run in a screen or tmux session: it is the only thing scheduling
runs, so if it dies the agents go idle. Restarting it is safe and resumes without
re-seeding the defaults.

Flags on `run`:

- `--max-pending` — suggestions kept queued for agents (default 4). Set it to roughly your
  agent count. wandb's own controller caps this at 1, which makes agents wait for each
  other to reach `wandb.init` — minutes, for a jax-aht run. Only applies to `random`;
  grid and bayes would return duplicates within a step, so they stay capped at 1.
- `--poll-interval` — seconds between controller steps (default 15).
- `--force-seed` — schedule the defaults even if a matching run already exists. Needed to
  re-run defaults that crashed, since a match counts regardless of run state.
- `--seed-only` — seed and exit without entering the scheduling loop.

### If agents are idle

The controller prints its queue depth and run counts each step. Check in this order:

1. **Are the agents alive?** Most commonly they are not — a slurm job that hit its
   walltime kills every agent at once. `squeue -u $USER` on the cluster running the
   agents, not the one running the controller.
2. **Is the sweep paused?** The controller holds rather than exiting on `PAUSED`, and says
   so each poll. Resume it in the UI.
3. **Is the queue full?** Suggestions sitting outstanding with agents alive means the
   backend is not handing them out; restarting the controller resets the queue.

Note that the controller's own status line reports at most 10 runs — wandb's sweep query
does not paginate — so `Runs: 10` says nothing about the size of the sweep. The controller
does not rely on that view for pool accounting; it releases a queue slot as soon as the
backend assigns the suggestion a run. 