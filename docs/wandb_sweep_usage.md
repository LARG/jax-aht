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

The workflow above uses wandb's cloud controller which does not guarantee that an algorithm's current default hyperparameters
are among them. [scripts/hparam_search/sweep_controller.py](../scripts/hparam_search/sweep_controller.py) replaces
the cloud controller with a local one that schedules the defaults as the sweep's first
run, then falls back to the sweep yaml's own search method for everything after.

Agents are unaffected: `wandb agent` polls the backend for scheduled runs
regardless of who scheduled them, so `scripts/hparam_search/run_hparam_sweep.sh` works unchanged.

### Usage

```bash
# 1. Create the sweep. This reads the committed sweep yaml but does not modify it.
PYTHONPATH=. python scripts/hparam_search/sweep_controller.py create \
    <param_sweep_config>.yml

# 2. Run the controller. It prints the exact command to use, including
#    --entity/--project when the sweep is not in aht-project/aht-parameter-sweep.
screen -S sweepctl
PYTHONPATH=. python scripts/hparam_search/sweep_controller.py run <sweep_id>
# detach with C-a d; reattach with `screen -r sweepctl`

# 3. Launch agents, one per node, exactly as for a cloud-controlled sweep.
bash scripts/hparam_search/run_hparam_sweep.sh <sweep_id>
```

The controller **must** run in a screen or tmux session. It is the only thing scheduling
runs, so if it dies the agents sit idle. Restarting it is safe (see below).

Useful flags on `run`:

- `--poll-interval` — seconds between controller steps (default 15).
- `--force-seed` — schedule the defaults even if a matching run already exists.
- `--seed-only` — seed and exit without entering the scheduling loop.

### Installation

`wandb.controller` needs the `sweeps` package, i.e. the `wandb[sweeps]` extra, which is
not installed in `bench311`. Without it every entry point exits with an install hint.

```bash
pip install 'wandb[sweeps]==0.19.9'
```

### Pause handling

The canonical wandb controller loop is `while not tuner.done(): tuner.step()`. In wandb
0.19.9, `_WandbController.done()` returns `False` *only* for the `PREEMPTING`, `PENDING`,
and `RUNNING` states. A sweep paused from the UI is in `PAUSED`, so `done()` returns
`True` and the canonical loop **exits** — a pause silently kills scheduling for good.

This script checks the sweep state explicitly instead: `PAUSED` sleeps and keeps polling,
and only genuinely terminal states (`FINISHED`, `CANCELED`, `CRASHED`, `KILLED`) exit.
Unrecognised states are treated as "keep polling" and warned about, on the grounds that
exiting by mistake is the failure mode worth guarding against.

### Restart safety

`_WandbController` rebuilds all of its state from the backend when it is constructed, so
killing and restarting the controller resumes cleanly. To avoid seeding a *second*
default run on restart, the seeding step first scans the sweep's existing runs for one
whose swept parameters already match the defaults, and skips seeding if it finds one.

A run counts as matching regardless of its state, so a default run that crashed is not
re-seeded automatically. Use `--force-seed` if you want it re-run.
