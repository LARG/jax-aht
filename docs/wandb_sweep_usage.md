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

If using the Bayesian sweep, you can limit the total number of runs by adding the `--count <num_runs>` flag:

```
PYTHONPATH=<PATH_TO_REPO_DIR>/jax-aht XLA_PYTHON_CLIENT_PREALLOCATE="false" wandb agent <sweep_id> --count 100
```

### Step 3: Monitor Results
View your sweep results at:
```
https://wandb.ai/<entity>/<project>/sweeps/<sweep_id>
```