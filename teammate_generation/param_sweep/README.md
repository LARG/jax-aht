# WandB Sweep Usage

## Overview
This guide explains how to run hyperparameter sweeps using Weights & Biases.

## Usage

### Step 1: Initialize the Sweep
From the repository root, run:

```
PYTHONPATH=<PATH_TO_REPO_DIR>/jax-aht XLA_PYTHON_CLIENT_PREALLOCATE="false" wandb sweep teammate_generation/param_sweep/fcp/param_sweep.yml
# PYTHONPATH=/home/rolando/GitHub/jax-aht XLA_PYTHON_CLIENT_PREALLOCATE="false" wandb agent fernandezr-the-university-of-texas-at-austin/PARAM_SWEEP-teammate_generation/8frmov0u
```

This will create a sweep and output a sweep ID like: `entity/project/sweep_id`

### Step 2: Run Sweep Agents
Launch one or more agents to execute the sweep runs:

```
PYTHONPATH=<PATH_TO_REPO_DIR>/jax-aht XLA_PYTHON_CLIENT_PREALLOCATE="false" wandb agent <sweep_id>
```

Or with your entity/project:

```
PYTHONPATH=<PATH_TO_REPO_DIR>/jax-aht XLA_PYTHON_CLIENT_PREALLOCATE="false" wandb agent <entity>/<project>/<sweep_id>
```

### Step 3: Monitor Results
View your sweep results at:
```
https://wandb.ai/<entity>/<project>/sweeps/<sweep_id>
```