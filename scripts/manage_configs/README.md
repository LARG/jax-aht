# `scripts/manage_configs/` — Benchmark Config Management

Scripts for setting up and auditing benchmark configurations across algorithms and tasks.

---

## Benchmark Setup Workflow

### Step 1 — Bake timesteps into sweep YAMLs (`update_sweep_timesteps.py`)

Reads the resolved timestep values from each algorithm's Hydra config and writes them as
fixed CLI args into the corresponding W&B param-sweep YAML. This ensures sweeps run with
the correct training budget and makes the budget explicit in the sweep file for reproducibility.

If a timestep key is already present in a sweep file with a **different** value, the script
prints a warning and skips rather than overwriting — resolve the discrepancy manually.

```bash
python scripts/manage_configs/update_sweep_timesteps.py trajedi
python scripts/manage_configs/update_sweep_timesteps.py fcp --dry-run
```

---

### Step 2 — Apply best hyperparameters (`apply_best_hparams.py`)

Fetches the best hyperparameter combination from a completed W&B sweep and writes it to
the corresponding algorithm config YAML. Sweep IDs must be registered in
`scripts/paper_vis/plot_globals.py` before running.

Always do a dry run first to confirm the reported heldout return matches expectations.

```bash
# Dry run — check best hparams and expected return without writing anything:
python scripts/manage_configs/apply_best_hparams.py \
    --task lbf/lbf_7x7_nolevels --algorithm trajedi --dry-run

# Apply:
python scripts/manage_configs/apply_best_hparams.py \
    --task lbf/lbf_7x7_nolevels --algorithm trajedi
```

---

### Step 3 — Standardize timesteps (`update_timesteps.py` + `report_timesteps.py`)

`update_timesteps.py` adjusts algorithm configs to hit a target total training budget for
each difficulty tier. `report_timesteps.py` prints a summary table so you can audit the
result.

```bash
# Set target budgets:
python scripts/manage_configs/update_timesteps.py teammate_generation/ \
    --easy-target 130M --hard-target 260M

python scripts/manage_configs/update_timesteps.py open_ended_training/ \
    --easy-target 195M --hard-target 390M --skip-algos open_ended_minimax paired

python scripts/manage_configs/update_timesteps.py ego_agent_training/ \
    --easy-target 11M --hard-target 23M

# Audit the results:
python scripts/manage_configs/report_timesteps.py \
    teammate_generation/ open_ended_training/ ego_agent_training/
```

---

## Other Scripts

- **`helpers.py`** — shared library (not run directly). Task lists, config resolution,
  value formatting, and timestep computation utilities used by the scripts above.

- **`report_timesteps.py`** — standalone audit tool; prints total timestep budgets for
  every algorithm × task combination. Useful at any point to verify configs are consistent.

- **`transfer_best_hparams.py`** — legacy script, no longer used.
