# `scripts/manage_configs/` — Benchmark Config Management Scripts

This directory contains utility scripts for standardising the benchmark configuration
across algorithms and tasks. Run them in the order listed below.

---

## Scripts

### 1. `helpers.py`
**Shared library** — not run directly. Imported by the other scripts.

Provides:
- **`EASY_TASKS` / `HARD_TASKS`** — canonical difficulty-tier classification:
  - Easy: `lbf`, `overcooked-v1/cramped_room`, `overcooked-v1/asymm_advantages`
  - Hard: `overcooked-v1/coord_ring`, `overcooked-v1/counter_circuit`, `overcooked-v1/forced_coord`
- **`resolve_algo_config()`** — resolves Hydra `defaults:` chains to produce a merged config dict
- **`format_value()` / `format_timesteps()`** — consistent numeric formatting for YAML/CLI output
- **`format_human()`** / **`parse_human_timesteps()`** — K/M/B human-readable formatting and parsing
- **`compute_total_timesteps()`** — computes total training timestep budget from a resolved config dict
- **`compute_target_params()`** — computes which parameters to change to hit a target total timestep count
- **`round_sig()`** — rounds a float to N significant figures (default 3)

---

### 2. `update_timesteps.py`
Adjusts algorithm configs to achieve a **target total timestep budget** for each
difficulty tier. For each algorithm × task the script resolves the config, computes the
current total, and adjusts the minimal set of parameters to reach the target.

Parameter adjustment priorities (per algorithm family):

| Algorithm | Adjusted parameters |
|-----------|---------------------|
| ROTATE / open_ended_minimax | `TIMESTEPS_PER_ITER_PARTNER`, `TIMESTEPS_PER_ITER_EGO`, `NUM_OPEN_ENDED_ITERS` — scaled equally by √(target/current) |
| FCP | `PARTNER_POP_SIZE` (integer) |
| COLE | `TOTAL_TIMESTEPS_PER_ITERATION` |
| CoMeDi | `TOTAL_TIMESTEPS_PER_ITERATION` and `PARTNER_POP_SIZE` — scaled equally by √(target/current) |
| All others | `TOTAL_TIMESTEPS` |

`ego_train_algorithm.TOTAL_TIMESTEPS` is always kept fixed.
Targets accept K / M / B suffixes (e.g. `130M`, `1.3B`, `500K`) or plain numbers.
A summary table of updated totals is printed at the end (non-dry-run only).

```bash
python scripts/manage_configs/update_timesteps.py teammate_generation/ \
    --easy-target 130M --hard-target 260M

python scripts/manage_configs/update_timesteps.py open_ended_training/ \
    --easy-target 195M --hard-target 390M --skip-algos open_ended_minimax paired

python scripts/manage_configs/update_timesteps.py ego_agent_training/ \
    --easy-target 11M --hard-target 23M

# Preview changes without writing:
python scripts/manage_configs/update_timesteps.py teammate_generation/ \
    --easy-target 130M --hard-target 260M --dry-run
```

---

### 3. `update_sweep_timesteps.py`
Copies the resolved `TOTAL_TIMESTEPS` (and related keys) from algorithm Hydra configs into
the W&B param-sweep YAML files as fixed CLI args.

Run **once per entry point** after finalising the timestep budgets for each task.

```bash
python scripts/manage_configs/update_sweep_timesteps.py teammate_generation/
python scripts/manage_configs/update_sweep_timesteps.py ego_agent_training/
python scripts/manage_configs/update_sweep_timesteps.py open_ended_training/

# Preview changes without writing:
python scripts/manage_configs/update_sweep_timesteps.py teammate_generation/ --dry-run
```

---

### 4. `transfer_best_hparams.py`
Transfers the **non-timestep** hyperparameters (those listed in the W&B sweep
`parameters:` section) from the swept source task (`overcooked-v1/coord_ring`) to the
other Overcooked layouts, once the sweep completes and best values have been chosen.

Run **after** each algorithm's hyperparameter sweep finishes.

```bash
python scripts/manage_configs/transfer_best_hparams.py teammate_generation/

# Skip algorithms whose sweep is still running:
python scripts/manage_configs/transfer_best_hparams.py teammate_generation/ --skip-algos comedi

# Preview changes without writing:
python scripts/manage_configs/transfer_best_hparams.py teammate_generation/ --dry-run
```

---

### 5. `report_timesteps.py`
Prints a table of total training timestep budgets (in K / M / B) for every
algorithm × task combination across one or more entry points.

Supports `--skip-algos` to exclude specific algorithms from the table.
Useful for **auditing** that timestep budgets are consistent after running the
scripts above.

```bash
python scripts/manage_configs/report_timesteps.py teammate_generation/
python scripts/manage_configs/report_timesteps.py ego_agent_training/ teammate_generation/ open_ended_training/

# Exclude specific algorithms:
python scripts/manage_configs/report_timesteps.py open_ended_training/ --skip-algos open_ended_minimax paired
```

---

## Recommended Run Order

```
1.  update_timesteps.py       (set target timestep budgets for all algo × task combos)

2.  report_timesteps.py       (audit: confirm budgets look correct)

3.  update_sweep_timesteps.py (bake timesteps into W&B param-sweep YAMLs)

4.  [Run W&B sweeps for hyperparameter tuning on coord_ring]

5.  transfer_best_hparams.py  (after sweeps complete, propagate best hparams to other tasks)
```
