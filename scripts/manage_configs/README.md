# `temp/` — Benchmark Setup Scripts

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
- **`TIMESTEP_TASK_MAP`** — source → target task mapping for timestep transfer
- **`resolve_algo_config()`** — resolves Hydra `defaults:` chains to produce a merged config dict
- **`format_value()` / `format_timesteps()`** — consistent numeric formatting for YAML/CLI output
- **`compute_total_timesteps()`** — computes total training timestep budget from a resolved config dict

---

### 2. `update_sweep_timesteps.py`
Copies the resolved `TOTAL_TIMESTEPS` (and related keys) from algorithm Hydra configs into
the W&B param-sweep YAML files as fixed CLI args.

Run **once per entry point** after finalising the timestep budgets for each task.

```bash
python temp/update_sweep_timesteps.py teammate_generation/
python temp/update_sweep_timesteps.py ego_agent_training/
python temp/update_sweep_timesteps.py open_ended_training/

# Preview changes without writing:
python temp/update_sweep_timesteps.py teammate_generation/ --dry-run
```

---

### 3. `update_timesteps.py`
Propagates timestep hyperparameters from the canonical source task to all other tasks
in the same difficulty tier:

| Tier | Source | Targets |
|------|--------|---------|
| Easy | `lbf` | `overcooked-v1/cramped_room`, `overcooked-v1/asymm_advantages` |
| Hard | `overcooked-v1/coord_ring` | `overcooked-v1/counter_circuit`, `overcooked-v1/forced_coord` |

Timestep keys transferred (any subset present in the source config):
`TOTAL_TIMESTEPS`, `TOTAL_TIMESTEPS_PER_ITERATION`, `TIMESTEPS_PER_ITER_PARTNER`,
`TIMESTEPS_PER_ITER_EGO`, `NUM_OPEN_ENDED_ITERS`, `ego_train_algorithm.TOTAL_TIMESTEPS`

Run **once per entry point** after setting the desired timestep budgets in the source
task configs (lbf and coord_ring).

```bash
python scripts/manage_configs/update_timesteps.py teammate_generation/
python scripts/manage_configs/update_timesteps.py ego_agent_training/
python scripts/manage_configs/update_timesteps.py open_ended_training/

# Skip algorithms whose configs are not yet finalised:
python scripts/manage_configs/update_timesteps.py open_ended_training/ --skip-algos rotate

# Preview changes without writing:
python scripts/manage_configs/update_timesteps.py teammate_generation/ --dry-run
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

Useful for **auditing** that timestep budgets are consistent after running the
transfer scripts above.

```bash
python scripts/manage_configs/report_timesteps.py teammate_generation/
python scripts/manage_configs/report_timesteps.py ego_agent_training/ teammate_generation/ open_ended_training/
```

---

## Recommended Run Order

```
1.  Edit source task configs (lbf.yaml, overcooked-v1/coord_ring.yaml) with desired
    timestep budgets for each algorithm.

2.  update_timesteps.py   (propagate timesteps to other tasks in the same tier)

3.  report_timesteps.py     (audit: confirm budgets look correct)

4.  update_sweep_timesteps.py  (bake timesteps into W&B param-sweep YAMLs)

5.  [Run W&B sweeps for hyperparameter tuning on coord_ring]

6.  transfer_best_hparams.py   (after sweeps complete, propagate best hparams)
```
