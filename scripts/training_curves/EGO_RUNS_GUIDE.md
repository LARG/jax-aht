# Investigating completed ego-algorithm wandb runs

A reference for how to find, fetch, and analyze completed
`neurips:benchmark` ego-training runs in the `aht-project/aht-benchmark`
wandb project. Covers `ppo_ego`, `liam_ego`, `meliba_ego`. The
plotting CLI under `scripts/training_curves/{ppo_ego,liam_ego,meliba_ego}/`
is already built — this doc is for ad-hoc analysis on top of the same data
sources.

## 1. Where to look

- **Wandb project**: `aht-project/aht-benchmark`
- **Tag**: `neurips:benchmark` — a Hydra `label` defined in
  `teammate_generation/configs/base_config_teammate.yaml:45` that flows into
  the wandb `tags` list at run init.
- **`config.algorithm.ALG`** values: `"ppo_ego"`, `"liam_ego"`,
  `"meliba_ego"`.
- **Source code** for what each algorithm logs:
  `ego_agent_training/ppo_ego.py`, `liam_ego.py`, `meliba_ego.py`. Search for
  `log_artifact` and `log_item` to see metric names + artifact saves.

## 2. Discover runs

### Preferred: existing helper

```python
from scripts.training_curves.common import find_benchmark_runs

runs = find_benchmark_runs(
    algorithm="ppo_ego",                  # or "liam_ego" / "meliba_ego"
    task="overcooked-v1/coord_ring",      # or "lbf/lbf_7x7_nolevels", etc.
    state="finished",                     # default
)
# runs is a list of wandb.apis.public.Run
```

### Equivalent raw API

```python
import wandb
api = wandb.Api()
runs = list(api.runs("aht-project/aht-benchmark", filters={
    "tags": "neurips:benchmark",
    "config.algorithm.ALG": "ppo_ego",
    "config.TASK_NAME": "overcooked-v1/coord_ring",
    "state": "finished",
}))
```

### Cross-tabulate everything by (algo, task)

```python
from collections import defaultdict
import wandb
api = wandb.Api()
runs = api.runs("aht-project/aht-benchmark",
                filters={"tags": "neurips:benchmark", "state": "finished"},
                per_page=300)
by_pair = defaultdict(list)
for r in runs:
    alg = (r.config.get("algorithm") or {}).get("ALG", "?")
    if alg in {"ppo_ego", "liam_ego", "meliba_ego"}:
        by_pair[(alg, r.config.get("TASK_NAME"))].append(r.id)
```

## 3. Disambiguating multiple runs per (algo, task)

Two knobs vary across runs at the same (algo, task):

1. **Teammate set** — fcp_teammates vs rotate_teammates. Look up the run id in
   `scripts/paper_vis/plot_globals.py:EGO_BENCHMARK_RUNS`, which maps
   `{task: {ego_algo: {teammate_type: run_id}}}`. Algorithm-name
   normalization quirk: that mapping uses `"ppo_ego"` (with suffix) but
   `"liam"`/`"meliba"` (without). Try both keys when looking up.
2. **Replicate** — some pairs have additional unidentified runs (no
   teammate-type entry in the mapping). Treat these as their own
   experimental conditions; key the disambiguation by run id when no label
   is known.

The plotting code's resolver lives in
`scripts/training_curves/ego_common.py:_build_run_id_to_teammate_type` —
look there for the canonical lookup logic.

## 4. Pull metrics

The full per-seed time series lives in the **`ego_train_run` artifact** —
**not** in `run.scan_history()`. Wandb-side metrics are pre-averaged across
seeds, so scan_history is useless for per-seed analysis. The artifact is an
orbax checkpoint produced by `save_train_run` in each ego algo's
`log_metrics`-style routine.

```python
from scripts.training_curves.common import fetch_train_run_metrics_cached

metrics = fetch_train_run_metrics_cached(
    run, artifact_kind="ego_train_run",
    entity="aht-project", project="aht-benchmark",
    cache_dir="results/figures/cache",
    reduce_per_update=True,   # collapse (rollout, num_envs) tail axes if present
    force_recompute=False,
)
```

Returns a `dict` mirroring the artifact's `out["metrics"]` subtree, all
arrays converted to `numpy.ndarray`.

### Auxiliary: heldout eval

Use the separate `heldout_eval_metrics` artifact if you want held-out
evaluations:

```python
from scripts.wandb_utils.wandb_cache import fetch_run_eval_metrics_cached
heldout = fetch_run_eval_metrics_cached(
    run.id, "aht-project", "aht-benchmark",
    cache_dir="results/figures/cache",
)
```

Some neurips:benchmark runs were logged with `log_train_out=False` /
similar — they will lack one or both artifacts. Always check
`run.logged_artifacts()` first if the call may fail.

## 5. Key shapes

For **standalone ego training** (the `ppo_ego` / `liam_ego` / `meliba_ego`
runs we care about, *not* the FCP-style partner+ego pipeline):

```
metrics["returned_episode_returns"]          shape (NUM_SEEDS, NUM_UPDATES)
metrics["eval_ep_last_info"]
       ["returned_episode_returns"]           shape (NUM_SEEDS, NUM_UPDATES, NUM_PARTNERS, NUM_EVAL_EPS, NUM_AGENTS)
metrics["actor_loss"|"value_loss"|"entropy_loss"|"avg_grad_norm"]
                                              shape (NUM_SEEDS, NUM_UPDATES)
```

`heldout_eval_metrics["returned_episode_returns"]` has shape
`(NUM_SEEDS, NUM_HELDOUT_AGENTS, NUM_EVAL_EPS, NUM_AGENTS)` for non-OEL
methods.

The shape note that matters most: the standalone ego runs are **2D**
`(NUM_SEEDS, NUM_UPDATES)`, *not* the 3D
`(NUM_SEEDS, NUM_EGO_TRAIN_SEEDS, NUM_UPDATES)` you get from the FCP-style
partner+ego pipeline. `extract_ego_curve` in `common.py` handles both.

## 6. Compute env-step axis

The wandb run config holds `algorithm.TOTAL_TIMESTEPS` for standalone ego
runs (top-level — *not* under `algorithm.ego_train_algorithm`, which is the
nested form used in the partner+ego pipeline). Use the helper:

```python
from scripts.training_curves.common import (
    get_config_value, make_env_steps_axis,
)
total = get_config_value(run.config, "algorithm.TOTAL_TIMESTEPS")
n_updates = metrics["returned_episode_returns"].shape[-1]
env_steps = make_env_steps_axis(n_updates, total)   # shape (n_updates,)
```

## 7. Quick example: per-seed training-return curve for one run

```python
import numpy as np
from scripts.training_curves.common import (
    find_benchmark_runs, fetch_train_run_metrics_cached,
    extract_ego_curve, get_config_value,
)

runs = find_benchmark_runs(algorithm="ppo_ego",
                           task="overcooked-v1/coord_ring")
run = runs[0]
metrics = fetch_train_run_metrics_cached(
    run, artifact_kind="ego_train_run", reduce_per_update=True,
)
total = get_config_value(run.config, "algorithm.TOTAL_TIMESTEPS")
curve = extract_ego_curve(metrics, total)
# curve.values: (NUM_SEEDS, NUM_UPDATES)
# curve.env_steps: (NUM_UPDATES,)
```

## 8. Caveats and gotchas

- **GPU OOM during artifact load**: orbax-on-GPU OOMs on the larger
  `ego_train_run` artifacts (cramped_room ego is 6+ GB once expanded). The
  helper sets `JAX_PLATFORMS=cpu` at import time of
  `scripts/training_curves/common.py`. If you import wandb/orbax before
  that, you may still hit GPU. Recipe: import `common` first.
- **Wandb artifact dedup cache fills the home-dir quota**: `common.py`
  redirects via `WANDB_CACHE_DIR=/tmp/wandb-cache-jyliu`. Don't override
  unless you know what you're doing.
- **Pre-2026-04-28 partner artifacts** (FCP cramped_room, BRDIV
  forced_coord, etc.) saved un-reduced trajectory metrics with shape
  `(seeds, pop, updates, rollout, envs)`. The `reduce_per_update=True` flag
  on the cache helper auto-detects 4-D+ leaves and applies `mask_and_mean`
  along the trailing axes. For ego artifacts we always pass
  `reduce_per_update=True` to be safe.
- **Wandb API timeouts**: occasionally `api.runs(...)` or `run.summary` can
  take >19s and time out. Retry, or pass a longer timeout via
  `wandb.Api(timeout=60)`.
- **Sweep summaries are not what you want**: `wandb_cache.fetch_sweep_cached`
  fetches summary rows for a hyperparameter sweep — it does **not** give you
  per-update history. For training curves use the artifact path above.

## 9. Where the cached data lives

After running the existing CLIs at least once, the cache layout is:

```
results/figures/cache/
├── train_run_metrics/
│   └── aht-project__aht-benchmark__<run_id>__<artifact_kind>[__<extras>].pkl
├── eval_metrics/
│   └── aht-project__aht-benchmark__<run_id>.pkl
└── run_configs/
    └── aht-project__aht-benchmark__<run_id>.json
```

The pickles store the post-reduce numpy metric trees. Loading from cache
is microseconds; missing-cache hits the wandb API.

## 10. Related plotting CLIs

If you just want plots, use the existing CLIs:

```bash
PYTHONPATH=. /scratch/cluster/jyliu/conda_envs/HANABI/bin/python \
    scripts/training_curves/ppo_ego/run.py --task overcooked-v1/coord_ring
```

(swap `ppo_ego` for `liam_ego` / `meliba_ego`). They drop PNGs and refresh
the `wandb_runs.json` / `wandb_runs.md` sidecars in the
`results/figures/training_curves/<algo>/` folder.
