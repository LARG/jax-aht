# Training Curves — Planning Doc

## Context

We want per-algorithm training-curve plots for the NeurIPS benchmark runs (wandb tag/label `neurips:benchmark`). The existing `scripts/paper_vis/` tooling deals with **end-of-training summary metrics and heldout eval artifacts** — it does not fetch time-series. This new directory is for the time-series side: scrape wandb run histories, cache them, and plot them.

Algorithms in eventual scope:
- Teammate generation: **FCP, CoMeDi, BRDIV, LBRDIV** (start here), then COLE, ROTATE, TrajeDI
- Ego: **PPO ego** (and any others)

Start with the easiest: **FCP**. Get the end-to-end pipeline working there before generalizing.

## Curves to plot (FCP first)

| Curve | wandb key | Where logged |
| --- | --- | --- |
| Partner training return | `Train/Partner_returned_episode_returns` | `teammate_generation/fcp.py:105` (one per env metric in `get_metric_names`) |
| Ego training return | `Eval/EgoReturn` | `teammate_generation/train_ego.py:113` |

For CoMeDi / BRDIV / LBRDIV the partner-side curve names differ — they log `Eval/AvgSPReturnCurve` and `Eval/AvgXPReturnCurve` instead of `Train/Partner_*` (see `BRDiv.py:790-791`, `CoMeDi.py:1117-1119`, `LBRDiv.py:1036-1037`). Ego curve is the same `Eval/EgoReturn`. We'll formalize per-algo metric maps once FCP is working.

## Existing infra to reuse

- `scripts/wandb_utils/wandb_cache.py`
  - `fetch_sweep_cached()` — pattern for cache-on-disk-as-pickle
  - `fetch_run_eval_metrics_cached()` — per-run artifact caching pattern; mirror its layout for history caching
- `scripts/paper_vis/plot_globals.py`
  - `ENTITY = "aht-project"`, `BENCHMARK_PROJECT = "aht-benchmark"`
  - `METHOD_TO_DISPLAY_NAME`, `TASK_TO_DISPLAY_NAME`, `TASK_TO_PLOT_TITLE`
  - `OEL_METHODS = ["rotate"]` — relevant later for ROTATE
- `common/plot_utils.get_metric_names(env_name)` — env-to-metric-name map (e.g. `returned_episode_returns` for LBF/Overcooked)

## How `neurips:benchmark` is applied & queried

It's a Hydra `label` (see `teammate_generation/experiments.sh:5`, `base_config_teammate.yaml:34,45`) which gets injected into the wandb `tags` list. So we query via:

```python
api.runs(
    f"{ENTITY}/{BENCHMARK_PROJECT}",
    filters={
        "tags": "neurips:benchmark",
        "config.algorithm.ALG": "fcp",
        "config.TASK_NAME": "overcooked-v1/coord_ring",
        "state": "finished",
    },
)
```

Validate filter shape against the wandb MongoDB-style query syntax — may need `$in` etc.

## Proposed directory layout

```
scripts/training_curves/
  PLAN.md                       (this file)
  __init__.py
  common.py                     (shared: wandb run-history fetcher + cache, alignment, smoothing)
  fcp/
    __init__.py
    fetch.py                    (run-spec resolution + history download)
    plot.py                     (matplotlib plotting)
    run.py                      (CLI entrypoint: fetch + plot for a task)
  comedi/  ...                  (added once FCP works)
  brdiv/   ...
  lbrdiv/  ...
```

Per-algorithm folder lets us specialize metric keys, axis labels, and any algo-specific quirks (e.g. CoMeDi's growing partner pool → `update_step` indexing). Shared infra goes in `common.py`.

Cache files live under the existing `results/figures/cache/` tree (consistent with `wandb_cache.DEFAULT_CACHE_DIR`):

```
results/figures/cache/run_history/{entity}__{project}__{run_id}.pkl
```

Output figures go under `results/figures/training_curves/<algo>/<task>.{png,pdf}`.

## Phase 1: FCP end-to-end

**Order matters**: do step 1 first as a one-shot exploration before writing fetcher abstractions. The format wandb returns for these specific runs is the source of truth — don't guess.

1. **Inspect wandb data format first (LBF)** — *before writing any abstraction*:
   - Pick one FCP `neurips:benchmark` run on `lbf/lbf_7x7_nolevels` (find via tag query against `aht-project/aht-benchmark`).
   - Open a throwaway script / interactive session and call:
     - `run.config` — confirm `algorithm.NUM_SEEDS`, `algorithm.NUM_ENVS`, `algorithm.NUM_STEPS`, `algorithm.TOTAL_TIMESTEPS`, `algorithm.TRAIN_SEED`, ego-train counterparts.
     - `list(run.summary.keys())` — see what's actually present (esp. whether `Train/Partner_*` and `Eval/EgoReturn` are both there for an FCP run).
     - `run.scan_history(keys=["Train/Partner_returned_episode_returns", "_step", "train_step"])` — pull a few rows and inspect.
     - Same for `Eval/EgoReturn`.
   - Write down (in a short scratch note added to this PLAN) the answers to:
     - Are seeds vmapped within a single run, or one run per seed? If vmapped, is the metric value a scalar (already averaged) or an array per seed? Look at the actual value type.
     - What's the `train_step` axis for the partner curve vs the ego curve — same counter or two independent counters?
     - Does `_step` line up with `train_step`, or is it a separate global step?
     - How do we convert `train_step` → env steps? Likely `env_steps = train_step * NUM_ENVS * NUM_STEPS`, but verify against config keys per algorithm (FCP partners use `algorithm.NUM_ENVS / NUM_STEPS`; ego uses `algorithm.ego_train_algorithm.NUM_ENVS / NUM_STEPS` or similar).
     - Same for the ego curve — does its x-axis multiplier come from a different config subtree?
   - Only after these answers are written down, proceed to step 2.

2. **Add history fetcher** in `scripts/training_curves/common.py`:
   - `fetch_run_history_cached(run_id, entity, project, keys: list[str], cache_dir, force_recompute)` → `pd.DataFrame`
   - Use `run.scan_history(keys=keys)` for full fidelity (no sampling). Persist as pickle by `(entity, project, run_id, keys-hash)`. Re-fetch on cache miss only.
   - Mirror the validation + cache-path conventions from `wandb_cache.py`.

3. **Add run discovery** in `scripts/training_curves/common.py`:
   - `find_benchmark_runs(algorithm, task, tag="neurips:benchmark", entity, project)` → list of run IDs.
   - Filter: `state == "finished"`, tag contains `neurips:benchmark`, `config.algorithm.ALG == algorithm`, `config.TASK_NAME == task`.
   - Return run IDs only — let callers fetch config separately via `fetch_run_config_cached`.

4. **Add env-step conversion** in `scripts/training_curves/common.py`:
   - `train_step_to_env_steps(train_step, run_config, axis: Literal["partner","ego"])` → int/array.
   - Reads the appropriate `NUM_ENVS / NUM_STEPS` (or `ROLLOUT_LENGTH`) from the config based on which curve we're plotting. Exact key paths come from step 1's analysis.

5. **Add `fcp/fetch.py`**:
   - For a given task, find all FCP `neurips:benchmark` runs, pull histories for the two metric keys, return per-seed arrays + env-step x-axis.
   - Note: partner and ego curves likely live on different x axes — keep them separate.

6. **Add `fcp/plot.py`**:
   - Two-panel figure (or two separate figures): partner training return on the left/top, ego training return on the right/bottom.
   - **One line per seed, no aggregation** — distinguish seeds by color or alpha.
   - Reuse `plot_globals` fontsize constants and display-name maps. X-axis label: "Environment Steps".

7. **Add `fcp/run.py`** CLI:
   - `python -m scripts.training_curves.fcp.run --task lbf/lbf_7x7_nolevels [--force-recompute]`
   - On first run: hits wandb, caches, plots. On rerun: serves from cache, replots.

## Task ordering

Within each phase, **LBF first**, then Overcooked tasks. Reason: per the user, LBF is the priority environment for the first cut.

## Phase 2: extend to CoMeDi / BRDIV / LBRDIV

- Copy FCP folder structure; swap metric keys to `Eval/AvgSPReturnCurve` / `Eval/AvgXPReturnCurve`; keep `Eval/EgoReturn` for the ego panel.
- Decide whether SP and XP go on the same axes (likely yes — they share x and units).
- CoMeDi adds partners over training; the curve is logged by `update_step` so each partner-pop-size gets its own segment. Confirm semantics before assuming a single contiguous curve.

## Phase 3: ROTATE / COLE / TrajeDI / PPO ego

Defer until phase 1–2 is solid. ROTATE is OEL (5D metrics) and will need its own treatment.

## Status (2026-05-03)

Built:
- `scripts/training_curves/{__init__.py,common.py,PLAN.md}` (this file)
- `scripts/training_curves/fcp/{__init__.py,fetch.py,plot.py,run.py}` — works end-to-end on all 5 Overcooked tasks; per-seed partner training return + ego training return.
- `scripts/training_curves/lbrdiv/{...}` — works on `coord_ring`; SP/XP partner curves limited to NUM_CHECKPOINTS eval points (renders as scatter+line).
- `scripts/training_curves/brdiv/{...}` — same structure as LBRDIV; running for all 5 Overcooked tasks now.

Notable adaptations along the way:
- `JAX_PLATFORMS=cpu` set at common.py import time — orbax restore on GPU OOMs on the larger train_run artifacts (cramped_room ego is 6+ GB once expanded).
- `WANDB_CACHE_DIR=/tmp/...` — the home-dir quota fills up if wandb's artifact dedup cache is left at default `~/.cache/wandb/`.
- `reduce_per_update` flag on `fetch_train_run_metrics_cached` — older partner artifacts (FCP cramped_room, asymm_advantages) saved the un-reduced trajectory metric of shape `(NUM_SEEDS, POP, NUM_UPDATES, ROLLOUT, NUM_ENVS)` instead of pre-reducing via `mask_and_mean`. The reducer auto-detects and collapses the trailing two axes so the cached pickle is tractable. (Commit `246d297` "reduced metrics to scalars" on 2026-04-28 fixed this for newer runs.)

Output paths:
- Plots: `results/figures/training_curves/<algo>/<task>__<run_id>.png`
- Cache: `results/figures/cache/train_run_metrics/<entity>__<project>__<run_id>__<artifact_kind>.pkl`

Remaining work:
- CoMeDi: structurally different artifact (SP/XP at top-level `outs["last_ep_infos_sp"]`, etc., NOT under `out["metrics"]`). Will need a small extension to `fetch_train_run_metrics_cached` to pull additional top-level keys, plus a CoMeDi-specific `fetch.py` aware of the (pop_size - 1) progressive-add axis.
- Refactor common partner+ego patterns once 4 algos exist.
- LBF runs: revisit once running CoMeDi/LBRDIV/TrajeDI LBF jobs finish (state=running as of 2026-05-03).

## Decisions (locked in)

1. **Tasks**: LBF (`lbf/lbf_7x7_nolevels`) first for every algorithm, then Overcooked tasks.
   - **Reality check from inspection**: as of 2026-05-03 there are *no* `neurips:benchmark` FCP/BRDIV runs on LBF yet — only Overcooked. CoMeDi/LBRDIV/TrajeDI on LBF are state=running. So **FCP must start on Overcooked** until LBF FCP runs land. Pipeline is task-agnostic so swapping in LBF later is one CLI flag change.
2. **Aggregation**: none — plot every seed as its own line.
3. **X-axis**: environment steps, computed as `env_steps[i] = (i+1) * TOTAL_TIMESTEPS / NUM_UPDATES` where `NUM_UPDATES = metrics["returned_episode_returns"].shape[-1]`. This avoids having to dig out NUM_ENVS / ROLLOUT_LENGTH at the right config path (which differ between partner and ego subtrees).
4. **Run-per-task model**: one wandb run per (algo, task), with `NUM_SEEDS=5` vmapped inside. Confirmed across all 27 `neurips:benchmark` runs.
5. **Dedup**: only one run per (algo, task) exists — no dedup needed for this pass.

## Findings from Phase 1 inspection (FCP on overcooked-v1/coord_ring, run `ikrlj1qe`)

**Wandb scan_history is NOT useful for per-seed curves.** Both `fcp.py:97-105` and `train_ego.py:88-89,98,111-113` average across the seed axis *before* calling `logger.log_item`. So everything in `scan_history` is already a mean over seeds.

**Per-seed curves come from the orbax-checkpointed train_run artifacts** logged via `logger.log_artifact`:

- **`saved_train_run`** (~282 MB) — partner training output. `out["metrics"]["returned_episode_returns"]` has shape `(NUM_SEEDS, PARTNER_POP_SIZE, NUM_PARTNER_UPDATES)` = `(5, 82, 156)` for FCP/coord_ring. Mean over the partner-pop axis (axis=1) gives the per-seed partner-training curve.
- **`ego_train_run`** (~602 MB) — ego training output. Two relevant arrays:
  - `out["metrics"]["returned_episode_returns"]` shape `(NUM_SEEDS, NUM_EGO_TRAIN_SEEDS, NUM_EGO_UPDATES)` = `(5, 1, 585)` — training return, already a scalar per update.
  - `out["metrics"]["eval_ep_last_info"]["returned_episode_returns"]` shape `(5, 1, 585, NUM_PARTNERS, NUM_EVAL_EPS, NUM_AGENTS_PER_GAME)` = `(5, 1, 585, 82, 20, 2)` — this is what gets averaged over (partners, eps, agents) and logged as `Eval/EgoReturn`. Same shape we want, average axes -3..-1 to get per-seed.

**Env-step conversion** (from `algorithm.TOTAL_TIMESTEPS` and `NUM_UPDATES = shape[-1]`):
- Partner: per-partner env steps. `TOTAL_TIMESTEPS = 4_000_000` (per partner), `NUM_UPDATES = 156` → 25,641 env-steps per partner update.
- Ego: total env steps. `algorithm.ego_train_algorithm.TOTAL_TIMESTEPS = 60_000_000`, `NUM_UPDATES = 585` → 102,564 env-steps per ego update.

**Other algorithms — partial reads from source code.**

- **BRDIV** (`BRDiv.py:769-797`) and **LBRDIV** (`LBRDiv.py:1017-1037`) have identical artifact structure. Partner curves come from `metrics["eval_ep_last_info"]["returned_episode_returns"]` shape `(NUM_SEEDS, NUM_PARTNER_UPDATES, POP_SIZE^2, NUM_EVAL_EPS, NUM_AGENTS)`. SP slice = pairs where `conf_id == br_id` (use `np.repeat / np.tile` to reconstruct the mask). Mean over (pair, eps, agents) → per-seed curves. Ego curve identical to FCP (read from `ego_train_run`).
- **CoMeDi** (`CoMeDi.py:1091-1120`) is structurally different — SP/XP arrays live at *top-level* `outs["last_ep_infos_sp"]` and `outs["last_ep_infos_xp"]`, NOT under `outs["metrics"]`. Pop_size dim is `pop_size - 1` (excludes initial policy). XP array has an extra pop_size axis (each new policy evaluated against all previous). Eval is only logged at `NUM_CHECKPOINTS` points along the update axis, so the curve will be sparse/stepped if plotted naively. Defer until BRDIV/LBRDIV are working.

## Available `neurips:benchmark` runs (snapshot 2026-05-03)

```
fcp:    overcooked × 5 tasks  (LBF: none yet)
brdiv:  overcooked × 5 tasks  (LBF: none yet)
lbrdiv: overcooked × 2 tasks (coord_ring, cramped_room) + LBF (RUNNING)
comedi: LBF (RUNNING) only
cole:   overcooked/cramped_room only
rotate: overcooked × 3 tasks
trajedi: LBF + overcooked/coord_ring (both RUNNING)
ppo_ego/liam_ego/meliba_ego: overcooked/coord_ring only
```

Once running LBF jobs finish, re-snapshot before generalizing.

## Verification

For FCP after Phase 1:
- `python -m scripts.training_curves.fcp.run --task overcooked-v1/coord_ring`
- Expect: PNG/PDF in `results/figures/training_curves/fcp/overcooked-v1__coord_ring.png` showing both curves with seed dispersion.
- Cache populated under `results/figures/cache/run_history/`.
- Re-run: no wandb call, regenerates from cache.
