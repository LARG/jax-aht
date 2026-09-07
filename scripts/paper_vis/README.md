# Paper Visualizations

All scripts must be run from the **repo root** with the conda environment activated:

```bash
conda activate bench311
```

wandb run/sweep IDs for all plots are stored in [plot_globals.py](plot_globals.py).

wandb downloads are cached in `results/figures/cache/` (gitignored; symlink it from worktrees).

To regenerate all figures after changing run IDs, run in order:
`recompute_best_returns.py --include_bc`, `benchmark_bar_charts.py` (unified and ego),
`plot_by_agent_type.py` (radar), `run_plot_sweep_distribution.sh`.

---

## Benchmark bar charts

`benchmark_bar_charts.py` plots normalized agent performance across tasks.
we always set `--use_best_returns_normalization` for all figures to obtain most accurate performance comparisons.

**Unified benchmark** (teammate-generation methods — FCP, BRDiv, LBRDiv, CoMeDi, COLE, TrajeDi):
```bash
PYTHONPATH=. python scripts/paper_vis/benchmark_bar_charts.py --plot_type unified --include_bc --filter_failed_seeds
```

**Ego benchmark** (ego-training methods — PPO, LIAM, MeLIBA):
```bash
PYTHONPATH=. python scripts/paper_vis/benchmark_bar_charts.py --plot_type ego --include_bc --filter_failed_seeds
```

**Key flags:**
- `--use_best_returns_normalization` (default true) — renormalize by the best observed return per heldout agent instead of the original per-agent bounds
- `--include_bc` (default false) — include the human proxy (BC) teammate in every cell, appending the separate BC eval from `BC_BENCHMARK_RUNS` for runs that lack it; prints a coverage report at the end
- `--filter_failed_seeds` (default false) — filter out failed seeds
- `--tasks lbf/lbf_7x7_nolevels overcooked-v1/cramped_room` — restrict to specific tasks
- `--force_recompute` — re-download from wandb and recompute everything
- `--save_dir PATH` — override output directory (default: `results/figures/`)

Figures are saved as PDFs to `results/figures/`.

Heldout partners are matched across runs by name, not index, so runs with
different heldout sets can be compared (see [heldout_partners.py](heldout_partners.py)).

---

## Performance bounds comparison

`plot_bounds_comparison.py` produces a stacked bar chart comparing the original
per-agent normalization bounds (from `evaluation/configs/global_heldout_settings.yaml`)
against the best-seen BR returns (from the cached best-returns files).

```bash
PYTHONPATH=. python scripts/paper_vis/plot_bounds_comparison.py
```

Only tasks with a cached best-returns file are plotted. The bottom bar segment
shows the original max bound; the stacked orange segment shows how much the
best-seen BR exceeds it.

---

## Best-returns cache

Performance bounds stored in `global_heldout_settings.yaml` are the original
normalization maxima used at evaluation time. `compute_best_returns.py` scans
all benchmark runs for each task and computes the highest return actually
observed for each heldout agent, caching results in
`results/figures/cache/best_returns/<task>.json`. Pass `--include_bc` to also
scan the separate BC eval runs.

To force recomputation from locally cached wandb artifacts (re-downloads from
wandb only for runs not yet cached locally):

```bash
PYTHONPATH=. python scripts/paper_vis/recompute_best_returns.py

# Restrict to specific tasks:
PYTHONPATH=. python scripts/paper_vis/recompute_best_returns.py \
    --tasks lbf/lbf_7x7_nolevels overcooked-v1/cramped_room
```

Runs evaluated against an older, smaller heldout set are automatically skipped
with a warning.

---

## Hyperparameter sweep plots

`run_plot_sweep_distribution.sh` generates distribution plots for hyperparameter
sweeps. Sweep IDs are stored in `plot_globals.py` under `HYPERPARAM_SWEEPS`.

```bash
bash scripts/paper_vis/run_plot_sweep_distribution.sh
```

Each point is one hyperparameter setting (mean over seeds). By default only the
140 settings sampled by `scripts/manage_configs/apply_best_hparams.py` (seed 0)
are shown; use `--max-hparams 0` to plot all of them.

Figures are saved to `results/figures/`.
