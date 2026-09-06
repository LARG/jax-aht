# Paper Visualizations

All scripts must be run from the **repo root** with the conda environment activated:

```bash
conda activate bench311
```

wandb run/sweep IDs for all plots are stored in [plot_globals.py](plot_globals.py).

Downloaded wandb artifacts and computed stats are cached under `results/figures/cache/`
(gitignored). From a worktree, symlink it to the main checkout's cache to avoid re-downloading.

To regenerate all paper figures after changing run IDs in `plot_globals.py`, run in order:

1. `recompute_best_returns.py --include_bc` (normalization bounds may shift)
2. `benchmark_bar_charts.py` for `--plot_type unified` and `--plot_type ego` (see below)
3. `plot_by_agent_type.py` (radar chart, `by_agent_type_br_norm.pdf`)
4. `run_plot_sweep_distribution.sh`

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
- `--include_bc` (default false) — ensure the human proxy (BC) teammate is part of every method's heldout set. iclr26-era runs already evaluate against `human_proxy` as part of the heldout set; for older runs that lack it, the separate BC eval run listed in `BC_BENCHMARK_RUNS` is appended at plot time. The script prints a per-cell coverage report (`builtin` / `bc_merge` / `MISSING`) at the end
- `--filter_failed_seeds` (default false) — filter out failed seeds
- `--tasks lbf/lbf_7x7_nolevels overcooked-v1/cramped_room` — restrict to specific tasks
- `--force_recompute` — re-download eval artifacts from wandb and recompute best returns and summary stats (normally unnecessary: caches are keyed by run IDs and invalidated automatically)
- `--save_dir PATH` — override output directory (default: `results/figures/`)

Figures are saved as PDFs to `results/figures/`.

### Heldout partner alignment

Runs may differ in their heldout sets (e.g. 17 vs 18 partners when `human_proxy`
was added), and the wandb run config does not preserve the heldout-set order, so
partners are identified **by name** rather than by index. The partner order for
each run is read from its logged `HeldoutEval/FinalEgoVsHeldout-*-CI` table
(cached under `results/figures/cache/run_heldout_names/`), and per-partner
bounds are looked up by name in the run config. See
[heldout_partners.py](heldout_partners.py). Best returns, renormalization, and
BC merging all operate on these named partners, so runs with different heldout
sets can be compared as long as their common partners share names.

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
observed for each heldout agent (by name), caching results in
`results/figures/cache/best_returns/<task>.json` together with a `_labels`
list giving the partner order. Pass `--include_bc` to also scan the separate
BC eval runs for methods whose heldout set lacks `human_proxy`.

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

Each point is one unique hyperparameter setting (mean score over its seeds). By
default only the settings actually considered when the benchmark configs were
chosen are shown: the seeded 140-setting subsample drawn by
`scripts/manage_configs/apply_best_hparams.py --max-hparams 140 --seed 0`
(`select_hparam_settings` there is the single source of truth; sweeps with at
most 140 settings are shown in full). Use `--max-hparams 0` to plot every setting,
or `--max-hparams N --seed S` to match a different selection.

Figures are saved to `results/figures/`.
