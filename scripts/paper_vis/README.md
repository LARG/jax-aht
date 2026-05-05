# Paper Visualizations

All scripts must be run from the **repo root** with the conda environment activated:

```bash
conda activate bench311
```

wandb run/sweep IDs for all plots are stored in [plot_globals.py](plot_globals.py).

---

## Benchmark bar charts

`benchmark_bar_charts.py` plots normalized agent performance across tasks.
we always set `--use_best_returns_normalization` for all figures to obtain most accurate performance comparisons.

**Unified benchmark** (teammate-generation methods — FCP, BRDiv, LBRDiv, CoMeDi, COLE, TrajeDi):
```bash
PYTHONPATH=. python scripts/paper_vis/benchmark_bar_charts.py --plot_type unified
```

**Ego benchmark** (ego-training methods — PPO, LIAM, MeLIBA):
```bash
PYTHONPATH=. python scripts/paper_vis/benchmark_bar_charts.py --plot_type ego
```

**Key flags:**
- `--use_best_returns_normalization` — renormalize by the best observed return per heldout agent instead of the original per-agent bounds
- `--tasks lbf/lbf_7x7_nolevels overcooked-v1/cramped_room` — restrict to specific tasks
- `--force_recompute` — recompute summary stats from cached wandb artifacts (does not re-download from wandb)
- `--save_dir PATH` — override output directory (default: `results/figures/`)

Figures are saved as PDFs to `results/figures/`.

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
`results/figures/cache/best_returns/<task>.json`.

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

Figures are saved to `results/figures/`.
