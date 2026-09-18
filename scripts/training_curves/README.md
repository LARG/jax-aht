# Training curves

Scrapes per-seed training time-series from the wandb benchmark runs and renders
them into a figure containing key metrics for each algorithm. 

## Generating Figure

The figure in the paper shows all algorithms on LBF 12x12.

```bash
python -m scripts.training_curves.meta.collect_meta_data && \
python -m scripts.training_curves.meta.plot_meta
```

Result: `results/figures/training_curves/aggregate_2/meta_plot.{pdf,png}`.

The two stages are separable and worth knowing about:

```bash
# 1. wandb -> meta_data.pkl   (slow: auth + several GB of artifacts)
python -m scripts.training_curves.meta.collect_meta_data

# 2. meta_data.pkl -> figure  (offline, seconds)
python -m scripts.training_curves.meta.plot_meta
```

Step 1 needs wandb auth and downloads several GB of artifacts the first time
(~15 min); afterwards it is served from the on-disk cache. Step 2 reads only
the pickle, so it is offline and takes a few seconds — iterate on layout by
re-running step 2 alone.

### Outputs

All under `results/figures/training_curves/aggregate_2/` (gitignored):

| File | Contents |
| --- | --- |
| `meta_plot.pdf` / `.png` | the figure (14.3 x 22.1 in, portrait) |
| `meta_data.pkl` | collected curves, ~3 MB |
| `meta_data.md` | manifest: which run ids and how many seeds fed each panel |

Check `meta_data.md` after collecting — it records the run id and seed count
behind every panel, which is the quickest way to spot a run that silently
resolved to the wrong thing.

### Which runs get used

Run ids are pinned in `scripts/paper_vis/plot_globals.py`
(`UNIFIED_BENCHMARK_RUNS`, `EGO_BENCHMARK_RUNS`) so this figure uses the same
runs as the rest of the paper. `common.find_benchmark_runs` prefers those pins
and falls back to querying the `neurips:benchmark` tag only when a
(algorithm, task) pair is not pinned. To move the figure onto different runs,
edit the pin tables rather than this package.

### Other Environments

Each run collects **one task**, not all of them — `--task` defaults to
`lbf/lbf_12x12`, which is the task the paper figure uses. For another task:

```bash
python -m scripts.training_curves.meta.collect_meta_data --task overcooked-v1/coord_ring
```

You might need to check the output paths and locations to get it to work. 

## Per-algorithm plots

Each algorithm also has a standalone CLI producing just its own curves:

```bash
python -m scripts.training_curves.fcp.run --task lbf/lbf_12x12
python -m scripts.training_curves.rotate.run --task lbf/lbf_12x12
```

Available for `fcp`, `brdiv`, `lbrdiv`, `comedi`, `cole`, `trajedi`, `rotate`,
`ppo_ego`, `liam_ego`, `meliba_ego`. All accept `--task` (required),
`--out-dir` and `--force-recompute`. These are independent of the meta-plot,
which reads BRDIV/LBRDIV matrices and the ego curves through `common.py`
directly.
