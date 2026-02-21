# BRDiv Multiple Runs & Aggregation Guide

## Overview

You now have two new tools to run BRDiv multiple times and aggregate the results:

1. **`run_brdiv_multiple_times.py`** - Run BRDiv multiple times in sequence and aggregate automatically
2. **`aggregate_brdiv_runs.py`** - Aggregate monitoring data from existing BRDiv runs

These tools solve the problem of having too few data points (6 per run) by allowing you to collect more data across multiple runs.

---

## Quick Start

### Option 1: Run Multiple Times from Scratch

Run BRDiv 5 times and automatically aggregate the data:

```bash
cd /scratch/cluster/adityam/jax-aht

python teammate_generation/run_brdiv_multiple_times.py \
    algorithm=brdiv/lbf \
    task=lbf \
    label=test_brdiv_aggregated \
    enable_brdiv_monitoring=true \
    num_runs=5 \
    run_heldout_eval=false \
    train_ego=false
```

**Output:**
- `./brdiv_aggregated_results/` - Aggregated data and plots
- `./brdiv_individual_runs/run_0/` through `./run_4/` - Individual run data

**Data generated:** If each run has 6 data points, you'll get **30 total data points** (5 runs × 6 points)

---

### Option 2: Aggregate Existing Runs

If you already have multiple BRDiv runs, aggregate their data:

```bash
# Find and aggregate all monitoring files in a directory
python teammate_generation/aggregate_brdiv_runs.py \
    --input-dir ./brdiv_individual_runs \
    --output-dir ./brdiv_aggregated

# Or specify directories explicitly
python teammate_generation/aggregate_brdiv_runs.py \
    --run-dirs ./brdiv_individual_runs/run_0 ./brdiv_individual_runs/run_1 \
    --output-dir ./brdiv_aggregated

# Or aggregate specific JSON files
python teammate_generation/aggregate_brdiv_runs.py \
    --files ./run1/monitoring/brdiv_monitoring_data.json \
           ./run2/monitoring/brdiv_monitoring_data.json \
    --output-dir ./brdiv_aggregated
```

---

## Output Files

Both tools produce the same output structure:

```
./brdiv_aggregated_results/
├── brdiv_aggregated_data.json          # All data points from all runs
├── aggregated_summary.json              # Summary statistics
├── brdiv_aggregated_plot.png           # Scatter plot with runs color-coded
└── brdiv_aggregated_combined_plot.png  # Combined plot with trend line
```

### `brdiv_aggregated_data.json`

```json
{
  "wall_clock_times": [0.0, 13.69, 27.38, ..., 68.45, ...],
  "update_steps": [0, 1, 2, ..., 5, ..., 0, 1, ...],
  "sp_returns": [0.0139, 0.0083, 0.0167, ..., 0.0139, ...],
  "xp_returns": [0.0208, 0.0069, 0.0125, ..., 0.0208, ...],
  "run_ids": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, ...]
}
```

### `aggregated_summary.json`

```json
{
  "num_runs": 5,
  "num_total_data_points": 30,
  "data_points_per_run": [6, 6, 6, 6, 6],
  "sp_returns": {
    "mean": 0.0123,
    "std": 0.0045,
    "min": 0.0028,
    "max": 0.0167
  },
  "xp_returns": {
    "mean": 0.0115,
    "std": 0.0050,
    "min": 0.0069,
    "max": 0.0208
  },
  "wall_clock_times": {
    "mean": 34.23,
    "std": 19.75,
    "min": 0.0,
    "max": 68.46,
    "total": 342.3
  }
}
```

---

## Plotting

The tools generate two types of plots:

### `brdiv_aggregated_plot.png`
- Shows individual runs with different colors
- Separate subplots for self-play and cross-play returns
- Good for seeing variation across runs

### `brdiv_aggregated_combined_plot.png`
- Individual data points from all runs (color-coded)
- Polynomial trend line overlaid
- Shows overall pattern across all data

---

## Configuration Parameters

### For `run_brdiv_multiple_times.py`

Add these to the command line:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_runs` | 3 | Number of times to run BRDiv |
| `aggregation_output_dir` | `./brdiv_aggregated_results` | Where to save aggregated data |
| `individual_dirs_base` | `./brdiv_individual_runs` | Where individual runs are saved |
| `enable_brdiv_monitoring` | true | Must be true to collect data |
| `brdiv_monitoring_dir` | Automatic | (Don't set - handled automatically) |

Example with more runs:
```bash
python teammate_generation/run_brdiv_multiple_times.py \
    algorithm=brdiv/lbf \
    task=lbf \
    num_runs=10 \
    aggregation_output_dir=./results/aggregated_lbf_10runs \
    enable_brdiv_monitoring=true \
    run_heldout_eval=false \
    train_ego=false
```

### For `aggregate_brdiv_runs.py`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input-dir` | - | Search directory for monitoring files |
| `--run-dirs` | - | Specific run directories to aggregate |
| `--files` | - | Specific JSON files to aggregate |
| `--output-dir` | `./brdiv_aggregated` | Output directory |
| `--pattern` | `**/monitoring/brdiv_monitoring_data.json` | Glob pattern for file discovery |

---

## Python API Usage

You can also use these tools programmatically:

```python
from teammate_generation.run_brdiv_multiple_times import run_brdiv_multiple_times_aggregated
from common.wandb_visualizations import Logger

# Set up config
cfg_dict = {
    "algorithm": {"ALG": "brdiv", ...},
    "enable_brdiv_monitoring": True,
    ...
}

# Create logger
wandb_logger = Logger(cfg_dict)

# Run and aggregate
aggregated_data, aggregator = run_brdiv_multiple_times_aggregated(
    cfg_dict,
    wandb_logger,
    num_runs=5,
    aggregation_output_dir="./my_aggregated_results",
)

# Access the data
print(f"Total points: {len(aggregated_data['wall_clock_times'])}")
print(f"Mean SP return: {np.mean(aggregated_data['sp_returns'])}")
```

Or for aggregating existing runs:

```python
from teammate_generation.aggregate_brdiv_runs import BRDivAggregator

aggregator = BRDivAggregator(output_dir="./results")
aggregator.load_runs_from_directory("./brdiv_runs")
aggregator.save_aggregated_data()
aggregator.plot_aggregated_results()
aggregator.plot_aggregated_results_combined()
aggregator.print_summary()
```

---

## Analysis & Visualization

### In Jupyter or Python script:

```python
import json
import numpy as np
import matplotlib.pyplot as plt

# Load aggregated data
with open("./brdiv_aggregated_results/brdiv_aggregated_data.json") as f:
    data = json.load(f)

# Load summary
with open("./brdiv_aggregated_results/aggregated_summary.json") as f:
    summary = json.load(f)

# Analyze
times = np.array(data["wall_clock_times"])
sp_returns = np.array(data["sp_returns"])
xp_returns = np.array(data["xp_returns"])
run_ids = np.array(data["run_ids"])

# Compute statistics
print(f"Runs: {summary['num_runs']}")
print(f"Total points: {summary['num_total_data_points']}")
print(f"SP return: {summary['sp_returns']['mean']:.6f} ± {summary['sp_returns']['std']:.6f}")
print(f"XP return: {summary['xp_returns']['mean']:.6f} ± {summary['xp_returns']['std']:.6f}")

# Per-run statistics
for run_id in np.unique(run_ids):
    mask = run_ids == run_id
    print(f"Run {run_id}: {mask.sum()} points, SP={sp_returns[mask].mean():.6f}, XP={xp_returns[mask].mean():.6f}")
```

---

## Troubleshooting

**Problem: "No data loaded!"**
- Check that monitoring files exist in the specified directories
- Verify you ran BRDiv with `enable_brdiv_monitoring=true`
- Try specifying files directly with `--files` option

**Problem: No trend line on combined plot**
- You need at least 3 data points to fit a polynomial
- Add more runs or data points

**Problem: Different number of data points per run**
- Normal! It depends on the training duration and evaluation frequency
- The aggregator handles variable lengths automatically

**Problem: Memory issues with many runs**
- The data is not loaded all at once - it's streamed
- If you have thousands of runs, the plots might get crowded
- Consider using the plot without individual run labels for large numbers

---

## Next Steps

1. **Run multiple times:**
   ```bash
   python teammate_generation/run_brdiv_multiple_times.py ... num_runs=10
   ```

2. **Analyze the aggregated data:**
   - Look at `aggregated_summary.json` for statistics
   - View the plots to see trends and variance

3. **Compare with other methods:**
   - Aggregate results from other algorithms (FCP, LBRDiv, CoMeDi)
   - Compare the aggregated plots side by side

4. **Statistical tests:**
   - Use the aggregated data for significance tests
   - Compare mean returns across algorithms

---

## Files Created

- `/scratch/cluster/adityam/jax-aht/teammate_generation/run_brdiv_multiple_times.py` - Multi-run orchestration
- `/scratch/cluster/adityam/jax-aht/teammate_generation/aggregate_brdiv_runs.py` - Aggregation utility
- This guide: `/scratch/cluster/adityam/jax-aht/BRDIV_MULTIPLE_RUNS_GUIDE.md`

