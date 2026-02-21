# BRDiv Multiple Runs - Solution Summary

## Problem
The current BRDiv implementation with monitoring only generates **6 data points** per run (one per training update step), making it difficult to analyze performance trends or generate sufficient data for statistical analysis.

## Solution
Two new utilities have been created to enable running BRDiv multiple times and aggregating the monitoring data:

### 1. **`run_brdiv_multiple_times.py`** - Orchestrate Multiple Runs
Runs the BRDiv algorithm multiple times sequentially and automatically aggregates the collected monitoring data.

**Features:**
- Runs BRDiv N times with different seeds
- Automatically collects monitoring data from each run
- Aggregates all data into a single file
- Generates comparison plots showing all runs
- Computes summary statistics

**Usage:**
```bash
python teammate_generation/run_brdiv_multiple_times.py \
    algorithm=brdiv/lbf \
    task=lbf \
    label=my_experiment \
    enable_brdiv_monitoring=true \
    num_runs=5 \
    run_heldout_eval=false \
    train_ego=false
```

**Output:**
- `brdiv_aggregated_results/brdiv_aggregated_data.json` - All 30 data points (5 runs × 6 points)
- `brdiv_aggregated_results/aggregated_summary.json` - Statistics
- `brdiv_aggregated_results/brdiv_aggregated_plot.png` - Individual runs colored separately
- `brdiv_aggregated_results/brdiv_aggregated_combined_plot.png` - Combined plot with trend line
- `brdiv_individual_runs/run_0/`, `run_1/`, etc. - Individual run data

### 2. **`aggregate_brdiv_runs.py`** - Post-hoc Aggregation
Aggregates monitoring data from existing BRDiv runs that were already executed.

**Features:**
- Works with existing run directories
- Flexible file discovery using glob patterns
- Can aggregate from multiple sources
- Generates plots and statistics

**Usage:**
```bash
# Aggregate from directory structure
python teammate_generation/aggregate_brdiv_runs.py \
    --input-dir ./brdiv_individual_runs \
    --output-dir ./aggregated_results

# Or from specific directories
python teammate_generation/aggregate_brdiv_runs.py \
    --run-dirs ./run1 ./run2 ./run3 \
    --output-dir ./aggregated_results

# Or from specific files
python teammate_generation/aggregate_brdiv_runs.py \
    --files ./run1/brdiv_monitoring_data.json ./run2/brdiv_monitoring_data.json \
    --output-dir ./aggregated_results
```

---

## Data Structure

### Input: Single Run Data
Original `brdiv_monitoring_data.json` (6 data points per run):
```json
{
  "wall_clock_times": [0.0, 13.69, 27.38, 41.08, 54.77, 68.46],
  "update_steps": [0, 1, 2, 3, 4, 5],
  "sp_returns": [0.0139, 0.0083, 0.0167, 0.0056, 0.0139, 0.0028],
  "xp_returns": [0.0208, 0.0069, 0.0125, 0.0069, 0.0125, 0.0139]
}
```

### Output: Aggregated Data
`brdiv_aggregated_data.json` (30 data points for 5 runs):
```json
{
  "wall_clock_times": [0.0, 13.69, 27.38, ..., 68.46, 0.0, 13.69, ...],
  "update_steps": [0, 1, 2, ..., 5, 0, 1, ...],
  "sp_returns": [0.0139, 0.0083, ..., 0.0028, 0.0142, 0.0081, ...],
  "xp_returns": [0.0208, 0.0069, ..., 0.0139, 0.0210, 0.0070, ...],
  "run_ids": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, ...]
}
```

### Summary Statistics
`aggregated_summary.json`:
```json
{
  "num_runs": 5,
  "num_total_data_points": 30,
  "data_points_per_run": [6, 6, 6, 6, 6],
  "sp_returns": {
    "mean": 0.010185,
    "std": 0.004986,
    "min": 0.0028,
    "max": 0.0167
  },
  "xp_returns": {
    "mean": 0.012269,
    "std": 0.004704,
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

## Quick Start Examples

### Example 1: Generate Data from Scratch
Run BRDiv 10 times and collect all data:
```bash
cd /scratch/cluster/adityam/jax-aht

python teammate_generation/run_brdiv_multiple_times.py \
    algorithm=brdiv/lbf \
    task=lbf \
    label=lbf_10runs \
    enable_brdiv_monitoring=true \
    num_runs=10 \
    run_heldout_eval=false \
    train_ego=false
```

**Result:** 60 data points total (10 runs × 6 points each)

### Example 2: Aggregate Existing Data
You already have runs in `./my_results/run_0/`, `./run_1/`, etc.:
```bash
python teammate_generation/aggregate_brdiv_runs.py \
    --run-dirs ./my_results/run_0 ./my_results/run_1 ./my_results/run_2 \
    --output-dir ./my_results/aggregated
```

### Example 3: Aggregate from Glob Pattern
Find all monitoring files in a complex directory structure:
```bash
python teammate_generation/aggregate_brdiv_runs.py \
    --input-dir ./results \
    --output-dir ./aggregated \
    --pattern "**/results/**/brdiv_monitoring_data.json"
```

---

## Configuration Parameters

### For `run_brdiv_multiple_times.py`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_runs` | int | 3 | Number of BRDiv runs to execute |
| `aggregation_output_dir` | str | `./brdiv_aggregated_results` | Directory for aggregated output |
| `individual_dirs_base` | str | `./brdiv_individual_runs` | Directory for individual run outputs |
| `enable_brdiv_monitoring` | bool | true | Must be true to collect data |
| All standard BRDiv config | - | - | Pass through to BRDiv |

### For `aggregate_brdiv_runs.py`

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--input-dir` | str | - | Search directory for monitoring files |
| `--run-dirs` | list | - | Specific directories to aggregate |
| `--files` | list | - | Specific JSON files to aggregate |
| `--output-dir` | str | `./brdiv_aggregated` | Output directory |
| `--pattern` | str | `**/monitoring/brdiv_monitoring_data.json` | Glob pattern for file discovery |

---

## Output Files Description

### `brdiv_aggregated_data.json`
Raw aggregated data with all data points and their run identifiers.
- Used for custom analysis and statistical tests
- Can be loaded directly into NumPy/Pandas/R

### `aggregated_summary.json`
Summary statistics computed from aggregated data.
- Quick overview of performance metrics
- Min/max/mean/std for each metric

### `brdiv_aggregated_plot.png`
Scatter plot showing individual runs with different colors.
- Left subplot: self-play returns vs wall-clock time
- Right subplot: cross-play returns vs wall-clock time
- Good for seeing variation across runs

### `brdiv_aggregated_combined_plot.png`
Combined plot with polynomial trend line overlaid.
- Shows overall pattern across all runs
- Polynomial fit helps identify trends
- Useful for presentation and reports

---

## Analysis Examples

### Load and Analyze in Python
```python
import json
import numpy as np
import pandas as pd

# Load aggregated data
with open("./brdiv_aggregated_results/brdiv_aggregated_data.json") as f:
    data = json.load(f)

# Convert to arrays
times = np.array(data["wall_clock_times"])
sp_returns = np.array(data["sp_returns"])
xp_returns = np.array(data["xp_returns"])
run_ids = np.array(data["run_ids"])

# Compute statistics
print(f"Overall SP return: {sp_returns.mean():.6f} ± {sp_returns.std():.6f}")
print(f"Overall XP return: {xp_returns.mean():.6f} ± {xp_returns.std():.6f}")

# Per-run statistics
for run_id in np.unique(run_ids):
    mask = run_ids == run_id
    print(f"Run {run_id}: SP={sp_returns[mask].mean():.6f}, XP={xp_returns[mask].mean():.6f}")

# Correlation analysis
correlation = np.corrcoef(times, sp_returns)[0, 1]
print(f"Correlation (time vs SP return): {correlation:.4f}")
```

### Statistical Comparison
```python
# T-test between two runs
from scipy import stats

run_0_mask = run_ids == 0
run_1_mask = run_ids == 1

t_stat, p_val = stats.ttest_ind(sp_returns[run_0_mask], sp_returns[run_1_mask])
print(f"T-test: t={t_stat:.4f}, p={p_val:.4f}")
```

### Compare Multiple Algorithms
```python
# Aggregate BRDiv, FCP, and LBRDiv results
algorithms = {
    "BRDiv": "./brdiv_aggregated/brdiv_aggregated_data.json",
    "FCP": "./fcp_aggregated/brdiv_aggregated_data.json",  # if using aggregator on FCP
    "LBRDiv": "./lbrdiv_aggregated/brdiv_aggregated_data.json",
}

for alg, data_file in algorithms.items():
    with open(data_file) as f:
        data = json.load(f)
    sp_ret = np.mean(data["sp_returns"])
    print(f"{alg}: {sp_ret:.6f}")
```

---

## Files Created

1. **`/scratch/cluster/adityam/jax-aht/teammate_generation/run_brdiv_multiple_times.py`**
   - Main orchestration script
   - Runs BRDiv N times and aggregates

2. **`/scratch/cluster/adityam/jax-aht/teammate_generation/aggregate_brdiv_runs.py`**
   - Standalone aggregation utility
   - Works with existing runs

3. **`/scratch/cluster/adityam/jax-aht/BRDIV_MULTIPLE_RUNS_GUIDE.md`**
   - Detailed user guide with examples
   - API documentation
   - Troubleshooting

4. **`/scratch/cluster/adityam/jax-aht/BRDIV_MULTIPLE_RUNS_SOLUTION.md`**
   - This file - overview and quick reference

---

## Troubleshooting

### Issue: Import errors when running scripts
**Solution:** Make sure you're using the correct Python environment:
```bash
source /scratch/cluster/adityam/jax-aht/.venv/bin/activate
python teammate_generation/aggregate_brdiv_runs.py ...
```

### Issue: "No data loaded!"
**Solution:** Check:
1. Monitoring files exist: `ls -R ./brdiv_individual_runs/*/monitoring/`
2. You ran with `enable_brdiv_monitoring=true`
3. File paths are correct: `--input-dir ./correct/path`

### Issue: Plots show only a few runs
**Solution:** Normal! The script uses `plt.cm.tab10` for colors, which supports up to 10 runs by default. For more runs, it will cycle colors. Consider splitting into smaller groups if visualization gets crowded.

### Issue: Memory error with many runs
**Solution:** The aggregator processes data sequentially, so it should be memory-efficient. If you hit limits:
1. Aggregate in smaller batches
2. Use `--pattern` to limit which files to load
3. Check available disk space

---

## Next Steps

1. **Try it out:**
   ```bash
   python teammate_generation/run_brdiv_multiple_times.py \
       algorithm=brdiv/lbf task=lbf num_runs=3 \
       enable_brdiv_monitoring=true run_heldout_eval=false train_ego=false
   ```

2. **Analyze the results:**
   - View the plots in `./brdiv_aggregated_results/`
   - Load the JSON data for custom analysis

3. **Scale up:**
   - Increase `num_runs` for more data points
   - Run on multiple algorithms for comparison

4. **Publish results:**
   - Use aggregated plots in papers/reports
   - Include summary statistics in tables

---

## Implementation Details

### How it Works

1. **Run Setup:** Each run gets its own directory with a unique seed
2. **Monitoring:** Data collected automatically if `enable_brdiv_monitoring=true`
3. **Data Aggregation:** JSON files loaded and combined with run_id tracking
4. **Analysis:** Summary statistics computed across all runs
5. **Visualization:** Separate scatter plots for each run, combined plot with trends

### Key Classes

- **`BRDivAggregator`:** Manages data collection and aggregation
  - `add_run()` - Add data from one run
  - `load_run_from_file()` - Load from JSON file
  - `save_aggregated_data()` - Save all data
  - `plot_aggregated_results()` - Generate plots

### Data Flow

```
Individual Runs
    ↓
brdiv_monitoring_data.json (per run)
    ↓
BRDivAggregator loads all files
    ↓
brdiv_aggregated_data.json (combined)
    ↓
Summary statistics + Plots
```

---

## For Questions or Issues

The code is well-documented with docstrings. Key entry points:
- [run_brdiv_multiple_times.py](../teammate_generation/run_brdiv_multiple_times.py) - Line 110
- [aggregate_brdiv_runs.py](../teammate_generation/aggregate_brdiv_runs.py) - Line 120
- [BRDIV_MULTIPLE_RUNS_GUIDE.md](../BRDIV_MULTIPLE_RUNS_GUIDE.md) - Detailed guide

