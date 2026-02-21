# Getting Started: BRDiv Multiple Runs

This guide explains how to use the new tools to run BRDiv multiple times and aggregate the monitoring data.

## The Problem You're Solving

Currently, each BRDiv run generates **6 data points** (one per training update step). This makes it difficult to:
- Analyze performance trends
- Perform statistical analysis
- Compare with other algorithms
- Generate publication-quality plots

## The Solution

Two new tools allow you to:
1. Run BRDiv multiple times (collecting 6+ data points per run)
2. Automatically aggregate all data into a single dataset
3. Generate plots and statistics

**With 5 runs:** 30 data points  
**With 10 runs:** 60 data points  
**With 20 runs:** 120 data points

## Quick Start (2 minutes)

### Option A: Run Multiple Times (Recommended)

```bash
cd /scratch/cluster/adityam/jax-aht

python teammate_generation/run_brdiv_multiple_times.py \
    algorithm=brdiv/lbf \
    task=lbf \
    label=my_experiment \
    enable_brdiv_monitoring=true \
    num_runs=5 \
    run_heldout_eval=false \
    train_ego=false
```

**What happens:**
- Runs BRDiv 5 times
- Each run takes ~1 minute
- Total time: ~5 minutes + 30 seconds for aggregation
- Saves 30 data points to `./brdiv_aggregated_results/`

### Option B: Aggregate Existing Runs

If you already have runs in directories:

```bash
python teammate_generation/aggregate_brdiv_runs.py \
    --run-dirs ./brdiv_individual_runs/run_0 \
              ./brdiv_individual_runs/run_1 \
              ./brdiv_individual_runs/run_2 \
    --output-dir ./my_aggregated_results
```

## Output Files

After running either script, you'll get:

```
brdiv_aggregated_results/
├── brdiv_aggregated_data.json         ← All data (30 points for 5 runs)
├── aggregated_summary.json             ← Statistics
├── brdiv_aggregated_plot.png          ← Individual runs (different colors)
└── brdiv_aggregated_combined_plot.png ← Combined with trend line
```

## View the Results

### Check Summary Statistics
```bash
cat ./brdiv_aggregated_results/aggregated_summary.json
```

Example output:
```json
{
  "num_runs": 5,
  "num_total_data_points": 30,
  "sp_returns": {
    "mean": 0.010185,
    "std": 0.004986
  },
  "xp_returns": {
    "mean": 0.012269,
    "std": 0.004704
  }
}
```

### View the Plots
Open in VS Code or any image viewer:
- `brdiv_aggregated_plot.png` - Color-coded by run
- `brdiv_aggregated_combined_plot.png` - Shows overall trend

### Load Data in Python
```python
import json
import numpy as np

# Load aggregated data
with open("./brdiv_aggregated_results/brdiv_aggregated_data.json") as f:
    data = json.load(f)

# Extract arrays
times = np.array(data["wall_clock_times"])
sp_returns = np.array(data["sp_returns"])
run_ids = np.array(data["run_ids"])

# Analyze
print(f"Total points: {len(times)}")
print(f"Runs: {len(np.unique(run_ids))}")
print(f"Mean return: {sp_returns.mean():.6f}")
print(f"Std return: {sp_returns.std():.6f}")

# Per-run analysis
for run_id in np.unique(run_ids):
    mask = run_ids == run_id
    print(f"Run {run_id}: mean={sp_returns[mask].mean():.6f}")
```

## Common Workflows

### 1. Quick Test (3 runs)
```bash
python teammate_generation/run_brdiv_multiple_times.py \
    algorithm=brdiv/lbf task=lbf num_runs=3 \
    enable_brdiv_monitoring=true run_heldout_eval=false train_ego=false
```

### 2. Comprehensive Analysis (10 runs)
```bash
python teammate_generation/run_brdiv_multiple_times.py \
    algorithm=brdiv/lbf task=lbf num_runs=10 \
    enable_brdiv_monitoring=true run_heldout_eval=false train_ego=false
```

### 3. Compare Different Environments

LBF:
```bash
python teammate_generation/run_brdiv_multiple_times.py \
    algorithm=brdiv/lbf task=lbf num_runs=5 \
    enable_brdiv_monitoring=true aggregation_output_dir=./lbf_results \
    run_heldout_eval=false train_ego=false
```

Hanabi:
```bash
python teammate_generation/run_brdiv_multiple_times.py \
    algorithm=brdiv/hanabi task=hanabi num_runs=5 \
    enable_brdiv_monitoring=true aggregation_output_dir=./hanabi_results \
    run_heldout_eval=false train_ego=false
```

### 4. Aggregate Multiple Sources
```bash
python teammate_generation/aggregate_brdiv_runs.py \
    --input-dir ./brdiv_individual_runs \
    --output-dir ./final_aggregation \
    --pattern "**/monitoring/brdiv_monitoring_data.json"
```

## Parameters Explained

### `run_brdiv_multiple_times.py` Parameters

| Parameter | Example | Description |
|-----------|---------|-------------|
| `num_runs` | 5 | How many times to run BRDiv (default: 3) |
| `aggregation_output_dir` | `./my_results` | Where to save aggregated data |
| `individual_dirs_base` | `./my_runs` | Where individual runs are saved |
| `enable_brdiv_monitoring` | true | Must be `true` to collect timing data |
| `algorithm` | `brdiv/lbf` | Which algorithm to use |
| `task` | `lbf` | Which environment/task |

### `aggregate_brdiv_runs.py` Parameters

| Parameter | Example | Description |
|-----------|---------|-------------|
| `--input-dir` | `./runs` | Search directory for monitoring files |
| `--run-dirs` | `./r1 ./r2` | Specific directories to aggregate |
| `--files` | `./d1.json ./d2.json` | Specific JSON files to aggregate |
| `--output-dir` | `./agg` | Output directory (default: `./brdiv_aggregated`) |
| `--pattern` | `**/data.json` | Glob pattern to find files |

## Understanding the Output

### `brdiv_aggregated_data.json` Structure

```json
{
  "wall_clock_times": [0.0, 13.7, 27.4, ..., 68.5, 0.0, 13.7, ...],
  "update_steps": [0, 1, 2, ..., 5, 0, 1, ...],
  "sp_returns": [0.0139, 0.0083, ..., 0.0028, 0.0142, ...],
  "xp_returns": [0.0208, 0.0069, ..., 0.0139, 0.0210, ...],
  "run_ids": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, ...]
}
```

- **wall_clock_times**: Elapsed seconds from algorithm start
- **update_steps**: Which training update (0-5 for BRDiv)
- **sp_returns**: Self-play returns (confederate vs confederate)
- **xp_returns**: Cross-play returns (confederate vs best response)
- **run_ids**: Which run this data point came from (0-4 for 5 runs)

### `aggregated_summary.json` Structure

```json
{
  "num_runs": 5,
  "num_total_data_points": 30,
  "data_points_per_run": [6, 6, 6, 6, 6],
  "sp_returns": {
    "mean": 0.0102,
    "std": 0.0050,
    "min": 0.0028,
    "max": 0.0167
  },
  "xp_returns": {
    "mean": 0.0123,
    "std": 0.0047,
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

## Tips & Tricks

### Get More Data Points
If you need more data:
1. Increase `num_runs` (more independent runs)
2. Run across different tasks/environments
3. Combine with `aggregate_brdiv_runs.py`

### Faster Aggregation
```bash
# For one-time setup, this is fast
python teammate_generation/aggregate_brdiv_runs.py \
    --input-dir ./brdiv_individual_runs \
    --output-dir ./results

# Output files generated in <5 seconds
```

### Programmatic Usage in Python
```python
from teammate_generation.run_brdiv_multiple_times import BRDivAggregator

# Manual aggregation
agg = BRDivAggregator(output_dir="./results")
agg.load_run_from_file("./run1/monitoring/brdiv_monitoring_data.json", run_id=0)
agg.load_run_from_file("./run2/monitoring/brdiv_monitoring_data.json", run_id=1)
agg.save_aggregated_data()
agg.plot_aggregated_results()
```

## Troubleshooting

### Script Won't Start
**Error:** `ModuleNotFoundError: No module named 'xxx'`

**Solution:** Activate the environment:
```bash
source /scratch/cluster/adityam/jax-aht/.venv/bin/activate
python teammate_generation/run_brdiv_multiple_times.py ...
```

### No Monitoring Data Found
**Error:** `Monitoring data not found at ...`

**Check:**
1. Did you set `enable_brdiv_monitoring=true`? (Required!)
2. Is the path correct?
3. Did the run complete successfully?

### Few Data Points
**Expected!** Each run generates 6 points by default. This is by design:
- Update 0: ~0 seconds elapsed
- Update 1: ~13.7 seconds
- Update 2: ~27.4 seconds
- Update 3: ~41.1 seconds
- Update 4: ~54.8 seconds
- Update 5: ~68.5 seconds

Run 5 times = 30 points total.

### Plots Look Empty
**Check:**
1. Did aggregation complete? (Should print "✓" messages)
2. Is there data in the JSON file? (Check file size > 1KB)
3. Are the paths correct?

## File Locations

**Main Scripts:**
- [run_brdiv_multiple_times.py](../teammate_generation/run_brdiv_multiple_times.py)
- [aggregate_brdiv_runs.py](../teammate_generation/aggregate_brdiv_runs.py)

**Documentation:**
- [BRDIV_MULTIPLE_RUNS_GUIDE.md](../BRDIV_MULTIPLE_RUNS_GUIDE.md) - Detailed guide
- [BRDIV_MULTIPLE_RUNS_SOLUTION.md](../BRDIV_MULTIPLE_RUNS_SOLUTION.md) - Technical overview

## What's Next?

1. **Run it:** Execute one of the commands above
2. **Analyze:** Open the JSON files and plots
3. **Iterate:** Adjust `num_runs` as needed
4. **Compare:** Aggregate different algorithms/tasks

## Questions?

Refer to:
- This file: Quick reference
- [BRDIV_MULTIPLE_RUNS_GUIDE.md](../BRDIV_MULTIPLE_RUNS_GUIDE.md): Detailed documentation
- [BRDIV_MULTIPLE_RUNS_SOLUTION.md](../BRDIV_MULTIPLE_RUNS_SOLUTION.md): Technical details
- Code docstrings: Implementation details

---

**Version:** 1.0  
**Date:** January 25, 2026  
**Location:** `/scratch/cluster/adityam/jax-aht/`

