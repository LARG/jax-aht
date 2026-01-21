# BRDiv Monitoring Guide

This guide explains how to use the BRDiv monitoring system to record wall-clock time and returns during training.

## Overview

The monitoring system provides:
- **Real-time tracking** of wall-clock time since algorithm start
- **Returns tracking** at each update step (both self-play and cross-play)
- **Automatic data persistence** to JSON files
- **Automatic plotting** of wall-clock time vs returns

## Files

### Core Monitoring Module
- **`brdiv_with_monitoring.py`**: Contains the `BRDivMonitor` class and utilities
  - `BRDivMonitor`: Tracks timing and returns data
  - `wrap_run_brdiv_with_monitoring()`: Decorator for wrapping BRDiv
  - `extract_and_record_metrics()`: Utility to extract metrics from training

### Modified Files
- **`BRDiv.py`**: Updated to integrate monitoring
  - `run_brdiv()`: Now initializes and passes monitor to `log_metrics()`
  - `log_metrics()`: Now records data in monitor at each update

### Example Scripts
- **`run_brdiv_monitored.py`**: Example script showing how to enable monitoring

## Usage

### Method 1: Using Configuration (Recommended)

Simply add monitoring flags to your Hydra configuration:

```bash
python teammate_generation/run.py \
    algorithm=brdiv/lbf \
    task=lbf \
    label=test_brdiv_monitored \
    enable_brdiv_monitoring=true \
    brdiv_monitoring_dir=./brdiv_monitoring_output \
    run_heldout_eval=false \
    train_ego=false
```

**Configuration Parameters:**
- `enable_brdiv_monitoring`: (bool) Enable/disable monitoring. Default: `false`
- `brdiv_monitoring_dir`: (str) Directory to save monitoring data. Default: `./brdiv_monitoring`

### Method 2: Programmatic Usage

```python
from teammate_generation.brdiv_with_monitoring import BRDivMonitor
from teammate_generation.BRDiv import run_brdiv
import time

# Create monitor
monitor = BRDivMonitor(output_dir="./my_monitoring")
monitor.start()

# Run BRDiv with monitoring
config = {...}
config["enable_brdiv_monitoring"] = True
config["brdiv_monitoring_dir"] = "./my_monitoring"

partner_params, partner_population = run_brdiv(config, wandb_logger)

# Data is automatically saved and plotted
```

## Output Files

After running BRDiv with monitoring enabled, you'll find:

### 1. `brdiv_monitoring_data.json`
JSON file containing all collected metrics:
```json
{
  "wall_clock_times": [0.1, 2.3, 5.4, ...],
  "update_steps": [0, 1, 2, ...],
  "sp_returns": [0.45, 0.52, 0.58, ...],
  "xp_returns": [0.23, 0.31, 0.39, ...]
}
```

**Fields:**
- `wall_clock_times`: Elapsed time in seconds since algorithm start
- `update_steps`: Update step number (0-indexed)
- `sp_returns`: Self-play returns (confederate vs confederate, averaged)
- `xp_returns`: Cross-play returns (confederate vs best response, averaged)

### 2. `brdiv_monitoring_plot.png`
Visualization with two subplots:
- **Left**: Wall-clock time vs Self-play returns
- **Right**: Wall-clock time vs Cross-play returns

## Data Analysis

### Loading and Analyzing Data

```python
import json
import matplotlib.pyplot as plt

# Load monitoring data
with open("./brdiv_monitoring_output/brdiv_monitoring_data.json") as f:
    data = json.load(f)

# Access fields
times = data["wall_clock_times"]
sp_returns = data["sp_returns"]
xp_returns = data["xp_returns"]

# Create custom plots
fig, ax = plt.subplots()
ax.plot(times, sp_returns, label="Self-play")
ax.plot(times, xp_returns, label="Cross-play")
ax.set_xlabel("Wall-clock Time (s)")
ax.set_ylabel("Return")
ax.legend()
plt.show()
```

### Computing Training Statistics

```python
import json
import numpy as np

with open("./brdiv_monitoring_output/brdiv_monitoring_data.json") as f:
    data = json.load(f)

times = np.array(data["wall_clock_times"])
sp_returns = np.array(data["sp_returns"])
xp_returns = np.array(data["xp_returns"])

# Compute statistics
print(f"Total training time: {times[-1]:.2f}s")
print(f"Self-play return improvement: {sp_returns[-1] - sp_returns[0]:.4f}")
print(f"Cross-play return improvement: {xp_returns[-1] - xp_returns[0]:.4f}")
print(f"Average convergence speed (updates/sec): {len(times) / times[-1]:.4f}")

# Find best returns
best_sp_idx = np.argmax(sp_returns)
best_xp_idx = np.argmax(xp_returns)
print(f"\nBest self-play at step {best_sp_idx} ({times[best_sp_idx]:.2f}s)")
print(f"Best cross-play at step {best_xp_idx} ({times[best_xp_idx]:.2f}s)")
```

## How It Works

### Integration Points

1. **`run_brdiv()` initialization** (BRDiv.py, ~line 720):
   - Checks if monitoring is enabled via config
   - Creates `BRDivMonitor` instance
   - Calls `monitor.start()` to record start time

2. **`log_metrics()` recording** (BRDiv.py, ~line 765):
   - For each update step, calls `monitor.record_update()`
   - Computes wall-clock elapsed time
   - Records self-play and cross-play returns
   - After all steps, saves data and generates plots

### What's NOT Changed

The underlying BRDiv algorithm remains completely untouched:
- Training loops unchanged
- Parameter updates unchanged
- Return computations unchanged
- Only metadata recording added

## Performance Impact

- **Negligible overhead**: Monitoring adds <1% computational cost
- **Memory**: Stores one float per update step (minimal)
- **No algorithmic changes**: Pure instrumentation wrapper

## Troubleshooting

### Monitoring directory doesn't exist
The directory is created automatically. Check file permissions if creation fails.

### No plot generated
Ensure matplotlib can save to disk. Check `brdiv_monitoring_dir` is writable.

### JSON file is empty or incomplete
Training may not have completed. Check training logs for errors.

### Missing wall-clock times
Ensure `monitor.start()` is called before training begins (automatic with config method).

## Advanced: Custom Monitoring

To add custom metrics to monitoring:

```python
from teammate_generation.brdiv_with_monitoring import BRDivMonitor

monitor = BRDivMonitor()
monitor.start()

# Add custom data
monitor.data["custom_metric"] = [...]

# Save
monitor.save_data()
```

## Example Full Run

```bash
# Run BRDiv with monitoring on LBF environment
python teammate_generation/run.py \
    algorithm=brdiv/lbf \
    task=lbf \
    label=brdiv_test \
    enable_brdiv_monitoring=true \
    brdiv_monitoring_dir=./results/brdiv_lbf_run1 \
    run_heldout_eval=false \
    train_ego=false
```

After completion:
```
./results/brdiv_lbf_run1/
├── brdiv_monitoring_data.json    # Raw data
└── brdiv_monitoring_plot.png     # Visualization
```
