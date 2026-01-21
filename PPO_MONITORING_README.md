# PPO Ego Agent Monitoring System

## Overview

Similar to the BRDiv monitoring system, the PPO ego agent training now includes monitoring capabilities that record:
- **Wall-clock time** since training start
- **Ego agent returns** at each training update
- **Training losses** (value, actor, and entropy losses)

All data is automatically saved to JSON files and visualized with plots.

## Enabling Monitoring

### Via Command Line

Add these flags when running the training:

```bash
python teammate_generation/run.py \
    algorithm=brdiv/lbf \
    task=lbf \
    train_ego=true \
    enable_ppo_monitoring=true \
    ppo_monitoring_dir=./ppo_results
```

### Configuration Parameters

- `enable_ppo_monitoring`: true/false (default: false)
- `ppo_monitoring_dir`: output directory (default: ./ppo_monitoring)

## Output Files

After PPO training completes with monitoring enabled:

```
./ppo_results/
├── ppo_monitoring_data.json      # Raw data
└── ppo_monitoring_plot.png       # 4-panel visualization
```

### Data Format: `ppo_monitoring_data.json`

```json
{
  "wall_clock_times": [0.0, 1.5, 3.2, ...],
  "update_steps": [0, 1, 2, ...],
  "ego_returns": [0.35, 0.42, 0.48, ...],
  "ego_value_loss": [0.95, 0.87, 0.72, ...],
  "ego_actor_loss": [0.12, 0.10, 0.08, ...],
  "ego_entropy_loss": [0.05, 0.04, 0.03, ...]
}
```

**Fields:**
- `wall_clock_times`: Elapsed time in seconds since training start
- `update_steps`: Training update step (0-indexed)
- `ego_returns`: Ego agent average return across all episodes
- `ego_value_loss`: Value function loss
- `ego_actor_loss`: Policy gradient loss
- `ego_entropy_loss`: Entropy regularization loss

### Plot: `ppo_monitoring_plot.png`

4-panel visualization:
1. **Top-left**: Wall-clock time vs Ego Returns
2. **Top-right**: Update step vs Ego Returns
3. **Bottom-left**: Wall-clock time vs Training Losses
4. **Bottom-right**: Wall-clock time vs Return Improvement

## Implementation Details

### Core Module

**File**: `ego_agent_training/ppo_monitoring.py`

`PPOMonitor` class provides:
- `start()` - Initialize timing
- `record_update_with_time()` - Record metrics with explicit elapsed time
- `save_data()` - Save to JSON
- `plot_results()` - Generate plots

### Integration Points

**File**: `teammate_generation/train_ego.py`

1. **Initialization** (line ~60):
   ```python
   if config.get("enable_ppo_monitoring", False):
       monitor = PPOMonitor(...)
       monitor.start()
   ```

2. **Metric Recording** (line ~105):
   ```python
   if monitor is not None:
       monitor.record_update_with_time(...)
   ```

3. **Data Saving** (line ~120):
   ```python
   if monitor is not None:
       monitor.save_data()
       monitor.plot_results()
   ```

## Usage Example

### Complete Workflow

```bash
# Run BRDiv and then train PPO with monitoring
python teammate_generation/run.py \
    algorithm=brdiv/lbf \
    task=lbf \
    label=my_experiment \
    enable_brdiv_monitoring=true \
    brdiv_monitoring_dir=./results/brdiv \
    train_ego=true \
    enable_ppo_monitoring=true \
    ppo_monitoring_dir=./results/ppo \
    run_heldout_eval=false

# Then analyze both results
python teammate_generation/brdiv_monitoring_analysis.py ./results/brdiv/brdiv_monitoring_data.json --detailed
# (Add similar analysis for PPO if needed)
```

### Python API

```python
from ego_agent_training.ppo_monitoring import PPOMonitor

# Create monitor
monitor = PPOMonitor(output_dir="./my_ppo_results")
monitor.start()

# Record metrics
monitor.record_update_with_time(
    update_step=0,
    ego_return=0.35,
    value_loss=0.95,
    actor_loss=0.12,
    entropy_loss=0.05,
    elapsed_time=1.5
)

# Save and plot
monitor.save_data()
monitor.plot_results()
```

## Performance

- **CPU Overhead**: Negligible (<1%)
- **Memory**: ~40 bytes per update (6 floats)
- **No algorithmic changes**: Pure observation wrapper

## What's Not Changed

- ✓ PPO training algorithm - untouched
- ✓ Parameter updates - untouched
- ✓ Gradient computations - untouched
- ✓ Existing configs - backward compatible

Only monitoring and logging added.

## Comparing BRDiv and PPO

| Aspect | BRDiv | PPO |
|--------|-------|-----|
| Module | `brdiv_with_monitoring.py` | `ppo_monitoring.py` |
| Metrics | SP/XP returns | Ego returns + losses |
| Plots | 2-panel | 4-panel |
| Data file | `brdiv_monitoring_data.json` | `ppo_monitoring_data.json` |

## Troubleshooting

**Q: PPO monitoring directory not created?**
A: Directory is auto-created. Check file permissions.

**Q: No plot generated?**
A: Ensure matplotlib can write to disk. Check `ppo_monitoring_dir` is writable.

**Q: Monitoring data empty?**
A: Training may have crashed. Check logs for errors.

## Configuration in Main run.py

You can enable both BRDiv and PPO monitoring simultaneously:

```yaml
enable_brdiv_monitoring: true
brdiv_monitoring_dir: ./results/brdiv

enable_ppo_monitoring: true
ppo_monitoring_dir: ./results/ppo
```

All outputs will be saved to their respective directories.
