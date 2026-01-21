# PPO Monitoring - Quick Reference

## Enable PPO Monitoring

```bash
python teammate_generation/run.py \
    algorithm=brdiv/lbf \
    task=lbf \
    train_ego=true \
    enable_ppo_monitoring=true \
    ppo_monitoring_dir=./results/ppo
```

## Output Files

```
./results/ppo/
├── ppo_monitoring_data.json    # Raw data (times, returns, losses)
└── ppo_monitoring_plot.png     # 4-panel visualization
```

## Data Format

```json
{
  "wall_clock_times": [0.0, 1.5, 3.2, ...],
  "update_steps": [0, 1, 2, ...],
  "ego_returns": [0.35, 0.42, 0.48, ...],
  "ego_value_loss": [0.95, 0.87, ...],
  "ego_actor_loss": [0.12, 0.10, ...],
  "ego_entropy_loss": [0.05, 0.04, ...]
}
```

## Configuration Flags

| Flag | Type | Default | Purpose |
|------|------|---------|---------|
| `enable_ppo_monitoring` | bool | false | Enable/disable monitoring |
| `ppo_monitoring_dir` | str | ./ppo_monitoring | Output directory |

## Combined BRDiv + PPO Monitoring

Run both algorithms with full monitoring:

```bash
python teammate_generation/run.py \
    algorithm=brdiv/lbf task=lbf \
    enable_brdiv_monitoring=true \
    brdiv_monitoring_dir=./results/brdiv \
    train_ego=true \
    enable_ppo_monitoring=true \
    ppo_monitoring_dir=./results/ppo
```

Output structure:
```
./results/
├── brdiv/
│   ├── brdiv_monitoring_data.json
│   └── brdiv_monitoring_plot.png
└── ppo/
    ├── ppo_monitoring_data.json
    └── ppo_monitoring_plot.png
```

## Plot Panels

4-panel `ppo_monitoring_plot.png`:
1. Time vs Ego Returns
2. Update Step vs Ego Returns
3. Time vs Training Losses (value, actor, entropy)
4. Time vs Return Improvement

## Python API

```python
from ego_agent_training.ppo_monitoring import PPOMonitor

monitor = PPOMonitor(output_dir="./my_results")
monitor.start()
monitor.record_update_with_time(
    update_step=0,
    ego_return=0.35,
    value_loss=0.95,
    actor_loss=0.12,
    entropy_loss=0.05,
    elapsed_time=1.5
)
monitor.save_data()
monitor.plot_results()
```

## What's Tracked

| Metric | Description |
|--------|-------------|
| wall_clock_times | Elapsed seconds since training start |
| update_steps | Training update number |
| ego_returns | Ego agent average return |
| ego_value_loss | Value function loss |
| ego_actor_loss | Policy gradient loss |
| ego_entropy_loss | Entropy regularization loss |

## Files Modified

- ✅ `ego_agent_training/ppo_monitoring.py` (NEW - 180 lines)
- ✅ `teammate_generation/train_ego.py` (+40 lines, 0 algo changes)
- ✅ `PPO_MONITORING_README.md` (NEW - documentation)

## See Also

- BRDiv monitoring: `BRDIV_MONITORING_QUICK_START.md`
- Full PPO guide: `PPO_MONITORING_README.md`
- BRDiv system: `teammate_generation/brdiv_with_monitoring.py`
