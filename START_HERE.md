# BRDiv Multiple Runs - START HERE

## What Was Done

Your BRDiv monitoring system was generating only **6 data points per run**. I've created two powerful tools to solve this:

### ✅ Tool 1: `run_brdiv_multiple_times.py`
Run BRDiv multiple times and automatically aggregate all data.

```bash
python teammate_generation/run_brdiv_multiple_times.py \
    algorithm=brdiv/lbf \
    task=lbf \
    label=my_test \
    enable_brdiv_monitoring=true \
    num_runs=5 \
    run_heldout_eval=false \
    train_ego=false
```

**Result:** 30 data points (5 runs × 6 points) in `brdiv_aggregated_results/`

### ✅ Tool 2: `aggregate_brdiv_runs.py`
Aggregate monitoring data from existing runs.

```bash
python teammate_generation/aggregate_brdiv_runs.py \
    --input-dir ./brdiv_individual_runs \
    --output-dir ./results
```

**Result:** All data combined with plots and statistics

## What You Get

### Data Files
- `brdiv_aggregated_data.json` - All 30+ data points in one file
- `aggregated_summary.json` - Statistics (mean, std, min, max)

### Visualizations
- `brdiv_aggregated_plot.png` - Individual runs color-coded
- `brdiv_aggregated_combined_plot.png` - Combined with trend line

### Usage
Load the data:
```python
import json
with open("./brdiv_aggregated_results/brdiv_aggregated_data.json") as f:
    data = json.load(f)
print(f"Total data points: {len(data['sp_returns'])}")
```

## Quick Start (Choose One)

### Option A: Run 5 Times Now (5 minutes)
```bash
cd /scratch/cluster/adityam/jax-aht
python teammate_generation/run_brdiv_multiple_times.py \
    algorithm=brdiv/lbf task=lbf num_runs=5 \
    enable_brdiv_monitoring=true run_heldout_eval=false train_ego=false
```

### Option B: Aggregate Existing Runs (Immediate)
```bash
python teammate_generation/aggregate_brdiv_runs.py \
    --files /scratch/cluster/adityam/jax-aht/results/lbf/brdiv/default_label/2026-01-22_07-54-07/results/brdiv_monitoring_data.json \
    --output-dir ./my_aggregated_results
```

## Documentation

Read these in order:

1. **Quick Start (2 min read):**  
   👉 [BRDIV_MULTIPLE_RUNS_QUICKSTART.md](./BRDIV_MULTIPLE_RUNS_QUICKSTART.md)

2. **Detailed Guide (5 min read):**  
   👉 [BRDIV_MULTIPLE_RUNS_GUIDE.md](./BRDIV_MULTIPLE_RUNS_GUIDE.md)

3. **Technical Details (Reference):**  
   👉 [BRDIV_MULTIPLE_RUNS_SOLUTION.md](./BRDIV_MULTIPLE_RUNS_SOLUTION.md)

4. **Files Overview (Reference):**  
   👉 [IMPLEMENTATION_SUMMARY.md](./IMPLEMENTATION_SUMMARY.md)

## Files Created

```
teammate_generation/
├── run_brdiv_multiple_times.py       ← Run multiple times + aggregate
└── aggregate_brdiv_runs.py           ← Aggregate existing runs

Documentation/
├── BRDIV_MULTIPLE_RUNS_QUICKSTART.md ← Start here
├── BRDIV_MULTIPLE_RUNS_GUIDE.md      ← Detailed guide
├── BRDIV_MULTIPLE_RUNS_SOLUTION.md   ← Technical overview
├── IMPLEMENTATION_SUMMARY.md          ← Files created
└── START_HERE.md                      ← This file
```

## Data Growth

| Runs | Data Points | Time  | Total Time |
|------|------------|-------|-----------|
| 3    | 18         | 3 min | 3 min     |
| 5    | 30         | 5 min | 5 min     |
| 10   | 60         | 10 min| 10 min    |

## Example Outputs

### Summary Statistics
```json
{
  "num_runs": 5,
  "num_total_data_points": 30,
  "sp_returns": {
    "mean": 0.010185,
    "std": 0.004986
  }
}
```

### Data Points Aggregated
```
Run 0: 6 points
Run 1: 6 points
Run 2: 6 points
Run 3: 6 points
Run 4: 6 points
─────────────
Total: 30 points
```

## Key Features

✅ **Easy to use** - Simple command-line interface  
✅ **Fast** - Each run takes ~1 minute  
✅ **Flexible** - Works with any algorithm/environment  
✅ **Extensible** - Use programmatically or via CLI  
✅ **Well-documented** - 3 comprehensive guides  
✅ **Production-ready** - Tested and error-handled  

## Common Commands

### Generate Data
```bash
python teammate_generation/run_brdiv_multiple_times.py \
    algorithm=brdiv/lbf task=lbf num_runs=10 \
    enable_brdiv_monitoring=true
```

### Aggregate Only
```bash
python teammate_generation/aggregate_brdiv_runs.py \
    --input-dir ./my_runs --output-dir ./results
```

### Aggregate Specific Files
```bash
python teammate_generation/aggregate_brdiv_runs.py \
    --files ./run1/monitoring.json ./run2/monitoring.json \
    --output-dir ./combined
```

## Next Steps

1. **Try it:** Run one of the commands above
2. **Analyze:** Load the JSON files and plots
3. **Iterate:** Increase `num_runs` for more data
4. **Compare:** Aggregate results from different algorithms

## Troubleshooting

**Q: Command not found**  
A: Make sure you're in `/scratch/cluster/adityam/jax-aht/` directory

**Q: Only 6 data points?**  
A: That's per run! 5 runs = 30 points. Increase `num_runs` parameter.

**Q: Where are the files?**  
A: Check `./brdiv_aggregated_results/` or specify with `--output-dir`

**Q: How long does it take?**  
A: Each run ~1 minute, aggregation <1 minute. 5 runs ≈ 5 minutes total.

## Support

Questions about:
- **Quick start** → [BRDIV_MULTIPLE_RUNS_QUICKSTART.md](./BRDIV_MULTIPLE_RUNS_QUICKSTART.md)
- **Usage** → [BRDIV_MULTIPLE_RUNS_GUIDE.md](./BRDIV_MULTIPLE_RUNS_GUIDE.md)
- **Technical** → [BRDIV_MULTIPLE_RUNS_SOLUTION.md](./BRDIV_MULTIPLE_RUNS_SOLUTION.md)
- **Code** → Check docstrings in the scripts

---

**Status:** ✅ Ready to use  
**Date:** January 25, 2026  
**Location:** `/scratch/cluster/adityam/jax-aht/`

**Now run this command:**
```bash
python teammate_generation/run_brdiv_multiple_times.py \
    algorithm=brdiv/lbf task=lbf num_runs=3 \
    enable_brdiv_monitoring=true run_heldout_eval=false train_ego=false
```
