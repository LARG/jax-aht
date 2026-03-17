# XP Matrix Outputs and Heatmaps

This guide explains how to run heldout cross-play (XP) evaluation, what outputs are saved, and how to generate heatmaps.

## Run XP Evaluation

Run heldout XP evaluation:

```bash
python3 evaluation/run.py --config-name heldout_xp task=lbf
```

You can swap tasks as needed, for example:

```bash
python3 evaluation/run.py --config-name heldout_xp task=overcooked-v1/cramped_room
```

## Auto-Generate Heatmaps

Heatmap generation is controlled by the flag in [evaluation/configs/heldout_xp.yaml](../evaluation/configs/heldout_xp.yaml):

```yaml
xp_matrix_outputs:
  save_heatmap: true
```

Disable heatmap generation for a run:

```bash
python3 evaluation/run.py --config-name heldout_xp task=lbf xp_matrix_outputs.save_heatmap=false
```

## Output Files

Each metric saves two CSV formats in the Hydra output directory:

- Matrix CSV (human-readable cells): `mean (ci_lower, ci_upper)`
- Tidy CSV (one row per matrix entry): easier for pandas/seaborn analysis

When `xp_matrix_outputs.save_heatmap=true`, a PNG heatmap is also generated for each metric CSV.

## Manual Heatmap Generation

Use [evaluation/plot_xp_csv_heatmap.py](../evaluation/plot_xp_csv_heatmap.py) to generate heatmaps manually.

Single CSV:

```bash
python3 evaluation/plot_xp_csv_heatmap.py results/lbf/heldout_xp_matrix/<run_id>/percent_eaten_mean_normalized=True.csv
```

All CSV files in a run directory:

```bash
python3 evaluation/plot_xp_csv_heatmap.py results/lbf/heldout_xp_matrix/<run_id>
```

Disable per-cell annotations:

```bash
python3 evaluation/plot_xp_csv_heatmap.py results/lbf/heldout_xp_matrix/<run_id> --no-annot
```
