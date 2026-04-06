# Publication t-SNE Generation (XP Matrix Columns)


## What These Plots Represent

- These are t-SNE embeddings of XP matrix columns.
- Each point is a policy represented by its cross-play return vector against teammates.

## Prerequisites

- Run from repo root.
- Python environment with project dependencies and scikit-learn.

```bash
cd /scratch/cluster/montek/projects/jax-aht
```

## 1) (If Needed) Generate Extra BR Policies

LBF extra teammates:

```bash
GPU_LIST=1,2 PARALLEL_JOBS=2 TOTAL_TIMESTEPS=10000000 bash scripts/run_lbf_extra_br_jobs.sh
```

Overcooked extra teammates:

```bash
GPU_LIST=1,2 PARALLEL_JOBS=2 TOTAL_TIMESTEPS=10000000 bash scripts/run_overcooked_extra_br_jobs.sh
```

## 2) Generate XP Matrices (All Teammates)

This uses the all-teammates config:
- evaluation/configs/heldout_xp_all_teammates.yaml

```bash
cd /scratch/cluster/montek/projects/jax-aht

for task in \
  lbf \
  overcooked-v1/cramped_room \
  overcooked-v1/asymm_advantages \
  overcooked-v1/counter_circuit \
  overcooked-v1/coord_ring \
  overcooked-v1/forced_coord
do
  safe_task=$(echo "$task" | tr '/' '_')
  label="all_teammates_norm_rows_${safe_task}"

  python3 evaluation/run.py \
    --config-name heldout_xp_all_teammates \
    task="$task" \
    label="$label" \
    xp_matrix_outputs.save_heatmap=true

done
```

## 3) Generate Per-Task Column t-SNE Plots

```bash
cd /scratch/cluster/montek/projects/jax-aht

for task in \
  lbf \
  overcooked-v1/cramped_room \
  overcooked-v1/asymm_advantages \
  overcooked-v1/counter_circuit \
  overcooked-v1/coord_ring \
  overcooked-v1/forced_coord
do
  safe_task=$(echo "$task" | tr '/' '_')
  label="all_teammates_norm_rows_${safe_task}"
  latest_dir=$(ls -td results/"$task"/heldout_xp_matrix/"$label"/* 2>/dev/null | head -n 1)

  if [ -n "$latest_dir" ] && [ -f "$latest_dir/returned_episode_returns_mean_normalized=True.csv" ]; then
    python3 evaluation/plot_xp_csv_tsne.py "$latest_dir" --publication --embedding cols
  fi
done
```

## 4) Generate Final Publication Meta-Figure (Option A)

This creates one subplot per task and outputs the final figure:
- results/publication/br_tsne_meta_option_a_kde_contour.png

```bash
cd /scratch/cluster/montek/projects/jax-aht

LBF_CSV=$(ls -td results/lbf/heldout_xp_matrix/all_teammates_norm_rows_lbf/* 2>/dev/null | head -n 1)/returned_episode_returns_mean_normalized=True.csv
CRAMPED_CSV=$(ls -td results/overcooked-v1/cramped_room/heldout_xp_matrix/all_teammates_norm_rows_overcooked-v1_cramped_room/* 2>/dev/null | head -n 1)/returned_episode_returns_mean_normalized=True.csv
ASYMM_CSV=$(ls -td results/overcooked-v1/asymm_advantages/heldout_xp_matrix/all_teammates_norm_rows_overcooked-v1_asymm_advantages/* 2>/dev/null | head -n 1)/returned_episode_returns_mean_normalized=True.csv
COUNTER_CSV=$(ls -td results/overcooked-v1/counter_circuit/heldout_xp_matrix/all_teammates_norm_rows_overcooked-v1_counter_circuit/* 2>/dev/null | head -n 1)/returned_episode_returns_mean_normalized=True.csv
COORD_CSV=$(ls -td results/overcooked-v1/coord_ring/heldout_xp_matrix/all_teammates_norm_rows_overcooked-v1_coord_ring/* 2>/dev/null | head -n 1)/returned_episode_returns_mean_normalized=True.csv
FORCED_CSV=$(ls -td results/overcooked-v1/forced_coord/heldout_xp_matrix/all_teammates_norm_rows_overcooked-v1_forced_coord/* 2>/dev/null | head -n 1)/returned_episode_returns_mean_normalized=True.csv

python3 evaluation/plot_tsne_meta_figure.py \
  "$LBF_CSV" \
  "$CRAMPED_CSV" \
  "$ASYMM_CSV" \
  "$COUNTER_CSV" \
  "$COORD_CSV" \
  "$FORCED_CSV" \
  --titles LBF cramped_room asymm_advantages counter_circuit coord_ring forced_coord \
  --out results/publication/br_tsne_meta_option_a_kde_contour.png \
  --preset journal-wide \
  --density-backdrop kde \
  --density-contours \
  --no-subplot-axis-ticks \
  --hide-subplot-axis-labels \
  --global-xlabel "t-SNE 1" \
  --global-ylabel "t-SNE 2"
```

## 5) Generate Alternative Style Variants (Optional)

Use the same CSV variables from Step 4.

Option B (subtle hexbin backdrop):

```bash
python3 evaluation/plot_tsne_meta_figure.py \
  "$LBF_CSV" \
  "$CRAMPED_CSV" \
  "$ASYMM_CSV" \
  "$COUNTER_CSV" \
  "$COORD_CSV" \
  "$FORCED_CSV" \
  --titles LBF cramped_room asymm_advantages counter_circuit coord_ring forced_coord \
  --out results/publication/br_tsne_meta_option_b_hexbin.png \
  --preset journal-wide \
  --density-backdrop hexbin \
  --no-subplot-axis-ticks \
  --hide-subplot-axis-labels \
  --global-xlabel "t-SNE 1" \
  --global-ylabel "t-SNE 2"
```

Option C (scatter only):

```bash
python3 evaluation/plot_tsne_meta_figure.py \
  "$LBF_CSV" \
  "$CRAMPED_CSV" \
  "$ASYMM_CSV" \
  "$COUNTER_CSV" \
  "$COORD_CSV" \
  "$FORCED_CSV" \
  --titles LBF cramped_room asymm_advantages counter_circuit coord_ring forced_coord \
  --out results/publication/br_tsne_meta_option_c_scatter.png \
  --preset journal-wide \
  --density-backdrop none \
  --no-subplot-axis-ticks \
  --hide-subplot-axis-labels \
  --global-xlabel "t-SNE 1" \
  --global-ylabel "t-SNE 2"
```
