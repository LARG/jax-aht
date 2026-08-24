# Trajectory Collection, Classifier Training, and Visualization Pipeline

This directory contains three independent scripts for collecting trajectories, training a recurrent trajectory classifier (supervised, predicts agent type from observations), and visualizing results via t-SNE. They can be run separately and at different times. Results are saved under `results/<env>/` for organization.

## Train/Validation Split

Trajectories are split into two disjoint sets at collection time:

- **Training set** (`train_episodes.pkl`) — one rollout per agent pair (all pairs). Used exclusively by `train_classifier.py`.
- **Validation set** (`val_episodes.pkl`) — a fixed number of additional episodes collected only for pairs where an agent plays against its specific best response. Used exclusively by `visualize_trajectories.py` for t-SNE. These episodes are never seen during training.

The number of validation episodes per specific-BR pair is controlled by `--num_points_per_pair`. Training episode counts are kept uniform across all pairs by design (one rollout batch of `num_envs` environments each), with `--max_samples_per_class` providing an additional cap during training.

## Directory Structure

```
results/<env>/
├── trajectory_data/                      # Collected trajectories (collect_trajectories.py)
│   ├── train_episodes.pkl                # Training trajectories (all agent pairs)
│   └── val_episodes.pkl                  # Validation trajectories (specific-BR pairs only)
├── models/                               # Trained models (train_classifier.py)
│   ├── trajectory_classifier
│   ├── training_losses.npy
│   └── loss_curve.png
├── latents.pkl                           # Encoded latents (visualize_trajectories.py)
└── tsne_trajectory_visualization.png     # t-SNE plot (visualize_trajectories.py)
```

## Workflow

### Step 1: Collect Trajectories

Collect and save pairwise trajectory data. One rollout batch (of `num_envs` parallel environments) is run for every agent pair and stored as the training set. For pairs where an agent plays against its specific best response, `num_points_per_pair` additional episodes are collected for the validation set.

```bash
python scripts/eval_teammate_diversity/collect_trajectories.py \
    --env_name overcooked-v1/coord_ring \
    --num_envs 2048 \
    --rollout_steps 450 \
    --num_points_per_pair 2048 \
    --data_dir results/overcooked-v1/coord_ring/trajectory_data
```

**Options:**
- `--env_name`: Environment name (default: `"overcooked-v1/coord_ring"`)
- `--num_envs`: Number of parallel environments per rollout batch (default: `2048`)
- `--rollout_steps`: Steps per rollout (default: auto-selected by env; `450` for overcooked, `128` otherwise)
- `--num_points_per_pair`: Episodes to collect per specific-BR pair for the validation set (default: disabled)
- `--data_dir`: Directory to save trajectory data (default: `"results/overcooked-v1/coord_ring/trajectory_data"`)

**Output:**
- `train_episodes.pkl` — training trajectories with pair labels (all pairs, one rollout each)
- `val_episodes.pkl` — validation trajectories with pair labels (specific-BR pairs only, `num_points_per_pair` episodes each)

### Step 2: Train Classifier

Train the recurrent LSTM classifier on the training trajectories. The model predicts agent-pair type directly from observations (no reconstruction objective):

```bash
python scripts/eval_teammate_diversity/train_classifier.py \
    --data_dir results/overcooked-v1/coord_ring/trajectory_data \
    --model_dir results/overcooked-v1/coord_ring/models \
    --env_name overcooked-v1/coord_ring \
    --hidden_dim 64 \
    --latent_dim 16 \
    --learning_rate 3e-4 \
    --num_epochs 200 \
    --batch_size 512 \
    --max_samples_per_class 5000
```

**Options:**
- `--data_dir`: Directory containing trajectory data
- `--model_dir`: Directory to save trained model (default: `"results/overcooked-v1/coord_ring/models"`)
- `--env_name`: Environment name for `obs_dim` inference
- `--hidden_dim`: LSTM hidden dimension (default: `64`)
- `--latent_dim`: Latent/embedding dimension (default: `16`)
- `--learning_rate`: Learning rate (default: `3e-4`)
- `--num_epochs`: Number of training epochs (default: `200`)
- `--batch_size`: Batch size (default: `512`)
- `--max_samples_per_class`: Max training episodes per class; `0` = use all data (default: `5000`)

**Output:**
- `trajectory_classifier` — model params and config
- `training_losses.npy` — per-epoch loss array
- `loss_curve.png` — training loss plot

### Step 3: Visualize Results

Encode validation trajectories into latents and create a t-SNE visualization. Only specific-BR pair trajectories from `val_episodes.pkl` are used — these were never seen during training.

```bash
python scripts/eval_teammate_diversity/visualize_trajectories.py \
    --data_dir results/overcooked-v1/coord_ring/trajectory_data \
    --model_dir results/overcooked-v1/coord_ring/models \
    --model_file trajectory_classifier \
    --output_file results/overcooked-v1/coord_ring/tsne_trajectory_visualization.png \
    --latents_file results/overcooked-v1/coord_ring/latents.pkl
```

To skip re-encoding and plot from previously saved latents:

```bash
python scripts/eval_teammate_diversity/visualize_trajectories.py \
    --latents_file results/overcooked-v1/coord_ring/latents.pkl \
    --output_file results/overcooked-v1/coord_ring/tsne_trajectory_visualization.png \
    --plot-only
```

**Options:**
- `--data_dir`: Directory containing trajectory data
- `--model_dir`: Directory containing trained model
- `--model_file`: Trained model filename (default: `"trajectory_classifier"`)
- `--output_file`: Output visualization filename (default: `"results/lbf/tsne_trajectory_visualization.png"`)
- `--latents_file`: Path to save/load encoded latents (default: `"results/lbf/latents.pkl"`)
- `--plot-only`: Skip encoding; plot directly from a saved `latents_file`

**Output:**
- `latents.pkl` — encoded latent vectors with labels
- `tsne_trajectory_visualization.png` — t-SNE visualization

## Running the Full Pipeline (tmux)

Shell scripts are provided to run all three steps sequentially in a tmux session:

```bash
# LBF
./scripts/eval_teammate_diversity/run_lbf_pipeline.sh

# Overcooked coord_ring
./scripts/eval_teammate_diversity/run_overcooked_coord_ring_pipeline.sh
```

Both scripts use sensible defaults (`NUM_ENVS`, `ROLLOUT_STEPS`, `NUM_POINTS_PER_PAIR`, `HIDDEN_DIM`, `LATENT_DIM`). Edit the configuration block at the top of each script to adjust parameters.

To monitor progress after launching:

```bash
tmux attach -t lbf_pipeline                    # or overcooked_coord_ring_pipeline
tmux kill-session -t lbf_pipeline              # to stop
```
