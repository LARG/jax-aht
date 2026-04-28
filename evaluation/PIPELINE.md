# Trajectory Collection, Classifier Training, and Visualization Pipeline

This directory contains three independent scripts for collecting trajectories, training a recurrent trajectory classifier (supervised, predicts agent type from observations), and visualizing results via t-SNE. They can be run separately and at different times. Results are saved under `results/<env>/` for organization.

## Directory Structure

```
results/<env>/
├── trajectory_data/                      # Collected trajectories (collect_trajectories.py)
│   └── heldout_episodes.pkl              # Heldout pairwise trajectories with pair labels
├── autoencoder_models/                   # Trained models (train_autoencoder.py)
│   ├── autoencoder.pkl
│   ├── training_losses.npy
│   └── loss_curve.png
├── latents.pkl                           # Encoded latents (visualize_trajectories.py)
└── tsne_trajectory_visualization.png     # t-SNE plot (visualize_trajectories.py)
```

## Workflow

### Step 1: Collect Trajectories

Collect and save pairwise heldout trajectory data:

```bash
python evaluation/collect_trajectories.py \
    --env_name lbf \
    --k 1 \
    --num_envs 2048 \
    --rollout_steps 64 \
    --data_dir results/lbf/trajectory_data
```

**Options:**
- `--env_name`: Environment name (default: `"lbf"`)
- `--k`: Number of rollouts per agent pair (default: `1`)
- `--num_envs`: Number of parallel environments (default: `2048`)
- `--rollout_steps`: Steps per rollout (default: `64`)
- `--data_dir`: Directory to save trajectory data (default: `"results/lbf/trajectory_data"`)

**Output:**
- `heldout_episodes.pkl` — heldout pairwise trajectories with embedded pair labels

### Step 2: Train Classifier

Train the recurrent LSTM classifier on the collected trajectories. The model predicts agent-pair type directly from observations (no reconstruction objective):

```bash
python evaluation/train_autoencoder.py \
    --data_dir results/lbf/trajectory_data \
    --model_dir results/lbf/autoencoder_models \
    --env_name lbf \
    --hidden_dim 64 \
    --latent_dim 16 \
    --learning_rate 3e-4 \
    --num_epochs 200 \
    --batch_size 512 \
    --max_samples_per_class 5000
```

**Options:**
- `--data_dir`: Directory containing trajectory data
- `--model_dir`: Directory to save trained model (default: `"results/lbf/autoencoder_models"`)
- `--env_name`: Environment name for `obs_dim` inference
- `--hidden_dim`: LSTM hidden dimension (default: `64`)
- `--latent_dim`: Latent/embedding dimension (default: `16`)
- `--learning_rate`: Learning rate (default: `3e-4`)
- `--num_epochs`: Number of training epochs (default: `200`)
- `--batch_size`: Batch size (default: `512`)
- `--max_samples_per_class`: Max training episodes per class; `0` = use all data (default: `5000`)

**Output:**
- `autoencoder.pkl` — model params and config
- `training_losses.npy` — per-epoch loss array
- `loss_curve.png` — training loss plot

### Step 3: Visualize Results

Encode trajectories into latents and create a t-SNE visualization:

```bash
python evaluation/visualize_trajectories.py \
    --data_dir results/lbf/trajectory_data \
    --model_dir results/lbf/autoencoder_models \
    --model_file autoencoder.pkl \
    --output_file results/lbf/tsne_trajectory_visualization.png \
    --latents_file results/lbf/latents.pkl
```

To skip re-encoding and plot from previously saved latents:

```bash
python evaluation/visualize_trajectories.py \
    --latents_file results/lbf/latents.pkl \
    --output_file results/lbf/tsne_trajectory_visualization.png \
    --plot-only
```

**Options:**
- `--data_dir`: Directory containing trajectory data
- `--model_dir`: Directory containing trained model
- `--model_file`: Trained model filename (default: `"autoencoder.pkl"`)
- `--output_file`: Output visualization filename (default: `"results/lbf/tsne_trajectory_visualization.png"`)
- `--latents_file`: Path to save/load encoded latents (default: `"results/lbf/latents.pkl"`)
- `--plot-only`: Skip encoding; plot directly from a saved `latents_file`
- `--env_name`: Environment name (default: `"lbf"`)
- `--k`: Number of rollouts per agent pair (default: `5`)
- `--num_envs`: Number of parallel environments (default: `256`)
- `--rollout_steps`: Steps per rollout (default: `128`)

**Output:**
- `latents.pkl` — encoded latent vectors with labels
- `tsne_trajectory_visualization.png` — t-SNE visualization

## Running the Full Pipeline (tmux)

Shell scripts are provided to run all three steps sequentially in a tmux session:

```bash
# LBF
./evaluation/run_lbf_pipeline.sh

# Overcooked coord_ring
./evaluation/run_overcooked_coord_ring_pipeline.sh
```

Both scripts use sensible defaults (`K=3`, `NUM_ENVS=8192`, `ROLLOUT_STEPS=128`, `HIDDEN_DIM=128`, `LATENT_DIM=16`). Edit the configuration block at the top of each script to adjust parameters.

To monitor progress after launching:

```bash
tmux attach -t lbf_pipeline                    # or overcooked_coord_ring_pipeline
tmux kill-session -t lbf_pipeline              # to stop
```