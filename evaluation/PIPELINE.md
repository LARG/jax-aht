# Trajectory Collection, Autoencoder Training, and Visualization Pipeline

This directory contains three independent scripts for collecting trajectories, training an LSTM-based trajectory autoencoder, and visualizing results. They can be run separately and at different times. All results are saved in `results/lbf/` for organization.

## Directory Structure

```
results/lbf/
├── trajectory_data/          # Collected trajectories (created by collect_trajectories.py)
│   ├── heldout_episodes.pkl
│   └── ippo_episodes.pkl
├── autoencoder_models/       # Trained models (created by train_autoencoder.py)
│   └── autoencoder.pkl
└── tsne_trajectory_visualization.png  # Visualization (created by visualize_trajectories.py)
```

## Workflow

### Step 1: Collect Trajectories

Collect and save trajectory data from agent pairs and IPPO self-play:

```bash
python evaluation/collect_trajectories.py \
    --env_name lbf \
    --k 5 \
    --num_envs 256 \
    --rollout_steps 128 \
    --data_dir results/lbf/trajectory_data
```

**Options:**
- `--env_name`: Environment name (default: "lbf")
- `--k`: Number of rollouts per agent pair (default: 5)
- `--num_envs`: Number of parallel environments (default: 256)
- `--rollout_steps`: Steps per rollout (default: 128)
- `--data_dir`: Directory to save trajectory data (default: "results/lbf/trajectory_data")

**Output:**
- `results/lbf/trajectory_data/heldout_episodes.pkl` - Heldout pairwise trajectories
- `results/lbf/trajectory_data/ippo_episodes.pkl` - IPPO self-play trajectories

### Step 2: Train Autoencoder

Train the S5 trajectory autoencoder on the collected trajectories:

```bash
python evaluation/train_autoencoder.py \
    --data_dir results/lbf/trajectory_data \
    --model_dir results/lbf/autoencoder_models \
    --env_name lbf \
    --hidden_dim 64 \
    --latent_dim 128 \
    --learning_rate 3e-4 \
    --num_epochs 200 \
    --batch_size 64
```

**Options:**
- `--data_dir`: Directory containing trajectory data
- `--model_dir`: Directory to save trained model (default: "results/lbf/autoencoder_models")
- `--env_name`: Environment name for obs_dim inference
- `--hidden_dim`: Hidden dimension (default: 64)
- `--latent_dim`: Latent dimension (default: 128)
- `--learning_rate`: Learning rate (default: 3e-4)
- `--num_epochs`: Number of training epochs (default: 200)
- `--batch_size`: Batch size (default: 64)

**Output:**
- `results/lbf/autoencoder_models/autoencoder.pkl` - Trained model and training state

### Step 3: Visualize Results

Create t-SNE visualization of trajectory encodings:

```bash
python evaluation/visualize_trajectories.py \
    --data_dir results/lbf/trajectory_data \
    --model_dir results/lbf/autoencoder_models \
    --model_file autoencoder.pkl \
    --output_file results/lbf/tsne_trajectory_visualization.png
```

**Options:**
- `--data_dir`: Directory containing trajectory data
- `--model_dir`: Directory containing trained model
- `--model_file`: Trained model filename (default: "autoencoder.pkl")
- `--output_file`: Output visualization filename

**Output:**
- `results/lbf/tsne_trajectory_visualization.png` - t-SNE visualization

## Benefits of This Pipeline

1. **Decoupled Stages**: Each stage can be run independently, allowing for:
   - Collecting more data without retraining
   - Training multiple autoencoders on the same data
   - Visualizing with different parameters

2. **Efficient Parallel Collection**: Trajectories can be collected in parallel with multiple GPUs by running `collect_trajectories.py` multiple times, then consolidating results

3. **Checkpoint & Resume**: If any stage fails, you can restart from that point without redoing previous work

4. **Experiment Iteration**: Easily experiment with different autoencoder architectures without recollecting data

## Example: Scale to 500,000 Trajectories

To collect 500,000 trajectories efficiently:

```bash
# Collect in baresults/lbf/trajectory_data/consolidated \
    --model_dir results/lbf
    python evaluation/collect_trajectories.py \
        --env_name lbf \
        --k 5 \
        --num_envs 256 \
        --rollout_steps 128 \
        --data_dir evaluation/trajectory_data/batch_$i
done

# Then consolidate and train (this requires additional consolidation script)
python evaluation/train_autoencoder.py \
    --data_dir evaluation/trajectory_data/consolidated \
    --model_dir evaluation/autoencoder_models
```
