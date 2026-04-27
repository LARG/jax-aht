#!/bin/bash

# Trajectory Pipeline Runner
# =========================
#
# This script runs the complete trajectory analysis pipeline in a tmux session:
# 1. collect_trajectories.py - Collects trajectory data from agent pairs
# 2. train_autoencoder.py - Trains an LSTM autoencoder on the trajectories
# 3. visualize_trajectories.py - Creates t-SNE visualizations of trajectory latents
#
# The scripts run sequentially - each step waits for the previous one to complete
# before starting, ensuring proper dependency management without fixed sleep times.
#
# Usage:
#   ./run_trajectory_pipeline.sh
#
# The script will:
# - Start a tmux session named "trajectory_pipeline"
# - Activate the jax-aht conda environment
# - Run each script in sequence with reasonable default parameters
# - Keep the tmux session running so you can monitor progress
#
# To monitor progress:
#   tmux attach -t trajectory_pipeline
#
# To check if it's still running:
#   tmux ls
#
# To kill the session:
#   tmux kill-session -t trajectory_pipeline
#
# Configuration variables are at the top of this script - modify as needed.

set -e  # Exit on any error

# Check if tmux is available
if ! command -v tmux &> /dev/null; then
    echo "Error: tmux is not installed. Please install tmux first."
    exit 1
fi

# Check if session already exists
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "Warning: tmux session '${SESSION_NAME}' already exists."
    echo "To attach: tmux attach -t ${SESSION_NAME}"
    echo "To kill and restart: tmux kill-session -t ${SESSION_NAME} && $0"
    exit 1
fi

# Configuration - adjust these as needed
SESSION_NAME="lbf_pipeline"
ENV_NAME="lbf"
DATA_DIR="results/${ENV_NAME}/trajectory_data"
MODEL_DIR="results/${ENV_NAME}/autoencoder_models"
OUTPUT_FILE="results/${ENV_NAME}/tsne_trajectory_visualization.png"

# Default parameters for the scripts
K=3  # Number of rollouts per agent pair
NUM_ENVS=8192  # Number of parallel environments
ROLLOUT_STEPS=128  # Steps per rollout
HIDDEN_DIM=128  # Autoencoder hidden dimension
LATENT_DIM=16  # Autoencoder latent dimension
LEARNING_RATE=0.0003  # Learning rate
NUM_EPOCHS=200  # Training epochs
BATCH_SIZE=64  # Training batch size

echo "Starting trajectory pipeline in tmux session: ${SESSION_NAME}"
echo "Environment: ${ENV_NAME}"
echo "Data directory: ${DATA_DIR}"
echo "Model directory: ${MODEL_DIR}"
echo "Output file: ${OUTPUT_FILE}"
echo ""
echo "The pipeline will run the following steps sequentially:"
echo "1. Activate conda environment (jax-aht)"
echo "2. Collect trajectories from agent pairs"
echo "3. Train LSTM autoencoder on trajectories"
echo "4. Generate t-SNE visualizations"
echo ""
echo "Each step will wait for the previous one to complete before starting."

# Start tmux session
tmux new-session -d -s "${SESSION_NAME}"

# Send commands to activate conda environment and run the pipeline sequentially
tmux send-keys -t "${SESSION_NAME}" "source /scratch/cluster/adityam/miniconda3/etc/profile.d/conda.sh && conda activate jax-aht && cd /scratch/cluster/adityam/jax-aht && echo 'Starting trajectory collection...' && python evaluation/collect_trajectories.py --env_name ${ENV_NAME} --k ${K} --num_envs ${NUM_ENVS} --rollout_steps ${ROLLOUT_STEPS} --data_dir ${DATA_DIR} && echo 'Trajectory collection complete. Starting autoencoder training...' && python evaluation/train_autoencoder.py --data_dir ${DATA_DIR} --model_dir ${MODEL_DIR} --env_name ${ENV_NAME} --hidden_dim ${HIDDEN_DIM} --latent_dim ${LATENT_DIM} --learning_rate ${LEARNING_RATE} --num_epochs ${NUM_EPOCHS} --batch_size ${BATCH_SIZE} && echo 'Autoencoder training complete. Starting visualization...' && python evaluation/visualize_trajectories.py --data_dir ${DATA_DIR} --model_dir ${MODEL_DIR} --output_file ${OUTPUT_FILE} --env_name ${ENV_NAME} --k 5 --num_envs 256 --rollout_steps 128 && echo 'Pipeline complete!'" C-m

echo "Pipeline started in tmux session '${SESSION_NAME}'"
echo "The commands will execute sequentially - each step waits for the previous to complete."
echo ""
echo "To attach to the session: tmux attach -t ${SESSION_NAME}"
echo "To check if it's still running: tmux ls"
echo "To kill the session: tmux kill-session -t ${SESSION_NAME}"