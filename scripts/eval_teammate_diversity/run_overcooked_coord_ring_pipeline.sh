#!/bin/bash

# Trajectory Pipeline Runner - Overcooked coord_ring
# ==================================================
#
# This script runs the complete trajectory analysis pipeline in a tmux session
# for the overcooked-v1/coord_ring environment:
# 1. collect_trajectories.py - Collects trajectory data from agent pairs
# 2. train_autoencoder.py - Trains an LSTM autoencoder on the trajectories
# 3. visualize_trajectories.py - Creates t-SNE visualizations of trajectory latents
#
# The scripts run sequentially - each step waits for the previous one to complete
# before starting, ensuring proper dependency management without fixed sleep times.
#
# Usage:
#   ./run_overcooked_coord_ring_pipeline.sh
#
# The script will:
# - Start a tmux session named "overcooked_coord_ring_pipeline"
# - Activate the jax-aht conda environment
# - Run each script in sequence with reasonable default parameters
# - Keep the tmux session running so you can monitor progress
#
# To monitor progress:
#   tmux attach -t overcooked_coord_ring_pipeline
#
# To check if it's still running:
#   tmux ls
#
# To kill the session:
#   tmux kill-session -t overcooked_coord_ring_pipeline
#
# Configuration variables are at the top of this script - modify as needed.

set -e  # Exit on any error

SKIP_COLLECT=false
for arg in "$@"; do
    case $arg in
        --skip-collect) SKIP_COLLECT=true ;;
    esac
done

# Resolve project root and conda base dynamically so this script works for any clone
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [ -n "${CONDA_EXE}" ]; then
    CONDA_BASE="$(dirname "$(dirname "${CONDA_EXE}")")"
elif command -v conda &> /dev/null; then
    CONDA_BASE="$(conda info --base)"
else
    echo "Error: conda not found. Please ensure conda is initialized (run 'conda init') and re-open your shell."
    exit 1
fi

# Check if tmux is available
if ! command -v tmux &> /dev/null; then
    echo "Error: tmux is not installed. Please install tmux first."
    exit 1
fi

# Configuration - adjust these as needed
SESSION_NAME="overcooked_coord_ring_pipeline"
CONDA_ENV_NAME="jax-aht"
TASK_NAME="overcooked-v1/coord_ring"

# Load ENV_NAME from task config using conda python
CONDA_PYTHON="${CONDA_BASE}/envs/${CONDA_ENV_NAME}/bin/python"
TASK_CONFIG="${PROJECT_ROOT}/evaluation/configs/task/${TASK_NAME}.yaml"
ENV_NAME=$(${CONDA_PYTHON} -c "import yaml; c=yaml.safe_load(open('${TASK_CONFIG}')); print(c['ENV_NAME'])")

# Check if session already exists
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "Warning: tmux session '${SESSION_NAME}' already exists."
    echo "To attach: tmux attach -t ${SESSION_NAME}"
    echo "To kill and restart: tmux kill-session -t ${SESSION_NAME} && $0"
    exit 1
fi
DATA_DIR="results/${TASK_NAME}/trajectory_data"
MODEL_DIR="results/${TASK_NAME}/models"
OUTPUT_FILE="results/${TASK_NAME}/tsne_trajectory_visualization.png"
LATENTS_FILE="results/${TASK_NAME}/latents.pkl"

# Default parameters for the scripts
NUM_POINTS_PER_PAIR=20  # Episodes collected per specific-BR pair for the validation / t-SNE set
NUM_ENVS=48  # Number of parallel environments (keeps trajectory pkl under 15 GB: 48 envs × 289 pairs × 400 steps × 650 floats × 4 bytes ≈ 14.4 GB)
HIDDEN_DIM=64  # Autoencoder hidden dimension
LATENT_DIM=16  # Autoencoder latent dimension
LEARNING_RATE=0.0005  # Learning rate
NUM_EPOCHS=200  # Training epochs
BATCH_SIZE=1024  # Training batch size

echo "Starting trajectory pipeline in tmux session: ${SESSION_NAME}"
echo "Task: ${TASK_NAME} (env: ${ENV_NAME})"
echo "Data directory: ${DATA_DIR}"
echo "Model directory: ${MODEL_DIR}"
echo "Output file: ${OUTPUT_FILE}"
echo "Skip collect: ${SKIP_COLLECT}"
echo ""
echo "The pipeline will run the following steps sequentially:"
echo "1. Activate conda environment (${CONDA_ENV_NAME})"
if [ "${SKIP_COLLECT}" = false ]; then
echo "2. Collect trajectories from agent pairs"
fi
echo "3. Train LSTM classifier on trajectories"
echo "4. Generate t-SNE visualizations"
echo ""
echo "Each step will wait for the previous one to complete before starting."

# Start tmux session
tmux new-session -d -s "${SESSION_NAME}"

# Build collect step conditionally
if [ "${SKIP_COLLECT}" = true ]; then
    COLLECT_CMD=""
else
    COLLECT_CMD="echo 'Starting trajectory collection...' && CUDA_VISIBLE_DEVICES=1 python scripts/eval_teammate_diversity/collect_trajectories.py --task_name ${TASK_NAME} --num_points_per_pair ${NUM_POINTS_PER_PAIR} --num_envs ${NUM_ENVS} --data_dir ${DATA_DIR} && echo 'Trajectory collection complete.' &&"
fi

# Send commands to activate conda environment and run the pipeline sequentially
tmux send-keys -t "${SESSION_NAME}" "source ${CONDA_BASE}/etc/profile.d/conda.sh && conda activate ${CONDA_ENV_NAME} && cd ${PROJECT_ROOT} && ${COLLECT_CMD} echo 'Starting classifier training...' && CUDA_VISIBLE_DEVICES=1 python scripts/eval_teammate_diversity/train_classifier.py --task_name ${TASK_NAME} --data_dir ${DATA_DIR} --model_dir ${MODEL_DIR} --hidden_dim ${HIDDEN_DIM} --latent_dim ${LATENT_DIM} --learning_rate ${LEARNING_RATE} --num_epochs ${NUM_EPOCHS} --batch_size ${BATCH_SIZE} && echo 'Classifier training complete. Starting visualization...' && CUDA_VISIBLE_DEVICES=1 python scripts/eval_teammate_diversity/visualize_trajectories.py --data_dir ${DATA_DIR} --model_dir ${MODEL_DIR} --output_file ${OUTPUT_FILE} --latents_file ${LATENTS_FILE} --plot_title 'Overcooked-V1 (coord_ring)' && echo 'Pipeline complete!'" C-m

echo "Pipeline started in tmux session '${SESSION_NAME}'"
echo "The commands will execute sequentially - each step waits for the previous to complete."
echo ""
echo "To attach to the session: tmux attach -t ${SESSION_NAME}"
echo "To check if it's still running: tmux ls"
echo "To kill the session: tmux kill-session -t ${SESSION_NAME}"