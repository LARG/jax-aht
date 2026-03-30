"""Visualize trajectory encodings using a trained autoencoder."""

import argparse
import pickle
from pathlib import Path

import jax
import numpy as np

from evaluation.trajectory_autoencoder import encode_episodes, pad_episodes
from evaluation.trajectory_plot import plot_tsne

# Config
DEFAULT_DATA_DIR = "results/lbf/trajectory_data"
DEFAULT_MODEL_DIR = "results/lbf/autoencoder_models"
DEFAULT_MODEL_FILE = "autoencoder.pkl"
DEFAULT_OUTPUT_FILE = "results/lbf/tsne_trajectory_visualization.png"


def main(
    data_dir=DEFAULT_DATA_DIR,
    model_dir=DEFAULT_MODEL_DIR,
    model_file=DEFAULT_MODEL_FILE,
    output_file=DEFAULT_OUTPUT_FILE,
):
    """Visualize saved trajectories using a trained autoencoder."""
    data_path = Path(data_dir)
    model_path = Path(model_dir) / model_file

    # Load trajectories
    heldout_path = data_path / "heldout_episodes.pkl"
    ippo_path = data_path / "ippo_episodes.pkl"

    if not heldout_path.exists() or not ippo_path.exists():
        raise FileNotFoundError(
            f"Trajectory data not found. Ensure both {heldout_path} and {ippo_path} exist. "
            "Run collect_trajectories.py first."
        )

    print(f"Loading trajectories from {data_path}...")
    with open(heldout_path, "rb") as f:
        heldout_episodes = pickle.load(f)
    with open(ippo_path, "rb") as f:
        ippo_episodes = pickle.load(f)
    print(f"Loaded {len(heldout_episodes)} heldout and {len(ippo_episodes)} IPPO episodes.")

    # Load trained model
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Run train_autoencoder.py first."
        )

    print(f"Loading trained model from {model_path}...")
    with open(model_path, "rb") as f:
        checkpoint = pickle.load(f)

    train_state = checkpoint["train_state"]
    model = checkpoint["model"]
    config = checkpoint["config"]
    max_seq_len = config["max_seq_len"]

    print(f"Model config: {config}")

    # Encode episodes
    print("Encoding trajectories...")
    heldout_latents = encode_episodes(model, train_state, heldout_episodes, max_seq_len)
    ippo_latents = encode_episodes(model, train_state, ippo_episodes, max_seq_len)

    # Plot
    print(f"Creating t-SNE visualization...")
    plot_tsne(
        {"heldout": heldout_latents, "ippo_selfplay": ippo_latents},
        save_path=output_file,
    )
    print(f"Visualization saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize trajectories using trained autoencoder.")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="Directory containing trajectory data")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR, help="Directory containing trained model")
    parser.add_argument("--model_file", type=str, default=DEFAULT_MODEL_FILE, help="Trained model filename")
    parser.add_argument("--output_file", type=str, default=DEFAULT_OUTPUT_FILE, help="Output visualization filename")

    args = parser.parse_args()
    main(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        model_file=args.model_file,
        output_file=args.output_file,
    )
