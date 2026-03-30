"""Train the trajectory autoencoder on saved trajectory data."""

import argparse
import pickle
from pathlib import Path

import jax
import numpy as np

from evaluation.trajectory_autoencoder import (
    init_autoencoder,
    make_train_step,
    train_autoencoder,
    pad_episodes,
)
# Config
DEFAULT_DATA_DIR = "results/lbf/trajectory_data"
DEFAULT_MODEL_DIR = "results/lbf/autoencoder_models"
DEFAULT_ENV_NAME = "lbf"
DEFAULT_HIDDEN_DIM = 64
DEFAULT_LATENT_DIM = 128
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_NUM_EPOCHS = 200
DEFAULT_BATCH_SIZE = 64


def get_obs_dim(env_name):
    """Get observation dimension for an environment."""
    from envs import make_env
    env = make_env(env_name, {})
    return env.observation_space("agent_0").shape[0]


def main(
    data_dir=DEFAULT_DATA_DIR,
    model_dir=DEFAULT_MODEL_DIR,
    env_name=DEFAULT_ENV_NAME,
    hidden_dim=DEFAULT_HIDDEN_DIM,
    latent_dim=DEFAULT_LATENT_DIM,
    learning_rate=DEFAULT_LEARNING_RATE,
    num_epochs=DEFAULT_NUM_EPOCHS,
    batch_size=DEFAULT_BATCH_SIZE,
):
    """Train autoencoder on saved trajectories."""
    data_path = Path(data_dir)
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)

    # Load trajectories
    heldout_path = data_path / "heldout_episodes.pkl"
    if not heldout_path.exists():
        raise FileNotFoundError(f"Heldout episodes not found at {heldout_path}. Run collect_trajectories.py first.")

    print(f"Loading trajectories from {heldout_path}...")
    with open(heldout_path, "rb") as f:
        all_episodes = pickle.load(f)
    print(f"Loaded {len(all_episodes)} episodes.")

    obs_dim = get_obs_dim(env_name)
    padded_episodes, masks, max_seq_len = pad_episodes(all_episodes)

    rng = jax.random.PRNGKey(42)
    rng, train_state, model = init_autoencoder(
        rng,
        obs_dim,
        max_seq_len,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        learning_rate=learning_rate,
    )
    train_step = make_train_step(model, obs_dim)

    print(f"Training on {len(all_episodes)} episodes (padded length {max_seq_len})")
    print(f"Model config: hidden_dim={hidden_dim}, latent_dim={latent_dim}")
    rng, train_state = train_autoencoder(
        rng,
        train_state,
        train_step,
        padded_episodes,
        masks,
        num_epochs=num_epochs,
        batch_size=batch_size,
    )

    print("Training complete. Saving model...")
    
    # Save only the parameters and config (JAX model objects aren't pickleable)
    checkpoint = {
        "params": train_state.params,
        "config": {
            "hidden_dim": hidden_dim,
            "latent_dim": latent_dim,
            "obs_dim": obs_dim,
            "max_seq_len": max_seq_len,
        }
    }
    
    model_file = model_path / "autoencoder.pkl"
    with open(model_file, "wb") as f:
        pickle.dump(checkpoint, f)
    print(f"Model saved to {model_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train trajectory autoencoder on saved data.")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="Directory containing trajectory data")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR, help="Directory to save trained model")
    parser.add_argument("--env_name", type=str, default=DEFAULT_ENV_NAME, help="Environment name")
    parser.add_argument("--hidden_dim", type=int, default=DEFAULT_HIDDEN_DIM, help="Hidden dimension")
    parser.add_argument("--latent_dim", type=int, default=DEFAULT_LATENT_DIM, help="Latent dimension")
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=DEFAULT_NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")

    args = parser.parse_args()
    main(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        env_name=args.env_name,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
    )
