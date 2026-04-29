"""Collect trajectories from training pairs and save to disk."""

import argparse
import pickle
from pathlib import Path

import jax
import numpy as np

from envs import make_env
from trajectory_collection import (
    collect_heldout_pairwise_trajectories,
)

# Config
DEFAULT_ENV_NAME = "overcooked-v1/coord_ring"
DEFAULT_ENV_KWARGS = {}
DEFAULT_NUM_ENVS = 2048
DEFAULT_ROLLOUT_STEPS = None  # auto-selected based on env when not provided
DEFAULT_NUM_POINTS_PER_PAIR = None  # episodes collected per specific-BR pair for val set; None disables val collection
DEFAULT_DATA_DIR = "results/overcooked-v1/coord_ring/trajectory_data"


def _default_rollout_steps(env_name: str) -> int:
    if env_name.startswith("overcooked"):
        return 450  # overcooked episodes last up to 400 steps
    return 128  # sufficient for LBF and other short-episode envs


def main(
    env_name=DEFAULT_ENV_NAME,
    num_points_per_pair=DEFAULT_NUM_POINTS_PER_PAIR,
    num_envs=DEFAULT_NUM_ENVS,
    rollout_steps=DEFAULT_ROLLOUT_STEPS,
    data_dir=DEFAULT_DATA_DIR,
):
    """Collect and save trajectories."""
    if rollout_steps is None:
        rollout_steps = _default_rollout_steps(env_name)

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    rng = jax.random.PRNGKey(42)
    env = make_env(env_name)

    print("Collecting pairwise heldout trajectories...")
    rng, train_episodes, val_episodes, pair_labels = collect_heldout_pairwise_trajectories(
        rng,
        env,
        num_points_per_pair=num_points_per_pair,
        rollout_steps=rollout_steps,
        num_envs=num_envs,
        env_name=env_name,
    )
    print(f"Collected {len(train_episodes)} train episodes, {len(val_episodes)} val episodes.")

    train_path = data_path / "train_episodes.pkl"
    with open(train_path, "wb") as f:
        pickle.dump({"episodes": train_episodes, "pair_labels": pair_labels}, f)
    print(f"Saved train episodes to {train_path}")

    val_path = data_path / "val_episodes.pkl"
    with open(val_path, "wb") as f:
        pickle.dump({"episodes": val_episodes, "pair_labels": pair_labels}, f)
    print(f"Saved val episodes to {val_path}")

    print(f"\nAll trajectories saved to {data_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect and save trajectories.")
    parser.add_argument("--env_name", type=str, default=DEFAULT_ENV_NAME, help="Environment name")
    parser.add_argument("--num_points_per_pair", type=int, default=None, help="Number of episodes to collect per specific-BR pair for the validation set (default: disabled)")
    parser.add_argument("--num_envs", type=int, default=DEFAULT_NUM_ENVS, help="Number of parallel environments")
    parser.add_argument("--rollout_steps", type=int, default=DEFAULT_ROLLOUT_STEPS, help="Steps per rollout")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="Directory to save trajectory data")

    args = parser.parse_args()
    main(
        env_name=args.env_name,
        num_points_per_pair=args.num_points_per_pair,
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        data_dir=args.data_dir,
    )
