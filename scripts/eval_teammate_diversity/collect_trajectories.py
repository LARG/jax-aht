"""Collect trajectories from training pairs and save to disk."""

import argparse
import pickle
from pathlib import Path

import jax
import numpy as np
import yaml

from envs import make_env
from trajectory_collection import (
    collect_heldout_pairwise_trajectories,
)

# Config
DEFAULT_TASK_NAME = "lbf/lbf_7x7_nolevels"
DEFAULT_NUM_ENVS = 2048
DEFAULT_NUM_POINTS_PER_PAIR = None  # episodes collected per specific-BR pair for val set; None disables val collection
DEFAULT_DATA_DIR = None  # auto-derived from task_name when not provided


def _load_task_config(task_name):
    config_path = Path("evaluation/configs/task") / f"{task_name}.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def main(
    task_name=DEFAULT_TASK_NAME,
    num_points_per_pair=DEFAULT_NUM_POINTS_PER_PAIR,
    num_envs=DEFAULT_NUM_ENVS,
    data_dir=DEFAULT_DATA_DIR,
):
    """Collect and save trajectories."""
    cfg = _load_task_config(task_name)
    env_name = cfg["ENV_NAME"]
    rollout_steps = cfg["ROLLOUT_LENGTH"]
    env_kwargs = cfg.get("ENV_KWARGS") or {}

    if data_dir is None:
        data_dir = f"results/{task_name}/trajectory_data"

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    rng = jax.random.PRNGKey(42)
    env = make_env(env_name, env_kwargs)

    print("Collecting pairwise heldout trajectories...")
    rng, train_episodes, val_episodes, pair_labels = collect_heldout_pairwise_trajectories(
        rng,
        env,
        num_points_per_pair=num_points_per_pair,
        rollout_steps=rollout_steps,
        num_envs=num_envs,
        task_name=task_name,
        env_kwargs=env_kwargs,
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
    parser.add_argument("--task_name", type=str, default=DEFAULT_TASK_NAME, help="Task name (e.g. lbf/lbf_7x7_nolevels)")
    parser.add_argument("--num_points_per_pair", type=int, default=None, help="Number of episodes to collect per specific-BR pair for the validation set (default: disabled)")
    parser.add_argument("--num_envs", type=int, default=DEFAULT_NUM_ENVS, help="Number of parallel environments")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="Directory to save trajectory data (default: results/<task_name>/trajectory_data)")

    args = parser.parse_args()
    main(
        task_name=args.task_name,
        num_points_per_pair=args.num_points_per_pair,
        num_envs=args.num_envs,
        data_dir=args.data_dir,
    )
