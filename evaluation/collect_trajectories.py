"""Collect trajectories from training pairs and save to disk."""

import argparse
import pickle
from pathlib import Path

import jax
import numpy as np

from envs import make_env
from evaluation.trajectory_collection import (
    collect_heldout_pairwise_trajectories,
    collect_ippo_selfplay_trajectories,
)

# Config
DEFAULT_ENV_NAME = "lbf"
DEFAULT_ENV_KWARGS = {}
DEFAULT_NUM_ENVS = 256
DEFAULT_ROLLOUT_STEPS = 128
DEFAULT_K = 5
DEFAULT_DATA_DIR = "results/lbf/trajectory_data"


def main(
    env_name=DEFAULT_ENV_NAME,
    k=DEFAULT_K,
    num_envs=DEFAULT_NUM_ENVS,
    rollout_steps=DEFAULT_ROLLOUT_STEPS,
    data_dir=DEFAULT_DATA_DIR,
):
    """Collect and save trajectories."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    rng = jax.random.PRNGKey(42)
    env = make_env(env_name, DEFAULT_ENV_KWARGS)

    print("Collecting pairwise heldout trajectories...")
    rng, heldout_episodes = collect_heldout_pairwise_trajectories(
        rng,
        env,
        k=k,
        rollout_steps=rollout_steps,
        num_envs=num_envs,
        env_name=env_name,
    )
    print(f"Collected {len(heldout_episodes)} heldout pairwise episodes.")
    
    # Verify that episodes contain agent indices
    if len(heldout_episodes) > 0:
        first_ep = heldout_episodes[0]
        if isinstance(first_ep, tuple) and len(first_ep) == 3:
            obs_array, agent_idx, br_idx = first_ep
            print(f"✓ Episodes contain agent pair indices: (agent_idx, br_idx)")
            
            # Extract and log unique agent pairs
            agent_indices = np.array([(ep[1], ep[2]) for ep in heldout_episodes])
            unique_pairs = np.unique(agent_indices, axis=0)
            print(f"  Found {len(unique_pairs)} unique agent-BR pairs:")
            for agent_idx, br_idx in unique_pairs:
                count = np.sum((agent_indices == [agent_idx, br_idx]).all(axis=1))
                print(f"    Agent {agent_idx} vs BR {br_idx}: {count} trajectories")
        else:
            print("  WARNING: Episodes do not contain agent indices (legacy format detected)")

    # Save heldout trajectories
    heldout_path = data_path / "heldout_episodes.pkl"
    with open(heldout_path, "wb") as f:
        pickle.dump(heldout_episodes, f)
    print(f"Saved heldout episodes to {heldout_path}")

    print("\nCollecting IPPO self-play trajectories...")
    rng, ippo_episodes = collect_ippo_selfplay_trajectories(
        rng,
        env,
        num_rollouts=k,
        rollout_steps=rollout_steps,
        num_envs=num_envs,
    )
    print(f"Collected {len(ippo_episodes)} IPPO self-play episodes.")
    
    # Verify that IPPO episodes contain agent indices
    if len(ippo_episodes) > 0:
        first_ep = ippo_episodes[0]
        if isinstance(first_ep, tuple) and len(first_ep) == 3:
            obs_array, agent_idx, br_idx = first_ep
            print(f"✓ IPPO episodes contain agent pair indices: (0, 0) for self-play")
        else:
            print("  WARNING: IPPO episodes do not contain agent indices (legacy format detected)")

    # Save IPPO trajectories
    ippo_path = data_path / "ippo_episodes.pkl"
    with open(ippo_path, "wb") as f:
        pickle.dump(ippo_episodes, f)
    print(f"Saved IPPO episodes to {ippo_path}")

    print(f"\nAll trajectories saved to {data_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect and save trajectories.")
    parser.add_argument("--env_name", type=str, default=DEFAULT_ENV_NAME, help="Environment name")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Number of rollouts per agent pair")
    parser.add_argument("--num_envs", type=int, default=DEFAULT_NUM_ENVS, help="Number of parallel environments")
    parser.add_argument("--rollout_steps", type=int, default=DEFAULT_ROLLOUT_STEPS, help="Steps per rollout")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="Directory to save trajectory data")

    args = parser.parse_args()
    main(
        env_name=args.env_name,
        k=args.k,
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        data_dir=args.data_dir,
    )
