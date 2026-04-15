"""Train an LSTM trajectory autoencoder on heldout pairwise agent trajectories and visualize random vs IPPO trajectories in t-SNE."""

import argparse
import jax
import numpy as np

from envs import make_env
from evaluation.trajectory_autoencoder import (
    init_autoencoder,
    make_train_step,
    train_autoencoder,
    encode_episodes,
    pad_episodes,
)
from evaluation.trajectory_collection import (
    collect_heldout_pairwise_trajectories,
    collect_ippo_selfplay_trajectories,
)
from evaluation.trajectory_plot import plot_tsne

# Config
DEFAULT_ENV_NAME = "lbf"
DEFAULT_ENV_KWARGS = {}
DEFAULT_NUM_ENVS = 64
DEFAULT_ROLLOUT_STEPS = 128
DEFAULT_ROLLOUTS_PER_ITER = 5
DEFAULT_NUM_ITERS = 10
DEFAULT_HIDDEN_DIM = 128
DEFAULT_LATENT_DIM = 128
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_NUM_EPOCHS = 200
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_BUFFER_SIZE = 1024
DEFAULT_K = 5


def main(env_name=DEFAULT_ENV_NAME, k=DEFAULT_K, num_envs=DEFAULT_NUM_ENVS, rollout_steps=DEFAULT_ROLLOUT_STEPS, num_epochs=DEFAULT_NUM_EPOCHS):
    rng = jax.random.PRNGKey(42)
    env = make_env(env_name, DEFAULT_ENV_KWARGS)
    obs_dim = env.observation_space("agent_0").shape[0]

    print("Collecting pairwise heldout trajectories for autoencoder training...")
    rng, all_episodes = collect_heldout_pairwise_trajectories(
        rng,
        env,
        k=k,
        rollout_steps=rollout_steps,
        num_envs=num_envs,
        env_name=env_name,
    )
    print(f"Collected {len(all_episodes)} heldout pairwise episodes.")

    padded_episodes, masks, max_seq_len, agent_indices = pad_episodes(all_episodes)
    
    # Log agent pair information if available
    if agent_indices is not None:
        unique_pairs = np.unique(agent_indices, axis=0)
        print(f"Found {len(unique_pairs)} unique agent pairs:")
        for agent_idx, br_idx in unique_pairs:
            count = np.sum((agent_indices == [agent_idx, br_idx]).all(axis=1))
            print(f"  Agent {agent_idx} vs BR {br_idx}: {count} trajectories")
    rng, train_state, model = init_autoencoder(
        rng,
        obs_dim,
        max_seq_len,
        hidden_dim=DEFAULT_HIDDEN_DIM,
        learning_rate=DEFAULT_LEARNING_RATE,
        latent_dim=DEFAULT_LATENT_DIM,
    )
    train_step = make_train_step(model, obs_dim)

    print(f"Training on {len(all_episodes)} episodes (padded length {max_seq_len})")
    rng, train_state = train_autoencoder(
        rng,
        train_state,
        train_step,
        padded_episodes,
        masks,
        num_epochs=num_epochs,
        batch_size=DEFAULT_BATCH_SIZE,
    )

    print("Training complete.")

    print("Collecting IPPO self-play trajectories...")
    rng, ippo_episodes = collect_ippo_selfplay_trajectories(
        rng,
        env,
        num_rollouts=DEFAULT_ROLLOUTS_PER_ITER,
        rollout_steps=rollout_steps,
        num_envs=num_envs,
    )

    print("Encoding episodes and plotting t-SNE...")
    random_latents = encode_episodes(model, train_state, all_episodes, max_seq_len)
    ippo_latents = encode_episodes(model, train_state, ippo_episodes, max_seq_len)
    plot_tsne(
        {"random": random_latents, "ippo_selfplay": ippo_latents},
        save_path="evaluation/tsne_random_vs_ippo.png",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM trajectory autoencoder on heldout pairwise trajectories and plot t-SNE.")
    parser.add_argument("--env_name", type=str, default=DEFAULT_ENV_NAME, help="Environment name")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Number of rollouts per agent pair")
    parser.add_argument("--num_envs", type=int, default=DEFAULT_NUM_ENVS, help="Number of parallel environments")
    parser.add_argument("--rollout_steps", type=int, default=DEFAULT_ROLLOUT_STEPS, help="Steps per rollout")
    parser.add_argument("--num_epochs", type=int, default=DEFAULT_NUM_EPOCHS, help="Number of training epochs")

    args = parser.parse_args()
    main(
        env_name=args.env_name,
        k=args.k,
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        num_epochs=args.num_epochs,
    )
