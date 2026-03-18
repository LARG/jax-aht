"""Train an S5 trajectory autoencoder on random episodes and visualize random vs IPPO trajectories in t-SNE."""

import jax
import numpy as np

from envs import make_env
from evaluation.trajectory_autoencoder import (
    create_autoencoder,
    init_autoencoder,
    make_train_step,
    train_autoencoder,
    encode_episodes,
    pad_episodes,
)
from evaluation.trajectory_collection import collect_random_trajectories, collect_ippo_selfplay_trajectories
from evaluation.trajectory_plot import plot_tsne

# Config
ENV_NAME = "lbf"
ENV_KWARGS = {}
NUM_ENVS = 64
ROLLOUT_STEPS = 128
ROLLOUTS_PER_ITER = 5
NUM_ITERS = 10
D_MODEL = 64
SSM_SIZE = 64
SSM_N_LAYERS = 3
LATENT_DIM = 128
LEARNING_RATE = 3e-4
NUM_EPOCHS = 200
BATCH_SIZE = 64
MAX_BUFFER_SIZE = 1024


def main():
    rng = jax.random.PRNGKey(42)
    env = make_env(ENV_NAME, ENV_KWARGS)
    obs_dim = env.observation_space("agent_0").shape[0]

    all_episodes = []
    train_state = None
    model = None
    train_step = None
    max_seq_len = None

    for iteration in range(NUM_ITERS):
        print(f"Iteration {iteration+1}/{NUM_ITERS}: collecting random trajectories")
        rng, new_eps = collect_random_trajectories(
            rng,
            env,
            num_rollouts=ROLLOUTS_PER_ITER,
            rollout_steps=ROLLOUT_STEPS,
            num_envs=NUM_ENVS,
        )
        all_episodes.extend(new_eps)
        if len(all_episodes) > MAX_BUFFER_SIZE:
            all_episodes = all_episodes[-MAX_BUFFER_SIZE:]

        padded_episodes, masks, max_seq_len = pad_episodes(all_episodes)

        if train_state is None:
            rng, train_state, model = init_autoencoder(
                rng,
                obs_dim,
                max_seq_len,
                d_model=D_MODEL,
                ssm_size=SSM_SIZE,
                ssm_n_layers=SSM_N_LAYERS,
                latent_dim=LATENT_DIM,
                learning_rate=LEARNING_RATE,
            )
            train_step = make_train_step(model, obs_dim)

        print(f"Training on {len(all_episodes)} episodes (padded length {max_seq_len})")
        rng, train_state = train_autoencoder(
            rng,
            train_state,
            train_step,
            padded_episodes,
            masks,
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
        )

    print("Training complete.")

    print("Collecting IPPO self-play trajectories...")
    rng, ippo_episodes = collect_ippo_selfplay_trajectories(
        rng,
        env,
        num_rollouts=ROLLOUTS_PER_ITER,
        rollout_steps=ROLLOUT_STEPS,
        num_envs=NUM_ENVS,
    )

    print("Encoding episodes and plotting t-SNE...")
    random_latents = encode_episodes(model, train_state, all_episodes, max_seq_len)
    ippo_latents = encode_episodes(model, train_state, ippo_episodes, max_seq_len)
    plot_tsne(
        {"random": random_latents, "ippo_selfplay": ippo_latents},
        save_path="evaluation/tsne_random_vs_ippo.png",
    )


if __name__ == "__main__":
    main()
