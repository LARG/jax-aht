"""Visualize trajectory encodings using a trained autoencoder."""

import argparse
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState
import optax

from evaluation.trajectory_autoencoder import (
    create_autoencoder,
    encode_episodes,
    pad_episodes,
)
from evaluation.trajectory_collection import (
    collect_pair_trajectories,
    get_agent_pair_configs,
)
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
    env_name="lbf",
    k=5,
    num_envs=256,
    rollout_steps=128,
):
    """Visualize saved trajectories using a trained autoencoder."""
    data_path = Path(data_dir)
    model_path = Path(model_dir) / model_file

    # Load trajectories
    heldout_path = data_path / "heldout_episodes.pkl"

    if not heldout_path.exists():
        raise FileNotFoundError(
            f"Heldout episodes not found. Ensure {heldout_path} exists. "
            "Run collect_trajectories.py first."
        )

    print(f"Loading trajectories from {data_path}...")
    with open(heldout_path, "rb") as f:
        heldout_episodes = pickle.load(f)
    print(f"Loaded {len(heldout_episodes)} heldout pairwise episodes.")

    # Load trained model
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Run train_autoencoder.py first."
        )

    print(f"Loading trained model from {model_path}...")
    with open(model_path, "rb") as f:
        checkpoint = pickle.load(f)

    params = checkpoint["params"]
    config = checkpoint["config"]
    hidden_dim = config["hidden_dim"]
    latent_dim = config["latent_dim"]
    obs_dim = config["obs_dim"]
    max_seq_len = config["max_seq_len"]

    print(f"Model config: {config}")

    # Recreate the model and train_state from saved parameters
    model = create_autoencoder(obs_dim, max_seq_len, hidden_dim, latent_dim)
    # Create a dummy train_state with loaded parameters
    tx = optax.adam(1e-3)  # dummy learning rate, not used for inference
    train_state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    # Encode heldout episodes
    print("Encoding heldout trajectories...")
    heldout_latents = encode_episodes(model, train_state, heldout_episodes, max_seq_len)

    # Collect and encode trajectories for each agent-BR pair
    print("Collecting trajectories for each agent-BR pair...")
    from envs import make_env
    env = make_env(env_name, {})

    rng = jax.random.PRNGKey(42)
    all_pairs = get_agent_pair_configs(env_name)

    # Filter pairs to only include matching agent-BR combinations
    # BR names like 'br_for_ippo_mlp_0' should match agent 'ippo_mlp'
    def get_base_agent_name(br_name):
        # Remove 'br_for_' prefix and any trailing numbers/underscores
        if br_name.startswith('br_for_'):
            base = br_name[7:]  # Remove 'br_for_'
            # Remove trailing patterns like '_0', '_1', '_2_0', etc.
            import re
            base = re.sub(r'_\d+(_\d+)*$', '', base)
            return base
        return br_name

    # Group pairs by agent
    agent_to_brs = {}
    for agent_name, agent_cfg, br_name, br_cfg in all_pairs:
        base_agent = get_base_agent_name(br_name)
        if base_agent == agent_name or base_agent.replace('_s2c0', '') == agent_name:
            if agent_name not in agent_to_brs:
                agent_to_brs[agent_name] = []
            agent_to_brs[agent_name].append((br_name, br_cfg))

    print(f"Found {len(agent_to_brs)} agent groups with BRs")

    pair_latents = {}
    for agent_name, br_list in agent_to_brs.items():
        for br_name, br_cfg in br_list:
            pair_key = f"{agent_name}_vs_{br_name}"
            print(f"Collecting trajectories for {pair_key}...")

            # Load agents
            from common.agent_loader_from_config import initialize_rl_agent_from_config

            def _load_agent(agent_cfg, agent_name, env, rng):
                policy, params, init_params, _ = initialize_rl_agent_from_config(agent_cfg, agent_name, env, rng)
                if "path" in agent_cfg:
                    params = jax.tree_map(jnp.squeeze, params)
                    idx_list = agent_cfg.get("idx_list", None)
                    if idx_list is not None and len(idx_list) > 1:
                        params = jax.tree_map(lambda x: x[0], params)
                    params = jax.tree_map(lambda p, i: p.reshape(i.shape) if p.size == i.size else p, params, init_params)
                else:
                    # Non-RL heuristic has no checkpoint params; keep empty dict for consistency.
                    params = {}
                    init_params = {}
                return policy, params, init_params

            rng, rng_agent = jax.random.split(rng)
            teammate_policy, teammate_params, _ = _load_agent(agent_cfg, agent_name, env, rng_agent)
            rng, rng_br = jax.random.split(rng)
            br_policy, br_params, _ = _load_agent(br_cfg, br_name, env, rng_br)

            # Collect trajectories for this pair
            rng, pair_episodes = collect_pair_trajectories(
                rng,
                env,
                teammate_policy,
                teammate_params,
                br_policy,
                br_params,
                num_rollouts=k,
                rollout_steps=rollout_steps,
                num_envs=num_envs,
            )

            print(f"Collected {len(pair_episodes)} episodes for {pair_key}")

            # Encode episodes
            pair_latents[pair_key] = encode_episodes(model, train_state, pair_episodes, max_seq_len)

    # Combine all latents for plotting
    all_latents = {"heldout": heldout_latents}
    all_latents.update(pair_latents)

    # Plot
    print(f"Creating t-SNE visualization with {len(all_latents)} categories...")
    plot_tsne(
        all_latents,
        save_path=output_file,
    )
    print(f"Visualization saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize trajectories using trained autoencoder.")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="Directory containing trajectory data")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR, help="Directory containing trained model")
    parser.add_argument("--model_file", type=str, default=DEFAULT_MODEL_FILE, help="Trained model filename")
    parser.add_argument("--output_file", type=str, default=DEFAULT_OUTPUT_FILE, help="Output visualization filename")
    parser.add_argument("--env_name", type=str, default="lbf", help="Environment name")
    parser.add_argument("--k", type=int, default=5, help="Number of rollouts per agent pair")
    parser.add_argument("--num_envs", type=int, default=256, help="Number of parallel environments")
    parser.add_argument("--rollout_steps", type=int, default=128, help="Steps per rollout")

    args = parser.parse_args()
    main(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        model_file=args.model_file,
        output_file=args.output_file,
        env_name=args.env_name,
        k=args.k,
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
    )
