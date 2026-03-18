from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from common.agent_loader_from_config import initialize_rl_agent_from_config

IPPO_CONFIG = {
    "path": "eval_teammates/lbf/ippo/2025-04-21_23-41-17/saved_train_run",
    "actor_type": "mlp",
    "ckpt_key": "final_params",
    "idx_list": [0],
    "test_mode": False,
}


def collect_random_trajectories(rng, env, num_rollouts, rollout_steps, num_envs):
    all_episodes = []

    for _ in range(num_rollouts):
        rng, rng_reset = jax.random.split(rng)
        rng_resets = jax.random.split(rng_reset, num_envs)
        obs, state = jax.vmap(env.reset)(rng_resets)

        obs_buffer = []
        done_buffer = []
        for _ in range(rollout_steps):
            obs_buffer.append(obs["agent_0"])
            actions = {}
            for agent in env.agents:
                rng, rng_a = jax.random.split(rng)
                avail = state.avail_actions[agent]
                rng_keys = jax.random.split(rng_a, num_envs)
                logits = jnp.where(avail, 0.0, -1e10)
                actions[agent] = jax.vmap(lambda key, lg: jax.random.categorical(key, lg))(rng_keys, logits)

            rng, rng_step = jax.random.split(rng)
            rng_steps = jax.random.split(rng_step, num_envs)
            obs, state, rewards, dones, infos = jax.vmap(env.step)(rng_steps, state, actions)
            done_buffer.append(dones["__all__"])

        obs_buffer = jnp.stack(obs_buffer)
        done_buffer = jnp.stack(done_buffer)
        obs_np = np.array(obs_buffer)
        done_np = np.array(done_buffer)

        for env_idx in range(num_envs):
            env_obs = obs_np[:, env_idx, :]
            env_done = done_np[:, env_idx]
            done_indices = np.where(env_done)[0]
            ep_start = 0
            for done_idx in done_indices:
                ep_end = done_idx + 1
                episode = env_obs[ep_start:ep_end]
                if len(episode) > 1:
                    all_episodes.append(episode)
                ep_start = ep_end

    return rng, all_episodes


def collect_ippo_selfplay_trajectories(rng, env, num_rollouts, rollout_steps, num_envs):
    rng, load_rng = jax.random.split(rng)
    policy, agent_params, _, _ = initialize_rl_agent_from_config(IPPO_CONFIG, "ippo_mlp", env, load_rng)
    agent_params = jax.tree_map(lambda x: x[0], agent_params)

    all_episodes = []
    for _ in range(num_rollouts):
        rng, rng_reset = jax.random.split(rng)
        rng_resets = jax.random.split(rng_reset, num_envs)
        obs, state = jax.vmap(env.reset)(rng_resets)

        obs_buffer = []
        done_buffer = []
        for _ in range(rollout_steps):
            obs_buffer.append(obs["agent_0"])
            actions = {}
            for agent in env.agents:
                rng, rng_a = jax.random.split(rng)
                avail = state.avail_actions[agent]
                rng_keys = jax.random.split(rng_a, num_envs)
                action, _ = jax.vmap(partial(policy.get_action, agent_params))(obs[agent], jnp.zeros(num_envs, dtype=bool), avail, None, rng_keys)
                actions[agent] = action

            rng, rng_step = jax.random.split(rng)
            rng_steps = jax.random.split(rng_step, num_envs)
            obs, state, rewards, dones, infos = jax.vmap(env.step)(rng_steps, state, actions)
            done_buffer.append(dones["__all__"])

        obs_buffer = jnp.stack(obs_buffer)
        done_buffer = jnp.stack(done_buffer)
        obs_np = np.array(obs_buffer)
        done_np = np.array(done_buffer)

        for env_idx in range(num_envs):
            env_obs = obs_np[:, env_idx, :]
            env_done = done_np[:, env_idx]
            done_indices = np.where(env_done)[0]
            ep_start = 0
            for done_idx in done_indices:
                ep_end = done_idx + 1
                episode = env_obs[ep_start:ep_end]
                if len(episode) > 1:
                    all_episodes.append(episode)
                ep_start = ep_end

    return rng, all_episodes
