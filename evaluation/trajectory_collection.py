from functools import partial
import yaml

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
    policy, agent_params, init_params, _ = initialize_rl_agent_from_config(IPPO_CONFIG, "ippo_mlp", env, load_rng)
    agent_params = jax.tree_map(jnp.squeeze, agent_params)
    agent_params = jax.tree_map(lambda p, i: p.reshape(i.shape) if p.size == i.size else p, agent_params, init_params)
    # agent_params now match init_params shape

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


def _policy_action(policy, params, obs, done, avail_actions, hstate, rng):
    # obs: (num_envs, obs_dim), done: (num_envs,), avail_actions: (num_envs, n_actions)
    # hstate: (1, num_envs, hidden), rng: single PRNGKey
    # Returns actions: (num_envs,), new_hstate: (1, num_envs, hidden)
    obs_exp = obs[None]  # (1, num_envs, obs_dim)
    done_exp = done[None]  # (1, num_envs)
    avail_exp = avail_actions[None]  # (1, num_envs, n_actions)

    # For S5-based policy, get_action expects sequence length first (1), batch dim next (num_envs).
    # Use a single key for the whole batch (distrax can sample a batch with one key).
    actions_exp, new_hstate = policy.get_action(params, obs_exp, done_exp, avail_exp, hstate, rng)
    actions = actions_exp.squeeze(0)  # (num_envs,)
    return actions, new_hstate


def collect_pair_trajectories(
    rng,
    env,
    teammate_policy,
    teammate_params,
    br_policy,
    br_params,
    num_rollouts=5,
    rollout_steps=128,
    num_envs=64,
):
    all_episodes = []
    agent0 = env.agents[0]
    agent1 = env.agents[1]

    # params are already flattened in the calling function

    for _ in range(num_rollouts):
        rng, rng_reset = jax.random.split(rng)
        rng_resets = jax.random.split(rng_reset, num_envs)
        obs, state = jax.vmap(env.reset)(rng_resets)

        # Initialize hidden states
        try:
            teammate_hstate = teammate_policy.init_hstate(num_envs)
        except AttributeError:
            try:
                teammate_hstate = teammate_policy.get_initial_hstate(num_envs)
            except AttributeError:
                teammate_hstate = None
        try:
            br_hstate = br_policy.init_hstate(num_envs)
        except AttributeError:
            try:
                br_hstate = br_policy.get_initial_hstate(num_envs)
            except AttributeError:
                br_hstate = None

        obs_buffer = []
        done_buffer = []

        for _ in range(rollout_steps):
            obs_buffer.append(obs[agent0])

            # rng, rng_t, rng_b = jax.random.split(rng, 3)
            # rng_t_keys = jax.random.split(rng_t, num_envs)
            # rng_b_keys = jax.random.split(rng_b, num_envs)

            actions = {}
            actions[agent0], teammate_hstate = _policy_action(
                teammate_policy,
                teammate_params,
                obs[agent0],
                jnp.zeros(num_envs, dtype=bool),
                state.avail_actions[agent0],
                teammate_hstate,
                rng,
            )
            actions[agent1], br_hstate = _policy_action(
                br_policy,
                br_params,
                obs[agent1],
                jnp.zeros(num_envs, dtype=bool),
                state.avail_actions[agent1],
                br_hstate,
                rng,
            )

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


def _load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_agent_pair_configs(env_name="lbf", settings_path=None, br_path=None):
    if settings_path is None:
        settings_path = "evaluation/configs/global_heldout_settings.yaml"
    if br_path is None:
        br_path = "evaluation/configs/global_heldout_br.yaml"

    settings = _load_yaml(settings_path)
    br = _load_yaml(br_path)

    heldout_set = settings.get("heldout_set", {})
    best_response_set = br.get("best_response_set", {})

    if env_name not in heldout_set:
        raise ValueError(f"Env {env_name} not found in heldout settings")
    if env_name not in best_response_set:
        raise ValueError(f"Env {env_name} not found in best response set")

    agent_configs = heldout_set[env_name]
    br_configs = best_response_set[env_name]

    pairs = []
    for ag_name, ag_cfg in agent_configs.items():
        for br_name, br_cfg in br_configs.items():
            pairs.append((ag_name, ag_cfg, br_name, br_cfg))

    return pairs


def collect_heldout_pairwise_trajectories(
    rng,
    env,
    k=5,
    rollout_steps=128,
    num_envs=64,
    env_name="lbf",
    settings_path=None,
    br_path=None,
):
    pairs = get_agent_pair_configs(env_name, settings_path=settings_path, br_path=br_path)
    all_episodes = []

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

    for agent_name, agent_cfg, br_name, br_cfg in pairs:
        print(f"Collecting {agent_name}, {br_name}...")
        for i in range(k):
            rng, rng_load_agent = jax.random.split(rng)
            teammate_policy, teammate_params, teammate_init_params = _load_agent(agent_cfg, agent_name, env, rng_load_agent)

            rng, rng_load_br = jax.random.split(rng)
            br_policy, br_params, br_init_params = _load_agent(br_cfg, br_name, env, rng_load_br)

            rng, episodes = collect_pair_trajectories(
                rng,
                env,
                teammate_policy,
                teammate_params,
                br_policy,
                br_params,
                num_rollouts=1,
                rollout_steps=rollout_steps,
                num_envs=num_envs,
            )
            all_episodes.extend(episodes)
        print(f"Collected {agent_name}, {br_name}")

    return rng, all_episodes
