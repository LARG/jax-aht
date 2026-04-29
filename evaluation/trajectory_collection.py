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


def collect_random_trajectories(rng, env, num_rollouts, rollout_steps, num_envs=256):
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
                    # Store raw trajectory data only
                    all_episodes.append(episode)
                ep_start = ep_end

    return rng, all_episodes


def collect_ippo_selfplay_trajectories(rng, env, num_rollouts, rollout_steps, num_envs=256):
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
                    # Store raw trajectory data only
                    all_episodes.append(episode)
                ep_start = ep_end

    return rng, all_episodes


def _policy_action(policy, params, obs, done, avail_actions, hstate, rng, env_state):
    # obs: (num_envs, obs_dim), done: (num_envs,), avail_actions: (num_envs, n_actions)
    # hstate: None/or recurrent tuple, rng: single PRNGKey, env_state: batched env state
    # Returns actions: (num_envs,), new_hstate

    num_envs = obs.shape[0]
    rng_keys = jax.random.split(rng, num_envs)

    # MLP policies: no hstate, vmap directly over envs
    if hstate is None:
        actions, _ = jax.vmap(partial(policy.get_action, params))(
            obs, done, avail_actions, None, rng_keys, env_state,
        )
        return actions, None

    # S5-style recurrent policy: hstate is a jnp.ndarray or list of jnp.ndarrays.
    # A single call handles all envs internally via the (1, num_envs, ...) convention.
    if isinstance(hstate, (jnp.ndarray, list)):
        obs_exp = obs[None]    # (1, num_envs, obs_dim)
        done_exp = done[None]  # (1, num_envs)
        avail_exp = avail_actions[None]  # (1, num_envs, n_actions)
        actions_exp, new_hstate = policy.get_action(params, obs_exp, done_exp, avail_exp, hstate, rng, env_state)
        if actions_exp.ndim == 2:
            actions = actions_exp.squeeze(0)
        else:
            actions = actions_exp
        return actions, new_hstate

    # Per-env structured hstate (e.g. heuristic AgentState): vmap over the env axis so
    # each call receives single-env obs/hstate/env_state.
    # init_hstate returns a single state; tile scalar leaves to (num_envs,) so vmap
    # has a leading axis to map over on the first step (subsequent steps are already batched).
    leaves = jax.tree_leaves(hstate)
    if leaves and jnp.asarray(leaves[0]).ndim == 0:
        hstate = jax.tree_map(
            lambda x: jnp.broadcast_to(jnp.asarray(x), (num_envs,) + jnp.asarray(x).shape),
            hstate,
        )
    actions, new_hstate = jax.vmap(partial(policy.get_action, params))(
        obs, done, avail_actions, hstate, rng_keys, env_state,
    )
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
    num_envs=256,
    agent_idx=None,
    br_idx=None,
):
    """Collect trajectories from a pair of policies.
    
    Args:
        agent_idx: Index of the teammate agent (optional, for tracking)
        br_idx: Index of the best response agent (optional, for tracking)
    
    Returns:
        rng, all_episodes where each episode is raw trajectory data
    """
    all_episodes = []
    agent0 = env.agents[0]
    agent1 = env.agents[1]

    # params are already flattened in the calling function

    for _ in range(num_rollouts):
        rng, rng_reset = jax.random.split(rng)
        rng_resets = jax.random.split(rng_reset, num_envs)
        obs, state = jax.vmap(env.reset)(rng_resets)

        # Initialize hidden states
        teammate_hstate = teammate_policy.init_hstate(num_envs, {"agent_id": 0})
        br_hstate = br_policy.init_hstate(num_envs, {"agent_id": 1})

        obs_buffer = []
        done_buffer = []

        for _ in range(rollout_steps):
            obs_buffer.append(obs[agent0])

            rng, rng_t = jax.random.split(rng)
            rng, rng_b = jax.random.split(rng)

            actions = {}
            actions[agent0], teammate_hstate = _policy_action(
                teammate_policy,
                teammate_params,
                obs[agent0],
                jnp.zeros(num_envs, dtype=bool),
                state.avail_actions[agent0],
                teammate_hstate,
                rng_t,
                state,
            )
            actions[agent1], br_hstate = _policy_action(
                br_policy,
                br_params,
                obs[agent1],
                jnp.zeros(num_envs, dtype=bool),
                state.avail_actions[agent1],
                br_hstate,
                rng_b,
                state,
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
                    # Store raw trajectory data only
                    all_episodes.append(episode)
                ep_start = ep_end

    return rng, all_episodes


def _normalize_name(name):
    return name.replace("-", "_")


def _find_specific_br(agent_name, br_names):
    """Return the specific BR name for agent_name among br_names, or None."""
    norm = _normalize_name(agent_name)
    for br in br_names:
        if not br.startswith("br_for_"):
            continue
        suffix = br[len("br_for_"):]
        if suffix == norm or suffix.startswith(norm + "_"):
            return br
    return None


def _find_agent_for_br(br_name, agent_names):
    """Return the agent name that br_name is specifically for, or None."""
    if not br_name.startswith("br_for_"):
        return None
    suffix = br_name[len("br_for_"):]
    for agent in agent_names:
        norm = _normalize_name(agent)
        if suffix == norm or suffix.startswith(norm + "_"):
            return agent
    return None


def _load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _expand_multi_index_configs(agent_configs):
    """Expand configs with multiple idx_list entries into one config per index.

    For 1-D indices like [0, 1, 2] the expanded names are ``{name}_0``,
    ``{name}_1``, ``{name}_2``.  For 2-D indices like [[1, 0], [1, 1]] the
    expanded names are ``{name}_1_0``, ``{name}_1_1``, matching the convention
    used in the BR config yaml.
    """
    expanded = {}
    for name, cfg in agent_configs.items():
        idx_list = cfg.get("idx_list", None)
        if idx_list is None or "path" not in cfg or len(idx_list) <= 1:
            expanded[name] = cfg
            continue
        for idx in idx_list:
            if isinstance(idx, (list, tuple)):
                suffix = "_".join(str(i) for i in idx)
            else:
                suffix = str(idx)
            new_cfg = dict(cfg)
            new_cfg["idx_list"] = [idx]
            expanded[f"{name}_{suffix}"] = new_cfg
    return expanded


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

    agent_configs = _expand_multi_index_configs(heldout_set[env_name])
    br_configs = best_response_set[env_name]

    pairs = []
    for br_name, br_cfg in br_configs.items():
        for ag_name, ag_cfg in agent_configs.items():
            pairs.append((ag_name, ag_cfg, br_name, br_cfg))

    return pairs


def collect_heldout_pairwise_trajectories(
    rng,
    env,
    k=5,
    rollout_steps=128,
    num_envs=256,
    env_name="lbf",
    settings_path=None,
    br_path=None,
):
    pairs = get_agent_pair_configs(env_name, settings_path=settings_path, br_path=br_path)
    all_episodes = []  # Will store (trajectory, agent_pair_label) tuples

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

    # Gather ordered unique names and their configs
    names_to_cfg = {}
    for agent_name, agent_cfg, br_name, br_cfg in pairs:
        names_to_cfg.setdefault(agent_name, agent_cfg)
        names_to_cfg.setdefault(br_name, br_cfg)

    all_agent_names = list({p[0] for p in pairs})
    all_br_names = list({p[2] for p in pairs})

    # Load all unique agents/BRs, catching failures instead of crashing
    print("Initializing Agents...")
    unique_agents = {}
    failed_names = set()
    for name, cfg in names_to_cfg.items():
        rng, rng_load = jax.random.split(rng)
        try:
            unique_agents[name] = _load_agent(cfg, name, env, rng_load)
        except Exception as e:
            print(f"ERROR: Could not load '{name}': {e}")
            failed_names.add(name)

    # Propagate each failure to its specific partner (agent ↔ its specific BR)
    for name in list(failed_names):
        partner = _find_specific_br(name, all_br_names)
        if partner is None:
            partner = _find_agent_for_br(name, all_agent_names)
        if partner is not None and partner not in failed_names:
            print(f"ERROR: Also removing '{partner}' from data collection (paired with failed '{name}').")
            failed_names.add(partner)
            unique_agents.pop(partner, None)

    print("Initialized Agents...")

    # Filter to pairs where both members loaded successfully
    valid_pairs = [
        (an, ac, bn, bc) for an, ac, bn, bc in pairs
        if an not in failed_names and bn not in failed_names
    ]

    # Create mappings from agent/BR names to indices for tracking
    agent_name_to_idx = {}
    br_name_to_idx = {}
    agent_idx_counter = 0
    br_idx_counter = 0

    for agent_name, _, br_name, _ in valid_pairs:
        if agent_name not in agent_name_to_idx:
            agent_name_to_idx[agent_name] = agent_idx_counter
            agent_idx_counter += 1
        if br_name not in br_name_to_idx:
            br_name_to_idx[br_name] = br_idx_counter
            br_idx_counter += 1

    # Create a mapping from (agent_idx, br_idx) to pair label
    pair_labels = {}
    for agent_name, _, br_name, _ in valid_pairs:
        agent_idx = agent_name_to_idx[agent_name]
        br_idx = br_name_to_idx[br_name]
        pair_labels[(agent_idx, br_idx)] = f"{agent_name}_{br_name}"

    for agent_name, agent_cfg, br_name, br_cfg in valid_pairs:
        print(f"Collecting {agent_name} (idx={agent_name_to_idx[agent_name]}), {br_name} (idx={br_name_to_idx[br_name]})...")
        teammate_policy, teammate_params, _ = unique_agents[agent_name]
        br_policy, br_params, _ = unique_agents[br_name]
        agent_idx = agent_name_to_idx[agent_name]
        br_idx = br_name_to_idx[br_name]
        pair_label = pair_labels[(agent_idx, br_idx)]

        for i in range(k):
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
                agent_idx=agent_idx,
                br_idx=br_idx,
            )
            # Store trajectories with their pair labels
            for episode in episodes:
                all_episodes.append((episode, pair_label))
        print(f"Collected {agent_name}, {br_name}")

    return rng, all_episodes, pair_labels
