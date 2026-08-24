'''Utility functions for N-agent parameter sharing.

Handles the split between ego agents (controlled by the ego policy) and
teammate agents (controlled by the confederate / BR policy). Each agent's
observation is augmented with a one-hot vector encoding its agent index so
that a single shared policy can condition on agent identity.
'''
from typing import List, Tuple

import jax
import jax.numpy as jnp


def get_ego_teammate_indices(config: dict, env) -> Tuple[List[int], List[int]]:
    '''Return (ego_indices, teammate_indices) from config.

    If EGO_INDICES is not set in the config, defaults to ego=[0] and
    teammates=[1, ..., N-1], i.e. a single ego with all remaining agents
    as teammates (the classic 2-agent AHT assumption generalised to N).

        ego_indices:     list of int agent indices controlled by the ego policy.
        teammate_indices: list of int agent indices controlled by the
                          confederate / BR policy.
    '''
    num_agents = env.num_agents
    ego_indices = list(config.get('EGO_INDICES') or [0])
    teammate_indices = [i for i in range(num_agents) if i not in ego_indices]
    assert len(ego_indices) > 0, 'EGO_INDICES must contain at least one agent.'
    assert len(teammate_indices) > 0, 'There must be at least one teammate agent.'
    assert set(ego_indices) | set(teammate_indices) == set(range(num_agents)), \
        'EGO_INDICES and teammate_indices must together cover all agents.'
    return ego_indices, teammate_indices


def augment_obs_with_id(obs: jnp.ndarray, agent_idx: int, num_agents: int, augment_obs: bool = True) -> jnp.ndarray:
    '''Append a one-hot agent-ID vector to a single agent observation.

    Args:
        obs:         observation array of shape (..., obs_dim)
        agent_idx:   integer agent index (0-based); must be a *Python* int so
                     the one-hot is materialised at trace time.
        num_agents:  total number of agents (one-hot size)
        augment_obs: bool. If False, returns the observation un-augmented.

    Returns:
        Augmented obs of shape (..., obs_dim + num_agents) or original obs.
    '''
    if not augment_obs:
        return obs
    one_hot = jnp.zeros(num_agents).at[agent_idx].set(1.0)
    # Expand dims with a single reshape instead of a Python loop
    one_hot = jnp.broadcast_to(
        one_hot.reshape((1,) * (obs.ndim - 1) + (num_agents,)),
        obs.shape[:-1] + (num_agents,)
    )
    return jnp.concatenate([obs, one_hot], axis=-1)


def augment_all_obs(obs_dict: dict,
                    ego_indices: List[int],
                    teammate_indices: List[int],
                    num_agents: int,
                    num_envs: int,
                    augment_obs: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
    '''Build augmented, stacked obs arrays for ego and teammate role groups.

    For each role group the observations from all agents in the group are
    stacked along the first (batch) axis, giving shape
    ``(n_role * num_envs, obs_dim + num_agents)``.

    The list comprehensions below are trace-time unrolled because
    ego_indices / teammate_indices are static Python lists.

    Args:
        obs_dict:         dict mapping "agent_i" -> obs of shape (num_envs, obs_dim)
        ego_indices:      agent indices for the ego role
        teammate_indices: agent indices for the teammate role
        num_agents:       total agent count (determines one-hot size)
        num_envs:         number of parallel environments
        augment_obs:      bool. If False, does not concatenate the one-hot ID.

    Returns:
        ego_obs:      shape (n_ego * num_envs, obs_dim + num_agents)
        teammate_obs: shape (n_teammate * num_envs, obs_dim + num_agents)
    '''
    # jnp.stack -> (n_role, num_envs, aug_obs_dim), reshape -> (n_role*num_envs, aug_obs_dim)
    ego_obs = jnp.stack(
        [augment_obs_with_id(obs_dict[f'agent_{i}'], i, num_agents, augment_obs) for i in ego_indices],
        axis=0
    ).reshape(len(ego_indices) * num_envs, -1)

    teammate_obs = jnp.stack(
        [augment_obs_with_id(obs_dict[f'agent_{i}'], i, num_agents, augment_obs) for i in teammate_indices],
        axis=0
    ).reshape(len(teammate_indices) * num_envs, -1)

    return ego_obs, teammate_obs


def augment_done(done_dict: dict,
                 indices: List[int],
                 num_envs: int) -> jnp.ndarray:
    '''Stack done flags for a set of agent indices.

    Args:
        done_dict:  dict mapping "agent_i" -> done of shape (num_envs,)
        indices:    agent indices to stack (static Python list)
        num_envs:   number of parallel environments

    Returns:
        stacked_done: shape (n_agents * num_envs,)
    '''
    # jnp.stack -> (n, num_envs), reshape -> (n*num_envs,)
    return jnp.stack(
        [done_dict[f'agent_{i}'] for i in indices], axis=0
    ).reshape(len(indices) * num_envs)


def split_actions(ego_acts: jnp.ndarray,
                  teammate_acts: jnp.ndarray,
                  ego_indices: List[int],
                  teammate_indices: List[int],
                  num_envs: int) -> dict:
    '''Reconstruct the {agent_i: actions} dict from stacked role-group actions.

    Args:
        ego_acts:         shape (n_ego * num_envs,)  or (n_ego, num_envs)
        teammate_acts:    shape (n_teammate * num_envs,) or (n_teammate, num_envs)
        ego_indices:      agent indices for the ego role
        teammate_indices: agent indices for the teammate role
        num_envs:         number of parallel environments

    Returns:
        env_act: dict mapping "agent_i" -> actions of shape (num_envs,)
    '''
    n_ego = len(ego_indices)
    n_teammate = len(teammate_indices)

    # Reshape to (n_role, num_envs) if flat
    ego_acts = ego_acts.reshape(n_ego, num_envs)
    teammate_acts = teammate_acts.reshape(n_teammate, num_envs)

    env_act = {}
    for slot, agent_idx in enumerate(ego_indices):
        env_act[f'agent_{agent_idx}'] = ego_acts[slot]
    for slot, agent_idx in enumerate(teammate_indices):
        env_act[f'agent_{agent_idx}'] = teammate_acts[slot]
    return env_act
