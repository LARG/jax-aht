'''
Episode runner for N-agent AHT evaluation.

Analogous to common/run_episodes.py but supports N agents organized into
two parameter-sharing groups:
  - ego agents   (ego_indices)   -> ego_policy + ego_param
  - teammate agents (teammate_indices) -> teammate_policy + teammate_param

Each agent's observation is augmented with a one-hot agent-ID vector before
being passed to the shared policy, so a single policy can condition on
identity.  This is used both for mid-training evaluation inside ROTATE and for
heldout evaluation of trained ego agents.

Output format is identical to LogWrapper / run_episodes so existing logging
helpers remain compatible.
'''
from functools import lru_cache, partial
from typing import List

import jax
import jax.numpy as jnp

from common.n_agent_utils import augment_obs_with_id


# ---------------------------------------------------------------------------
# Single-episode runner
# ---------------------------------------------------------------------------

def run_n_agent_single_episode(
    rng,
    env,
    ego_policy,
    ego_param,
    teammate_policy,
    teammate_param,
    ego_indices: List[int],
    teammate_indices: List[int],
    max_episode_steps: int,
    ego_test_mode: bool = False,
    teammate_test_mode: bool = False,
):
    '''Run one episode with two parameter-sharing agent groups.

    Args:
        rng:                 JAX PRNGKey
        env:                 LogWrapper-wrapped environment
        ego_policy:          AgentPolicy for ego agents (shared)
        ego_param:           parameters for ego_policy
        teammate_policy:     AgentPolicy for teammate agents (shared)
        teammate_param:      parameters for teammate_policy
        ego_indices:         agent indices controlled by ego_policy
        teammate_indices:    agent indices controlled by teammate_policy
        max_episode_steps:   maximum number of steps before forced termination
        ego_test_mode:       passed through to ego_policy.get_action
        teammate_test_mode:  passed through to teammate_policy.get_action

    Returns:
        The final LogWrapper info dict (same structure as run_single_episode).
    '''
    num_agents = env.num_agents
    n_ego = len(ego_indices)
    n_teammate = len(teammate_indices)

    # Reset
    rng, reset_rng = jax.random.split(rng)
    init_obs, init_env_state = env.reset(reset_rng)
    init_done = {k: jnp.zeros((1,), dtype=bool) for k in env.agents + ['__all__']}

    # Initialise hidden states for every agent slot
    # ego: one hstate per ego slot, batch_size=1 each
    ego_hstates = [
        ego_policy.init_hstate(1, aux_info={'agent_id': agent_idx})
        for agent_idx in ego_indices
    ]
    # teammate: one hstate per teammate slot, batch_size=1 each
    teammate_hstates = [
        teammate_policy.init_hstate(1, aux_info={'agent_id': agent_idx})
        for agent_idx in teammate_indices
    ]

    # Helper: collect actions for all agents given current obs / done
    def _get_all_actions(rng, obs, done, env_state, ego_hstates, teammate_hstates):
        actions = {}
        new_ego_hstates = [None] * n_ego
        new_teammate_hstates = [None] * n_teammate

        # Fetch available actions for all agents from the env state
        avail_actions_dict = env.get_avail_actions(env_state.env_state)

        for slot, agent_idx in enumerate(ego_indices):
            rng, act_rng = jax.random.split(rng)
            agent_key = f'agent_{agent_idx}'
            aug_obs = augment_obs_with_id(obs[agent_key], agent_idx, num_agents)
            avail = avail_actions_dict[agent_key].astype(jnp.float32).reshape(1, -1)
            # Truncate obs to expected_obs_dim for compatibility with policies
            # that do not use agent_id as a feature. A policy with obs_dim=None
            # (e.g. heuristic heldout agents) expects the raw, un-augmented obs,
            # so strip the one-hot agent-ID we appended above.
            expected_obs_dim = getattr(ego_policy, 'obs_dim', None)
            if expected_obs_dim is None:
                expected_obs_dim = aug_obs.shape[-1] - num_agents
            policy_obs = aug_obs[..., :expected_obs_dim]
            
            act, new_h = ego_policy.get_action(
                params=ego_param,
                obs=policy_obs.reshape(1, 1, -1),
                done=done[agent_key].reshape(1, 1),
                avail_actions=avail,
                hstate=ego_hstates[slot],
                rng=act_rng,
                env_state=env_state,
                test_mode=ego_test_mode,
            )
            actions[agent_key] = act.squeeze()
            new_ego_hstates[slot] = new_h

        for slot, agent_idx in enumerate(teammate_indices):
            rng, act_rng = jax.random.split(rng)
            agent_key = f'agent_{agent_idx}'
            aug_obs = augment_obs_with_id(obs[agent_key], agent_idx, num_agents)
            avail = avail_actions_dict[agent_key].astype(jnp.float32).reshape(1, -1)
            # Truncate obs to expected_obs_dim for compatibility with policies
            # that do not use agent_id as a feature. A policy with obs_dim=None
            # (e.g. heuristic heldout agents) expects the raw, un-augmented obs,
            # so strip the one-hot agent-ID we appended above.
            expected_obs_dim = getattr(teammate_policy, 'obs_dim', None)
            if expected_obs_dim is None:
                expected_obs_dim = aug_obs.shape[-1] - num_agents
            policy_obs = aug_obs[..., :expected_obs_dim]
            
            act, new_h = teammate_policy.get_action(
                params=teammate_param,
                obs=policy_obs.reshape(1, 1, -1),
                done=done[agent_key].reshape(1, 1),
                avail_actions=avail,
                hstate=teammate_hstates[slot],
                rng=act_rng,
                env_state=env_state,
                test_mode=teammate_test_mode,
            )
            actions[agent_key] = act.squeeze()
            new_teammate_hstates[slot] = new_h

        return rng, actions, new_ego_hstates, new_teammate_hstates

    # Bootstrap one step to get dummy info for scan
    avail_actions = env.get_avail_actions(init_env_state.env_state)
    rng, act0_rng, step_rng = jax.random.split(rng, 3)

    rng, boot_actions, ego_hstates, teammate_hstates = _get_all_actions(
        rng, init_obs, init_done, init_env_state, ego_hstates, teammate_hstates
    )
    obs, env_state, _, done, dummy_info = env.step(step_rng, init_env_state, boot_actions)

    # Scan
    ep_ts = 1
    # Pack hstates as tuples for JAX pytree compatibility inside scan
    init_carry = (ep_ts, env_state, obs, rng, done,
                  tuple(ego_hstates), tuple(teammate_hstates), dummy_info)

    def scan_step(carry, _):
        def take_step(carry_step):
            ep_ts, env_state, obs, rng, done, ego_hstates, teammate_hstates, last_info = carry_step
            rng, step_rng = jax.random.split(rng)
            rng, actions, new_ego_hstates, new_teammate_hstates = _get_all_actions(
                rng, obs, done, env_state, list(ego_hstates), list(teammate_hstates)
            )
            obs_next, env_state_next, _, done_next, info_next = env.step(
                step_rng, env_state, actions
            )
            return (ep_ts + 1, env_state_next, obs_next, rng, done_next,
                    tuple(new_ego_hstates), tuple(new_teammate_hstates), info_next)

        ep_ts, env_state, obs, rng, done, ego_hstates, teammate_hstates, last_info = carry
        new_carry = jax.lax.cond(
            done['__all__'],
            lambda c: c,    # episode over: hold state
            take_step,
            operand=carry
        )
        return new_carry, None

    final_carry, _ = jax.lax.scan(scan_step, init_carry, None, length=max_episode_steps)
    return final_carry[-1]  # final info dict


@lru_cache(maxsize=128)
def _compiled_run_n_agent_episodes(env, ego_policy, teammate_policy, ego_indices,
                                   teammate_indices, max_episode_steps,
                                   ego_test_mode, teammate_test_mode):
    '''See the equivalent helper in common/run_episodes.py.'''
    @jax.jit
    def _run(ep_rngs, ego_param, teammate_param):
        return jax.vmap(lambda ep_rng: run_n_agent_single_episode(
            ep_rng, env,
            ego_policy, ego_param,
            teammate_policy, teammate_param,
            ego_indices, teammate_indices,
            max_episode_steps,
            ego_test_mode, teammate_test_mode,
        ))(ep_rngs)
    return _run

def run_n_agent_episodes(
    rng,
    env,
    ego_policy,
    ego_param,
    teammate_policy,
    teammate_param,
    ego_indices: List[int],
    teammate_indices: List[int],
    max_episode_steps: int,
    num_eps: int,
    ego_test_mode: bool = False,
    teammate_test_mode: bool = False,
):
    '''Run num_eps parallel episodes via vmap.

    Args:
        (same as run_n_agent_single_episode, plus num_eps)

    Returns:
        Dict where each leaf has shape (num_eps, ...).
    '''
    rngs = jax.random.split(rng, num_eps)

    vmap_episode = _compiled_run_n_agent_episodes(
        env, ego_policy, teammate_policy,
        tuple(ego_indices), tuple(teammate_indices),
        max_episode_steps, ego_test_mode, teammate_test_mode)
    return vmap_episode(rngs, ego_param, teammate_param)
