'''
This file contains the code for running evaluation episodes with an ego agent and a partner agent.
Does not currently support actors that require aux_obs.
'''
import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple

from marl.ppo_utils import batchify, unbatchify

class Transition(NamedTuple):
    worst_case_reward: jnp.ndarray
    optimal_reward: jnp.ndarray

def _get_vf_restricted_avail_actions(obs, done, avail_actions, hstate, vf_params, vf_policy, epsilon_optimal):
    """
    Get restricted available actions based on value function Q-values.
    Only allows actions within epsilon of the maximum Q-value.

    Args:
        obs: observations (shape: num_envs, obs_dim)
        done: done flags (shape: num_envs,)
        avail_actions: available actions mask (shape: num_envs, num_actions)
        hstate: hidden state for the value function policy (shape depends on policy)
        vf_params: value function parameters
        vf_policy: value function policy
        epsilon_optimal: threshold for considering actions as near-optimal

    Returns:
        restricted_avail_actions: mask with only near-optimal actions (shape: num_envs, num_actions)
        next_hstate: next hidden state from the value function policy (shape depends on policy)
    """
    # Get Q-values from value function
    _, q_values, _, next_hstate = vf_policy.get_action_value_policy(
        vf_params, obs, done, avail_actions,
        hstate, jax.random.PRNGKey(0) # Use dummy since we only need the qvals
    ) # (num_envs, num_actions)

    # Mask Q-values with available actions (set unavailable to -inf)
    q_values_masked = jnp.where(avail_actions, q_values, -jnp.inf)

    # Find max Q-value for each environment
    max_q = jnp.max(q_values_masked, axis=-1, keepdims=True)  # (num_envs, 1)

    # Create mask for actions within epsilon of max
    optimal_action_mask = (q_values_masked >= (max_q - epsilon_optimal)).astype(jnp.float32)

    # Combine with original available actions
    restricted_avail_actions = optimal_action_mask * avail_actions

    # remove extra batch dim
    return restricted_avail_actions.squeeze(axis=0), next_hstate

def run_single_episode(rng, env, optimal_env, agent_idx, agent_params, agent_policy,
                       vf_params, vf_policies,
                       max_episode_steps, epsilon_optimal, use_full_obs=True, render=False, agent_test_mode=False):
    # Reset the env.
    rng, reset_rng = jax.random.split(rng)
    (init_obs, init_obs_full), init_env_state = env.reset(reset_rng)
    init_done = {k: jnp.zeros((1), dtype=bool) for k in env.agents + ["__all__"]}

    (init_opt_obs, init_opt_obs_full), init_opt_env_state = optimal_env.reset(reset_rng)
    init_opt_done = {k: jnp.zeros((1), dtype=bool) for k in optimal_env.agents + ["__all__"]}

    init_reward = {k: jnp.zeros((1)) for i, k in enumerate(env.agents)}

    num_agents = env.num_agents

    # Initialize hidden states. Agent id is passed as part of the hstate initialization to support heuristic agents.
    joint_init_hstate = agent_policy.init_hstate(1 * num_agents, aux_info={"agent_id": agent_idx})
    vf_init_hstates = [vf_policies[i].init_hstate(1, aux_info={"agent_id": i}) for i in range(num_agents)]

    # Create functions to get agent-specific data based on agent_idx
    # This avoids indexing with traced values
    def get_agent_data(data_dict, idx):
        """Select data for agent based on index using switch for any number of agents"""
        branches = [lambda d=data_dict, k=agent_key: d[k] for agent_key in env.agents]
        return jax.lax.switch(idx, branches)

    def set_agent_data(data_dict, idx, value):
        """Set data for agent based on index using switch for any number of agents"""
        branches = [lambda d=data_dict, k=agent_key, v=value: {**d, k: v} for agent_key in env.agents]
        return jax.lax.switch(idx, branches)

    # Get available actions for the agent from environment state
    avail_actions = env.get_avail_actions(init_env_state.env_state)
    avail_actions = jax.lax.stop_gradient(avail_actions)
    init_avail_actions_per_agent = [get_agent_data(avail_actions, i).astype(jnp.float32) for i in range(num_agents)]

    init_obs_per_agent = [get_agent_data(init_obs, i).reshape(1, 1, -1) for i in range(num_agents)]
    init_done_per_agent = [get_agent_data(init_done, i).reshape(1, 1) for i in range(num_agents)]

    opt_avail_actions = env.get_avail_actions(init_opt_env_state.env_state)
    opt_avail_actions = jax.lax.stop_gradient(opt_avail_actions)
    init_opt_avail_per_agent = [get_agent_data(opt_avail_actions, i).astype(jnp.float32) for i in range(num_agents)]

    init_opt_obs_per_agent = [get_agent_data(init_opt_obs, i).reshape(1, 1, -1) for i in range(num_agents)]
    init_opt_obs_full_per_agent = [get_agent_data(init_opt_obs_full, i).reshape(1, 1, -1) for i in range(num_agents)]
    init_opt_done_per_agent = [get_agent_data(init_opt_done, i).reshape(1, 1) for i in range(num_agents)]

    # Restrict available actions based on value function
    vf_restricted_avail_actions = []
    vf_hstates = []
    for i in range(num_agents):
        vf_restricted_i, vf_hstate_i = _get_vf_restricted_avail_actions(
            obs=init_obs_per_agent[i],
            done=init_done_per_agent[i],
            avail_actions=init_avail_actions_per_agent[i],
            hstate=vf_init_hstates[i],
            vf_params=vf_params[i],
            vf_policy=vf_policies[i],
            epsilon_optimal=epsilon_optimal
        )
        vf_restricted_avail_actions.append(vf_restricted_i)
        vf_hstates.append(vf_hstate_i)

    # Do one step to get a dummy info structure
    rng, *act_rngs, step_rng = jax.random.split(rng, num_agents + 2)

    ###########################################
    #                Worst Case               #
    ###########################################

    # Centralized joint observation - concatenate both agents' full observations
    joint_init_obs = batchify(init_obs, env.agents, env.num_agents * 1) # shape (num_agents * num_envs, obs_dim)
    joint_init_full_obs = batchify(init_obs_full, env.agents, env.num_agents * 1) # shape (num_agents * num_envs)
    joint_init_done = batchify(init_done, env.agents, env.num_agents * 1).squeeze(axis=-1)

    # Stack restricted available actions for all agents
    joint_vf_restricted_avail_actions = jnp.stack(vf_restricted_avail_actions, axis=0).reshape(num_agents * 1, -1)

    # Get joint agent action
    joint_act, joint_hstate = agent_policy.get_action(
        params=agent_params,
        obs=jnp.expand_dims(joint_init_full_obs, axis=0) if use_full_obs else jnp.expand_dims(joint_init_obs, axis=0),
        done=jnp.expand_dims(joint_init_done, axis=0),
        avail_actions=joint_vf_restricted_avail_actions,
        hstate=joint_init_hstate,
        rng=act_rngs[0],
        aux_obs=None,
        env_state=init_env_state,
        test_mode=agent_test_mode
    )
    joint_act = joint_act.squeeze(axis=0)

    env_act = unbatchify(joint_act, env.agents, 1, env.num_agents)

    env_act_onehot = {k: jax.nn.one_hot(env_act[env.agents[i]], env.action_space(env.agents[i]).n) for i, k in enumerate(env.agents)}
    obs, env_state, reward, done, info = env.step(step_rng, init_env_state, env_act)

    ###########################################
    #               Optimal Case              #
    ###########################################

    # Get optimal action based on agent_idx using JAX switch
    def make_init_optimal_branch(i):
        def branch():
            act, hstate_next = vf_policies[i].get_action(
                params=vf_params[i],
                obs=init_opt_obs_per_agent[i],
                done=init_opt_done_per_agent[i],
                avail_actions=init_opt_avail_per_agent[i],
                hstate=vf_init_hstates[i],
                rng=act_rngs[i],
                aux_obs=None,
                env_state=init_opt_env_state,
                test_mode=True
            )
            # updated = tuple(hstate_next if j == i else vf_init_hstates[j] for j in range(num_agents))
            # return act.squeeze(axis=0), updated
            return act.squeeze(axis=0), hstate_next
        return branch

    optimal_act, optimal_vf_hstates = jax.lax.switch(agent_idx, [make_init_optimal_branch(i) for i in range(num_agents)])

    optimal_env_act = {k: jnp.zeros_like(optimal_act) for i, k in enumerate(optimal_env.agents)}
    optimal_env_act = set_agent_data(optimal_env_act, agent_idx, optimal_act)
    optimal_env_act_onehot = {k: jax.nn.one_hot(optimal_env_act[optimal_env.agents[i]], optimal_env.action_space(optimal_env.agents[i]).n) for i, k in enumerate(optimal_env.agents)}
    (optimal_obs, optimal_obs_full), optimal_env_state, optimal_reward, optimal_done, optimal_info = optimal_env.step(step_rng, init_opt_env_state, optimal_env_act)

    # We'll use separate scans to iterate steps for worst case and optimal case
    # since they may have different episode lengths

    ###########################################
    #          Worst Case Scan                #
    ###########################################
    worst_ep_ts = 1
    worst_init_carry = (worst_ep_ts, env_state, obs, rng, done, reward, env_act_onehot, joint_hstate, vf_hstates, info)
    def worst_scan_step(carry, _):
        def take_worst_step(carry_step):
            ep_ts, env_state, (obs, obs_full), rng, done, reward, act_onehot, joint_hstate, vf_hstates, last_info = carry_step

            # Get available actions for the agent from environment state
            avail_actions = env.get_avail_actions(env_state.env_state)
            avail_actions = jax.lax.stop_gradient(avail_actions)

            # Restrict available actions based on value function
            vf_restricted_avail_actions = []
            vf_hstates_next = []
            for i in range(num_agents):
                vf_restricted_i, vf_hstate_next_i = _get_vf_restricted_avail_actions(
                    obs=get_agent_data(obs, i).reshape(1, 1, -1),
                    done=get_agent_data(done, i).reshape(1, 1),
                    avail_actions=get_agent_data(avail_actions, i).astype(jnp.float32),
                    hstate=vf_hstates[i],
                    vf_params=vf_params[i],
                    vf_policy=vf_policies[i],
                    epsilon_optimal=epsilon_optimal
                )
                vf_restricted_avail_actions.append(vf_restricted_i)
                vf_hstates_next.append(vf_hstate_next_i)

            rng, *act_rngs, step_rng = jax.random.split(rng, num_agents + 2)

            # Centralized joint observation - concatenate both agents' full observations
            joint_obs = batchify(obs, env.agents, num_agents * 1) # shape (num_agents * num_envs, obs_dim)
            joint_full_obs = batchify(obs_full, env.agents, num_agents * 1) # shape (num_agents * num_envs)
            joint_done = batchify(done, env.agents, num_agents * 1).squeeze(axis=-1)

            # Stack restricted available actions for all agents
            joint_vf_restricted_avail_actions = jnp.stack(vf_restricted_avail_actions, axis=0).reshape(num_agents * 1, -1)

            # Get joint action
            joint_act, joint_hstate_next = agent_policy.get_action(
                params=agent_params,
                obs=jnp.expand_dims(joint_full_obs, axis=0) if use_full_obs else jnp.expand_dims(joint_obs, axis=0),
                done=jnp.expand_dims(joint_done, axis=0),
                avail_actions=joint_vf_restricted_avail_actions,
                hstate=joint_hstate,
                rng=act_rngs[0],
                aux_obs=None,
                env_state=init_env_state,
                test_mode=agent_test_mode
            )
            joint_act = joint_act.squeeze(axis=0)

            env_act = unbatchify(joint_act, env.agents, 1, env.num_agents)

            env_act_onehot = {k: jax.nn.one_hot(env_act[env.agents[i]], env.action_space(env.agents[i]).n) for i, k in enumerate(env.agents)}
            obs_next, env_state_next, reward, done_next, info_next = env.step(step_rng, env_state, env_act)

            return (ep_ts + 1, env_state_next, obs_next, rng, done_next, reward, env_act_onehot, joint_hstate_next, vf_hstates_next, info_next)

        ep_ts, env_state, obs, rng, done, reward, act_onehot, hstate, vf_hstates, last_info = carry
        output = carry
        new_carry = jax.lax.cond(
            done["__all__"],
            lambda curr_carry: curr_carry,  # True fn
            take_worst_step,  # False fn
            operand=carry
        )
        return new_carry, output

    worst_final_carry, worst_stacked_carry = jax.lax.scan(
        worst_scan_step, worst_init_carry, None, length=max_episode_steps)
    worst_case_info = worst_final_carry[-1]

    ###########################################
    #          Optimal Case Scan              #
    ###########################################
    optimal_ep_ts = 1
    optimal_init_carry = (optimal_ep_ts, optimal_env_state, optimal_obs, rng, optimal_done, optimal_reward, optimal_env_act_onehot, optimal_vf_hstates, optimal_info)
    def optimal_scan_step(carry, _):
        def take_optimal_step(carry_step):
            ep_ts, env_state, obs, rng, done, reward, act_onehot, hstates, last_info = carry_step
            vf_hstates = hstate

            # Get available actions for the agent from environment state
            avail_actions = env.get_avail_actions(env_state.env_state)
            avail_actions = jax.lax.stop_gradient(avail_actions)
            avail_actions_per_agent_step = [get_agent_data(avail_actions, i).astype(jnp.float32) for i in range(num_agents)]

            obs_per_agent = [get_agent_data(obs, i).reshape(1, 1, -1) for i in range(num_agents)]

            rng, *act_rngs, step_rng = jax.random.split(rng, num_agents + 2)

            # Get optimal action based on agent_idx using JAX switch
            def make_optimal_branch(i):
                def branch():
                    act, hstate_next = vf_policies[i].get_action(
                        params=vf_params[i],
                        obs=obs_per_agent[i],
                        done=get_agent_data(done, i).reshape(1, 1),
                        avail_actions=avail_actions_per_agent_step[i],
                        hstate=hstates,
                        rng=act_rngs[i],
                        aux_obs=None,
                        env_state=env_state,
                        test_mode=True
                    )
                    # updated = tuple(hstate_next if j == i else vf_hstates[j] for j in range(num_agents))
                    # return act.squeeze(axis=0), updated
                    return act.squeeze(axis=0), hstate_next
                return branch

            act, hstate_next = jax.lax.switch(agent_idx, [make_optimal_branch(i) for i in range(num_agents)])

            env_act = {k: jnp.zeros_like(act) for i, k in enumerate(optimal_env.agents)}
            env_act = set_agent_data(env_act, agent_idx, act)
            env_act_onehot = {k: jax.nn.one_hot(env_act[optimal_env.agents[i]], optimal_env.action_space(optimal_env.agents[i]).n) for i, k in enumerate(optimal_env.agents)}
            (obs_next, obs_full_next), env_state_next, reward, done_next, info_next = optimal_env.step(step_rng, env_state, env_act)

            return (ep_ts + 1, env_state_next, obs_next, rng, done_next, reward, env_act_onehot, hstate_next, info_next)

        ep_ts, env_state, obs, rng, done, reward, act_onehot, hstate, last_info = carry
        output = carry
        new_carry = jax.lax.cond(
            done["__all__"],
            lambda curr_carry: curr_carry,  # True fn
            take_optimal_step,  # False fn
            operand=carry
        )
        return new_carry, output

    optimal_final_carry, optimal_stacked_carry = jax.lax.scan(
        optimal_scan_step, optimal_init_carry, None, length=max_episode_steps)
    optimal_case_info = optimal_final_carry[-1]

    if render:
        # If rendering, we return all step data (stacked_carry contains data for each timestep)
        return (worst_stacked_carry, optimal_stacked_carry, init_env_state)
    else:
        # If not rendering, we just return the final info (which includes the episode return via LogWrapper)
        # Return the final info (which includes the episode return via LogWrapper).
        return (worst_case_info, optimal_case_info)

def run_episodes(rng, env, optimal_env, agent_idx, agent_params, agent_policy,
                 vf_params, vf_policies,
                 max_episode_steps, num_eps, epsilon_optimal, use_full_obs=True, render=False, agent_test_mode=False):
    '''Run num_eps episodes sequentially using scan.'''
    # Create episode-specific RNGs
    rngs = jax.random.split(rng, num_eps + 1)
    ep_rngs = rngs[1:]

    # Define scan function to run episodes sequentially
    def scan_episode(carry, ep_rng):
        all_out = run_single_episode(
            ep_rng, env, optimal_env, agent_idx, agent_params, agent_policy,
            vf_params, vf_policies,
            max_episode_steps, epsilon_optimal, use_full_obs, render, agent_test_mode
        )
        return carry, all_out

    # Run episodes sequentially using scan
    _, all_outs = jax.lax.scan(scan_episode, None, ep_rngs)
    return all_outs  # each leaf has shape (num_eps, ...)

def run_episodes_vmap(rng, env, optimal_env, agent_idx, agent_params, agent_policy,
                      vf_params, vf_policies,
                      max_episode_steps, num_eps, epsilon_optimal, use_full_obs=True, render=False, agent_test_mode=False):
    '''Run num_eps episodes in parallel using vmap.'''
    # Create episode-specific RNGs
    rngs = jax.random.split(rng, num_eps + 1)
    ep_rngs = rngs[1:]

    # Vmap over episodes - all episodes run in parallel
    vmapped_run = jax.vmap(
        run_single_episode,
        in_axes=(0, None, None, None, None, None,
                 None, None, None, None, None, None, None),  # only RNG varies
        out_axes=0  # stack outputs along first axis
    )

    all_outs = vmapped_run(
        ep_rngs, env, optimal_env, agent_idx, agent_params, agent_policy,
        vf_params, vf_policies,
        max_episode_steps, epsilon_optimal, use_full_obs, render, agent_test_mode
    )

    return all_outs  # each leaf has shape (num_eps, ...)
