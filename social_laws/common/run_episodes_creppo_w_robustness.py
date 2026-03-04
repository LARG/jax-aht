'''
This file contains the code for running evaluation episodes with an ego agent and a partner agent.
Does not currently support actors that require aux_obs.
'''
import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple

class Transition(NamedTuple):
    worst_case_reward: jnp.ndarray
    optimal_reward: jnp.ndarray

def _get_optimal_restricted_avail_actions(obs, done, avail_actions, hstate, optimal_params, optimal_policy, epsilon_optimal):
    """
    Get restricted available actions based on value function Q-values.
    Only allows actions within epsilon of the maximum Q-value.

    Args:
        obs: observations (shape: num_envs, obs_dim)
        done: done flags (shape: num_envs,)
        avail_actions: available actions mask (shape: num_envs, num_actions)
        hstate: hidden state for the value function policy (shape depends on policy)
        optimal_params: value function parameters
        optimal_policy: value function policy
        epsilon_optimal: threshold for considering actions as near-optimal

    Returns:
        restricted_avail_actions: mask with only near-optimal actions (shape: num_envs, num_actions)
        next_hstate: next hidden state from the value function policy (shape depends on policy)
    """
    critic_outs, new_hstate = optimal_policy.get_critic_out(
        params=(optimal_params["params"], optimal_params["batch_stats"]),
        obs=obs,
        done=done,
        avail_actions=avail_actions,
        hstate=hstate,
        rng=jax.random.PRNGKey(0)
    ) # (num_envs, num_actions)

    # Find max Q-value for each environment
    max_q = jnp.max(critic_outs["q_values"], axis=-1, keepdims=True)  # (num_envs, 1)

    # Create mask for actions within epsilon of max
    optimal_action_mask = (critic_outs["q_values"] >= (max_q - epsilon_optimal)).astype(jnp.float32)

    # Combine with original available actions
    restricted_avail_actions = optimal_action_mask * avail_actions

    # remove extra batch dim
    return restricted_avail_actions, new_hstate

def run_single_episode(rng, env, optimal_env, agent_idx, agent_params, agent_policies,
                       optimal_params, optimal_policies,
                       max_episode_steps, epsilon_optimal, use_full_obs, render, agent_test_mode):
    # Reset the env.
    rng, reset_rng = jax.random.split(rng)
    (init_obs, init_obs_full), init_env_state = env.reset(reset_rng)
    init_done = {k: jnp.zeros((1), dtype=bool) for k in env.agents + ["__all__"]}

    (init_opt_obs, init_opt_obs_full), init_opt_env_state = optimal_env.reset(reset_rng)
    init_opt_done = {k: jnp.zeros((1), dtype=bool) for k in optimal_env.agents + ["__all__"]}

    init_reward = {k: jnp.zeros((1)) for i, k in enumerate(env.agents)}

    num_agents = len(env.agents)

    # Initialize hidden states. Agent id is passed as part of the hstate initialization to support heuristic agents.
    init_hstates = [agent_policies[i].init_hstate(1, aux_info={"agent_id": i}) for i in range(num_agents)]
    optimal_init_hstates = [optimal_policies[i].init_hstate(1, aux_info={"agent_id": i}) for i in range(num_agents)]

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

    # Per-agent obs/done for worst-case env
    avail_actions = env.get_avail_actions(init_env_state.env_state)
    avail_actions = jax.lax.stop_gradient(avail_actions)
    init_avail_per_agent = [get_agent_data(avail_actions, i).astype(jnp.float32) for i in range(num_agents)]

    init_obs_per_agent = [get_agent_data(init_obs, i).reshape(1, 1, -1) for i in range(num_agents)]
    init_obs_full_per_agent = [get_agent_data(init_obs_full, i).reshape(1, 1, -1) for i in range(num_agents)]
    init_done_per_agent = [get_agent_data(init_done, i).reshape(1, 1) for i in range(num_agents)]

    # Per-agent obs/done for optimal env
    opt_avail_actions = env.get_avail_actions(init_opt_env_state.env_state)
    opt_avail_actions = jax.lax.stop_gradient(opt_avail_actions)

    init_opt_avail_per_agent = [get_agent_data(opt_avail_actions, i).astype(jnp.float32) for i in range(num_agents)]
    init_opt_obs_per_agent = [get_agent_data(init_opt_obs, i).reshape(1, 1, -1) for i in range(num_agents)]
    init_opt_done_per_agent = [get_agent_data(init_opt_done, i).reshape(1, 1) for i in range(num_agents)]

    # Restrict available actions based on optimal critic for each agent
    optimal_restricted_avail_actions = []
    optimal_vf_hstates = []
    for i in range(num_agents):
        restricted_i, hstate_i = _get_optimal_restricted_avail_actions(
            obs=init_opt_obs_per_agent[i],
            done=init_opt_done_per_agent[i],
            avail_actions=init_opt_avail_per_agent[i],
            hstate=optimal_init_hstates[i],
            optimal_params=optimal_params[i],
            optimal_policy=optimal_policies[i],
            epsilon_optimal=epsilon_optimal
        )
        optimal_restricted_avail_actions.append(restricted_i)
        optimal_vf_hstates.append(hstate_i)

    # Do one step to get a dummy info structure
    rng, *act_rngs, step_rng = jax.random.split(rng, num_agents + 2)

    ###########################################
    #                Worst Case               #
    ###########################################

    acts = []
    hstates = []
    for i in range(num_agents):
        act_i, hstate_i = agent_policies[i].get_action(
            params=(agent_params[i]["params"], agent_params[i]["batch_stats"]),
            obs=init_obs_full_per_agent[i] if use_full_obs else init_obs_per_agent[i],
            done=init_done_per_agent[i],
            avail_actions=optimal_restricted_avail_actions[i],
            hstate=init_hstates[i],
            rng=act_rngs[i],
            aux_obs=None,
            env_state=init_env_state,
            test_mode=agent_test_mode
        )
        acts.append(act_i.squeeze(axis=0))
        hstates.append(hstate_i)

    env_act = {k: acts[i] for i, k in enumerate(env.agents)}
    env_act_onehot = {k: jax.nn.one_hot(env_act[env.agents[i]], env.action_space(env.agents[i]).n) for i, k in enumerate(env.agents)}
    obs, env_state, reward, done, info = env.step(step_rng, init_env_state, env_act)

    ###########################################
    #               Optimal Case              #
    ###########################################

    def make_get_optimal_agent(agent_i):
        def get_optimal_agent_i():
            act, hstate_next = optimal_policies[agent_i].get_action(
                params=(optimal_params[agent_i]["params"], optimal_params[agent_i]["batch_stats"]),
                obs=init_opt_obs_per_agent[agent_i],
                done=init_opt_done_per_agent[agent_i],
                avail_actions=init_opt_avail_per_agent[agent_i],
                hstate=optimal_init_hstates[agent_i],
                rng=act_rngs[agent_i],
                aux_obs=None,
                env_state=init_opt_env_state,
                test_mode=agent_test_mode
            )
            # Build output hstate tuple: updated slot for agent_i, initial for all others
            # hstates_out = tuple(
            #     hstate_next if j == agent_i else optimal_init_hstates[j]
            #     for j in range(num_agents)
            # )
            # return act.squeeze(axis=0), hstates_out
            return act.squeeze(axis=0), hstate_next
        return get_optimal_agent_i

    optimal_act, optimal_hstates = jax.lax.switch(agent_idx, [make_get_optimal_agent(i) for i in range(num_agents)])

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
    worst_init_carry = (worst_ep_ts, env_state, obs, rng, done, reward, env_act_onehot,
                        hstates, optimal_vf_hstates, info)
    def worst_scan_step(carry, _):
        def take_worst_step(carry_step):
            ep_ts, env_state, (obs, obs_full), rng, done, reward, act_onehot, hstates, optimal_vf_hstates, last_info = carry_step

            avail_actions = env.get_avail_actions(env_state.env_state)
            avail_actions = jax.lax.stop_gradient(avail_actions)
            avail_per_agent = [get_agent_data(avail_actions, i).astype(jnp.float32) for i in range(num_agents)]

            obs_per_agent = [get_agent_data(obs, i).reshape(1, 1, -1) for i in range(num_agents)]
            obs_full_per_agent = [get_agent_data(obs_full, i).reshape(1, 1, -1) for i in range(num_agents)]
            done_per_agent = [get_agent_data(done, i).reshape(1, 1) for i in range(num_agents)]

            # Restrict available actions via optimal critic for each agent
            opt_restricted_avail_actions = []
            new_opt_hstates = []
            for i in range(num_agents):
                restricted_i, new_opt_hstate_i = _get_optimal_restricted_avail_actions(
                    obs=obs_per_agent[i],
                    done=done_per_agent[i],
                    avail_actions=avail_per_agent[i],
                    hstate=optimal_vf_hstates[i],
                    optimal_params=optimal_params[i],
                    optimal_policy=optimal_policies[i],
                    epsilon_optimal=epsilon_optimal
                )
                opt_restricted_avail_actions.append(restricted_i)
                new_opt_hstates.append(new_opt_hstate_i)

            rng, *act_rngs, step_rng = jax.random.split(rng, num_agents + 2)

            acts = []
            new_hstates = []
            for i in range(num_agents):
                act_i, new_hstate_i = agent_policies[i].get_action(
                    params=(agent_params[i]["params"], agent_params[i]["batch_stats"]),
                    obs=obs_full_per_agent[i] if use_full_obs else obs_per_agent[i],
                    done=done_per_agent[i],
                    avail_actions=opt_restricted_avail_actions[i],
                    hstate=hstates[i],
                    rng=act_rngs[i],
                    aux_obs=None,
                    env_state=env_state,
                    test_mode=agent_test_mode
                )
                acts.append(act_i.squeeze(axis=0))
                new_hstates.append(new_hstate_i)

            env_act = {k: acts[i] for i, k in enumerate(env.agents)}
            env_act_onehot = {k: jax.nn.one_hot(env_act[env.agents[i]], env.action_space(env.agents[i]).n) for i, k in enumerate(env.agents)}
            obs_next, env_state_next, reward, done_next, info_next = env.step(step_rng, env_state, env_act)

            return (ep_ts + 1, env_state_next, obs_next, rng, done_next, reward, env_act_onehot,
                    new_hstates, new_opt_hstates, info_next)

        ep_ts, env_state, obs, rng, done, reward, act_onehot, hstates, optimal_vf_hstates, last_info = carry
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
    optimal_init_carry = (optimal_ep_ts, optimal_env_state, optimal_obs, rng, optimal_done, optimal_reward, optimal_env_act_onehot, optimal_hstates, optimal_info)
    def optimal_scan_step(carry, _):
        def take_optimal_step(carry_step):
            ep_ts, env_state, obs, rng, done, reward, act_onehot, hstates, last_info = carry_step

            avail_actions = env.get_avail_actions(env_state.env_state)
            avail_actions = jax.lax.stop_gradient(avail_actions)
            avail_per_agent = [get_agent_data(avail_actions, i).astype(jnp.float32) for i in range(num_agents)]

            obs_per_agent = [get_agent_data(obs, i).reshape(1, 1, -1) for i in range(num_agents)]
            done_per_agent = [get_agent_data(done, i).reshape(1, 1) for i in range(num_agents)]

            rng, *act_rngs, step_rng = jax.random.split(rng, num_agents + 2)

            def make_optimal_step_fn(agent_i):
                def optimal_step_fn():
                    act, hstate_next = optimal_policies[agent_i].get_action(
                        params=(optimal_params[agent_i]["params"], optimal_params[agent_i]["batch_stats"]),
                        obs=obs_per_agent[agent_i],
                        done=done_per_agent[agent_i],
                        avail_actions=avail_per_agent[agent_i],
                        hstate=hstates,
                        rng=act_rngs[agent_i],
                        aux_obs=None,
                        env_state=env_state,
                        test_mode=agent_test_mode
                    )
                    # Only the ego agent's hstate is updated; others carry forward unchanged
                    # hstates_out = tuple(
                    #     hstate_next if j == agent_i else hstates[j]
                    #     for j in range(num_agents)
                    # )
                    # return act.squeeze(axis=0), hstates_out
                    return act.squeeze(axis=0), hstate_next
                return optimal_step_fn

            act, hstates_next = jax.lax.switch(agent_idx, [make_optimal_step_fn(i) for i in range(num_agents)])

            env_act = {k: jnp.zeros_like(act) for i, k in enumerate(optimal_env.agents)}
            env_act = set_agent_data(env_act, agent_idx, act)
            env_act_onehot = {k: jax.nn.one_hot(env_act[optimal_env.agents[i]], optimal_env.action_space(optimal_env.agents[i]).n) for i, k in enumerate(optimal_env.agents)}
            (obs_next, obs_full_next), env_state_next, reward, done_next, info_next = optimal_env.step(step_rng, env_state, env_act)

            return (ep_ts + 1, env_state_next, obs_next, rng, done_next, reward, env_act_onehot, hstates_next, info_next)

        ep_ts, env_state, obs, rng, done, reward, act_onehot, hstates, last_info = carry
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

def run_episodes(rng, env, optimal_env, agent_idx, agent_params, agent_policies,
                 optimal_params, optimal_policies,
                 max_episode_steps, num_eps, epsilon_optimal, use_full_obs=True, render=False, agent_test_mode=True):
    '''Run num_eps episodes sequentially using scan.'''
    # Create episode-specific RNGs
    rngs = jax.random.split(rng, num_eps + 1)
    ep_rngs = rngs[1:]

    # Define scan function to run episodes sequentially
    def scan_episode(carry, ep_rng):
        all_out = run_single_episode(
            ep_rng, env, optimal_env, agent_idx, agent_params, agent_policies,
            optimal_params, optimal_policies,
            max_episode_steps, epsilon_optimal, use_full_obs, render, agent_test_mode
        )
        return carry, all_out

    # Run episodes sequentially using scan
    _, all_outs = jax.lax.scan(scan_episode, None, ep_rngs)
    return all_outs  # each leaf has shape (num_eps, ...)

def run_episodes_vmap(rng, env, optimal_env, agent_idx, agent_params, agent_policies,
                      optimal_params, optimal_policies,
                      max_episode_steps, num_eps, epsilon_optimal, use_full_obs=True, render=False, agent_test_mode=True):
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
        ep_rngs, env, optimal_env, agent_idx, agent_params, agent_policies,
        optimal_params, optimal_policies,
        max_episode_steps, epsilon_optimal, use_full_obs, render, agent_test_mode
    )

    return all_outs  # each leaf has shape (num_eps, ...)
