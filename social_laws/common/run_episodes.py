'''
This file contains the code for running evaluation episodes with an ego agent and a partner agent.
Does not currently support actors that require aux_obs.
'''
import jax
import jax.numpy as jnp
from functools import partial


def run_single_episode(rng, env, agent_idx, agent_param, agent_policy,
                       max_episode_steps, render=False, agent_test_mode=False):
    # Reset the env.
    rng, reset_rng = jax.random.split(rng)
    (init_obs, init_obs_full), init_env_state = env.reset(reset_rng)
    init_done = {k: jnp.zeros((1), dtype=bool) for k in env.agents + ["__all__"]}
    init_reward = {k: jnp.zeros((1)) for i, k in enumerate(env.agents)}

    # Initialize hidden states. Agent id is passed as part of the hstate initialization to support heuristic agents.
    init_hstate = agent_policy.init_hstate(1, aux_info={"agent_id": agent_idx})

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
    avail_actions = get_agent_data(avail_actions, agent_idx).astype(jnp.float32)

    # Do one step to get a dummy info structure
    rng, act_rng, step_rng = jax.random.split(rng, 3)

    # Get ego action
    act, hstate = agent_policy.get_action(
        params=agent_param,
        obs=get_agent_data(init_obs, agent_idx).reshape(1, 1, -1),
        done=get_agent_data(init_done, agent_idx).reshape(1, 1),
        avail_actions=avail_actions,
        hstate=init_hstate,
        rng=act_rng,
        aux_obs=None,
        env_state=init_env_state,
        test_mode=agent_test_mode
    )
    act = act.squeeze(axis=0)

    env_act = {k: jnp.zeros_like(act) for i, k in enumerate(env.agents)}
    env_act = set_agent_data(env_act, agent_idx, act)
    env_act_onehot = {k: jax.nn.one_hot(env_act[env.agents[i]], env.action_space(env.agents[i]).n) for i, k in enumerate(env.agents)}
    (obs, obs_full), env_state, reward, done, dummy_info = env.step(step_rng, init_env_state, env_act)

    # We'll use a scan to iterate steps until the episode is done.
    ep_ts = 1
    init_carry = (ep_ts, env_state, obs, rng, done, reward, env_act_onehot, hstate, dummy_info)
    def scan_step(carry, _):
        def take_step(carry_step):
            ep_ts, env_state, obs, rng, done, reward, act_onehot, hstate, last_info = carry_step
            # Get available actions for the agent from environment state
            avail_actions = env.get_avail_actions(env_state.env_state)
            avail_actions = jax.lax.stop_gradient(avail_actions)
            avail_actions = get_agent_data(avail_actions, agent_idx).astype(jnp.float32)

            # Get ego action
            rng, act_rng, step_rng = jax.random.split(rng, 3)
            act, hstate_next = agent_policy.get_action(
                params=agent_param,
                obs=get_agent_data(obs, agent_idx).reshape(1, 1, -1),
                done=get_agent_data(done, agent_idx).reshape(1, 1),
                avail_actions=avail_actions,
                hstate=hstate,
                rng=act_rng,
                aux_obs=None,
                env_state=env_state,
                test_mode=agent_test_mode
            )
            act = act.squeeze(axis=0)

            env_act = {k: jnp.zeros_like(act) for i, k in enumerate(env.agents)}
            env_act = set_agent_data(env_act, agent_idx, act)
            env_act_onehot = {k: jax.nn.one_hot(env_act[env.agents[i]], env.action_space(env.agents[i]).n) for i, k in enumerate(env.agents)}
            (obs_next, obs_full_next), env_state_next, reward, done_next, info_next = env.step(step_rng, env_state, env_act)

            return (ep_ts + 1, env_state_next, obs_next, rng, done_next, reward, env_act_onehot, hstate_next, info_next)

        ep_ts, env_state, obs, rng, done, reward, act_onehot, hstate, last_info = carry
        output = carry
        new_carry = jax.lax.cond(
            done["__all__"],
            lambda curr_carry: curr_carry, # True fn
            take_step, # False fn
            operand=carry
        )
        return new_carry, output

    final_carry, stacked_carry = jax.lax.scan(
        scan_step, init_carry, None, length=max_episode_steps)

    if render:
        # If rendering, we return all step data (stacked_carry contains data for each timestep)
        return (stacked_carry, init_env_state)
    else:
        # If not rendering, we just return the final info (which includes the episode return via LogWrapper)
        # Return the final info (which includes the episode return via LogWrapper).
        return final_carry[-1]

def run_episodes(rng, env, agent_idx, agent_param, agent_policy,
                 max_episode_steps, num_eps, render=False, agent_test_mode=False):
    '''Run num_eps episodes sequentially using scan.'''
    # Create episode-specific RNGs
    rngs = jax.random.split(rng, num_eps + 1)
    ep_rngs = rngs[1:]

    # Define scan function to run episodes sequentially
    def scan_episode(carry, ep_rng):
        all_out = run_single_episode(
            ep_rng, env, agent_idx, agent_param, agent_policy,
            max_episode_steps, render, agent_test_mode
        )
        return carry, all_out

    # Run episodes sequentially using scan
    _, all_outs = jax.lax.scan(scan_episode, None, ep_rngs)
    return all_outs  # each leaf has shape (num_eps, ...)

def run_episodes_vmap(rng, env, agent_idx, agent_param, agent_policy,
                      max_episode_steps, num_eps, render=False, agent_test_mode=False):
    '''Run num_eps episodes in parallel using vmap.'''
    # Create episode-specific RNGs
    rngs = jax.random.split(rng, num_eps + 1)
    ep_rngs = rngs[1:]

    # Vmap over episodes - all episodes run in parallel
    vmapped_run = jax.vmap(
        run_single_episode,
        in_axes=(0, None, None, None, None, None, None, None),  # only RNG varies
        out_axes=0  # stack outputs along first axis
    )

    all_outs = vmapped_run(
        ep_rngs, env, agent_idx, agent_param, agent_policy,
        max_episode_steps, render, agent_test_mode
    )

    return all_outs  # each leaf has shape (num_eps, ...)
