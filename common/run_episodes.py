'''
This file contains the code for running evaluation episodes with an ego agent and a partner agent.
Does not currently support actors that require aux_obs.
'''
import jax
import jax.numpy as jnp


def run_single_episode(rng, env, agent_0_param, agent_0_policy,
                       agent_1_param, agent_1_policy,
                       max_episode_steps, agent_0_test_mode=False, agent_1_test_mode=False):
    # Reset the env.
    rng, reset_rng = jax.random.split(rng)
    init_obs, init_env_state = env.reset(reset_rng)
    init_done = {k: jnp.zeros((1), dtype=bool) for k in env.agents + ["__all__"]}
    init_act_onehot = {k: jnp.zeros((env.action_space(env.agents[i]).n)) for i, k in enumerate(env.agents)}

    # Initialize hidden states. Agent id is passed as part of the hstate initialization to support heuristic agents.
    init_hstate_0 = agent_0_policy.init_hstate(1, aux_info={"agent_id": 0})
    init_hstate_1 = agent_1_policy.init_hstate(1, aux_info={"agent_id": 1})

    # Get available actions for agent 0 from environment state
    avail_actions = env.get_avail_actions(init_env_state.env_state)
    avail_actions = jax.lax.stop_gradient(avail_actions)
    avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
    avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)

    # Do one step to get a dummy info structure
    rng, act0_rng, act1_rng, step_rng = jax.random.split(rng, 4)

    # Get ego action
    act_0, hstate_0 = agent_0_policy.get_action(
        params=agent_0_param,
        obs=init_obs["agent_0"].reshape(1, 1, -1),
        done=init_done["agent_0"].reshape(1, 1),
        avail_actions=avail_actions_0,
        hstate=init_hstate_0,
        rng=act0_rng,
        aux_obs=init_act_onehot["agent_0"].reshape(1, 1, -1),
        env_state=init_env_state,
        test_mode=agent_0_test_mode
    )
    act_0 = act_0.squeeze()

    # Get partner action using the underlying policy class's get_action method directly
    act_1, hstate_1 = agent_1_policy.get_action(
        params=agent_1_param,
        obs=init_obs["agent_1"].reshape(1, 1, -1),
        done=init_done["agent_1"].reshape(1, 1),
        avail_actions=avail_actions_1,
        hstate=init_hstate_1,  # shape of entry 0 is (1, 1, 8)
        rng=act1_rng,
        aux_obs=None,
        env_state=init_env_state,
        test_mode=agent_1_test_mode
    )
    act_1 = act_1.squeeze()

    both_actions = [act_0, act_1]
    env_act = {k: both_actions[i] for i, k in enumerate(env.agents)}
    env_act_onehot = {k: jax.nn.one_hot(both_actions[i], env.action_space(env.agents[i]).n) for i, k in enumerate(env.agents)}
    obs, env_state, _, done, dummy_info = env.step(step_rng, init_env_state, env_act)

    # We'll use a scan to iterate steps until the episode is done.
    ep_ts = 1
    init_carry = (ep_ts, env_state, obs, rng, done, env_act_onehot, hstate_0, hstate_1, dummy_info)
    def scan_step(carry, _):
        def take_step(carry_step):
            ep_ts, env_state, obs, rng, done, act_onehot, hstate_0, hstate_1, last_info = carry_step
            # Get available actions for agent 0 from environment state
            avail_actions = env.get_avail_actions(env_state.env_state)
            avail_actions = jax.lax.stop_gradient(avail_actions)
            avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
            avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)

            # Get ego action
            rng, act0_rng, act1_rng, step_rng = jax.random.split(rng, 4)
            act_0, hstate_0_next = agent_0_policy.get_action(
                params=agent_0_param,
                obs=obs["agent_0"].reshape(1, 1, -1),
                done=done["agent_0"].reshape(1, 1),
                avail_actions=avail_actions_0,
                hstate=hstate_0,
                rng=act0_rng,
                aux_obs=act_onehot["agent_0"].reshape(1, 1, -1),
                env_state=env_state,
                test_mode=agent_0_test_mode
            )
            act_0 = act_0.squeeze()

            # Get partner action with proper hidden state tracking
            act_1, hstate_1_next = agent_1_policy.get_action(
                params=agent_1_param,
                obs=obs["agent_1"].reshape(1, 1, -1),
                done=done["agent_1"].reshape(1, 1),
                avail_actions=avail_actions_1,
                hstate=hstate_1,
                rng=act1_rng,
                env_state=env_state,
                test_mode=agent_1_test_mode
            )
            act_1 = act_1.squeeze()

            both_actions = [act_0, act_1]
            env_act = {k: both_actions[i] for i, k in enumerate(env.agents)}
            env_act_onehot = {k: jax.nn.one_hot(both_actions[i], env.action_space(env.agents[i]).n) for i, k in enumerate(env.agents)}
            obs_next, env_state_next, reward, done_next, info_next = env.step(step_rng, env_state, env_act)

            return (ep_ts + 1, env_state_next, obs_next, rng, done_next, env_act_onehot, hstate_0_next, hstate_1_next, info_next)

        ep_ts, env_state, obs, rng, done, act_onehot, hstate_0, hstate_1, last_info = carry
        new_carry = jax.lax.cond(
            done["__all__"],
            lambda curr_carry: curr_carry, # True fn
            take_step, # False fn
            operand=carry
        )
        return new_carry, None

    final_carry, _ = jax.lax.scan(
        scan_step, init_carry, None, length=max_episode_steps)
    # Return the final info (which includes the episode return via LogWrapper).
    return final_carry[-1]

def run_episodes(rng, env, agent_0_param, agent_0_policy,
                 agent_1_param, agent_1_policy,
                 max_episode_steps, num_eps, agent_0_test_mode=False, agent_1_test_mode=False):
    '''Given a single ego agent and a single partner agent, run num_eps episodes in parallel using vmap.'''
    # Create episode-specific RNGs
    rngs = jax.random.split(rng, num_eps + 1)
    ep_rngs = rngs[1:]

    # Vectorize run_single_episode over the first argument (rng)
    vmap_run_single_episode = jax.jit(jax.vmap(
        lambda ep_rng: run_single_episode(
            ep_rng, env, agent_0_param, agent_0_policy,
            agent_1_param, agent_1_policy, max_episode_steps,
            agent_0_test_mode, agent_1_test_mode
        )
    ))
    # Run episodes in parallel
    all_outs = vmap_run_single_episode(ep_rngs)
    return all_outs  # each leaf has shape (num_eps, ...)
