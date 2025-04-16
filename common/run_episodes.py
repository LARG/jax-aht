'''
This file contains the code for running episodes with some ego agent and some population 
of partner agents in the environment.
'''
import jax
import jax.numpy as jnp


def run_single_episode(rng, env, agent_0_param, agent_0_policy, 
                       agent_1_param, agent_1_policy, 
                       max_episode_steps
                       ):
    '''
    Agent 0 is the ego agent, agent 1 is the partner agent.
    '''
    # Reset the env.
    rng, reset_rng = jax.random.split(rng)
    obs, env_state = env.reset(reset_rng)
    init_done = {k: jnp.zeros((1), dtype=bool) for k in env.agents + ["__all__"]}
    
    # Initialize hidden states
    init_hstate_0 = agent_0_policy.init_hstate(1)
    init_hstate_1 = agent_1_policy.init_hstate(1)

    # Get agent obses
    obs_0 = obs["agent_0"]
    obs_1 = obs["agent_1"]

    # Get available actions for agent 0 from environment state
    avail_actions = env.get_avail_actions(env_state.env_state)
    avail_actions = jax.lax.stop_gradient(avail_actions)
    avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
    avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)

    # Do one step to get a dummy info structure
    rng, act_rng, part_rng, step_rng = jax.random.split(rng, 4)
    
    # Reshape inputs
    obs_0_reshaped = obs_0.reshape(1, 1, -1)
    done_0_reshaped = init_done["agent_0"].reshape(1, 1)
    
    # Get ego action
    act_0, hstate_0 = agent_0_policy.get_action(
        agent_0_param,
        obs_0_reshaped,
        done_0_reshaped,
        avail_actions_0,
        init_hstate_0,
        act_rng
    )
    act_0 = act_0.squeeze()

    # Get partner action using the underlying policy class's get_action method directly
    obs_1_reshaped = obs_1.reshape(1, 1, -1)
    done_1_reshaped = init_done["agent_1"].reshape(1, 1)
    act_1, hstate_1 = agent_1_policy.get_action(
        agent_1_param, 
        obs_1_reshaped, 
        done_1_reshaped,
        avail_actions_1,
        init_hstate_1,  # Pass the proper hidden state
        part_rng
    )
    act_1 = act_1.squeeze()
    
    both_actions = [act_0, act_1]
    env_act = {k: both_actions[i] for i, k in enumerate(env.agents)}
    _, _, _, done, dummy_info = env.step(step_rng, env_state, env_act)


    # We'll use a scan to iterate steps until the episode is done.
    ep_ts = 1
    init_carry = (ep_ts, env_state, obs, rng, done, hstate_0, hstate_1, dummy_info)
    def scan_step(carry, _):
        def take_step(carry_step):
            ep_ts, env_state, obs, rng, done, hstate_0, hstate_1, last_info = carry_step
            # Get available actions for agent 0 from environment state
            avail_actions = env.get_avail_actions(env_state.env_state)
            avail_actions = jax.lax.stop_gradient(avail_actions)
            avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
            avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)

            # Get agent obses
            obs_0, obs_1 = obs["agent_0"], obs["agent_1"]
            prev_done_0, prev_done_1 = done["agent_0"], done["agent_1"]
            
            # Reshape inputs for S5
            obs_0_reshaped = obs_0.reshape(1, 1, -1)
            done_0_reshaped = prev_done_0.reshape(1, 1)
            obs_1_reshaped = obs_1.reshape(1, 1, -1)
            done_1_reshaped = prev_done_1.reshape(1, 1)
            
            # Get ego action
            rng, act_rng, part_rng, step_rng = jax.random.split(rng, 4)
            act_0, hstate_0_next = agent_0_policy.get_action(
                agent_0_param,
                obs_0_reshaped,
                done_0_reshaped,
                avail_actions_0,
                hstate_0,
                act_rng
            )
            act_0 = act_0.squeeze()

            # Get partner action with proper hidden state tracking
            act_1, hstate_1_next = agent_1_policy.get_action(
                agent_1_param, 
                obs_1_reshaped,
                done_1_reshaped,
                avail_actions_1,
                hstate_1,
                part_rng
            )
            act_1 = act_1.squeeze()
            
            both_actions = [act_0, act_1]
            env_act = {k: both_actions[i] for i, k in enumerate(env.agents)}
            obs_next, env_state_next, reward, done_next, info_next = env.step(step_rng, env_state, env_act)

            return (ep_ts + 1, env_state_next, obs_next, rng, done_next, hstate_0_next, hstate_1_next, info_next)
                
        ep_ts, env_state, obs, rng, done, hstate_0, hstate_1, last_info = carry
        new_carry = jax.lax.cond(
            done["__all__"],
            # if done, execute true function(operand). else, execute false function(operand).
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
                 max_episode_steps, num_eps):
    '''Given a single ego agent and a single partner agent, run num_eps episodes in parallel using vmap.'''
    # Create episode-specific RNGs
    rngs = jax.random.split(rng, num_eps + 1)
    ep_rngs = rngs[1:]
    
    # Vectorize run_single_episode over the first argument (rng)
    vmap_run_single_episode = jax.vmap(
        lambda ep_rng: run_single_episode(
            ep_rng, env, agent_0_param, agent_0_policy,
            agent_1_param, agent_1_policy, max_episode_steps
        )
    )
    # Run episodes in parallel
    all_outs = vmap_run_single_episode(ep_rngs)
    return all_outs  # each leaf has shape (num_eps, ...)
