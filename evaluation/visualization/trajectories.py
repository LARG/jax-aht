# TODO: collect trajectories
# @ Johnny

from typing import List
import jax
import jax.numpy as jnp

# this function is a modified version of common.run_episodes.run_single_episode
def run_single_episode(rng, env, agent_0_param, agent_0_policy, 
                       agent_1_param, agent_1_policy, 
                       max_episode_steps, agent_0_test_mode=False, agent_1_test_mode=False):
    # Reset the env.
    rng, reset_rng = jax.random.split(rng)
    obs, env_state = env.reset(reset_rng)
    init_done = {k: jnp.zeros((1), dtype=bool) for k in env.agents + ["__all__"]}
    
    # Initialize hidden states
    init_hstate_0 = agent_0_policy.init_hstate(1, aux_info={"agent_id": 0})
    init_hstate_1 = agent_1_policy.init_hstate(1, aux_info={"agent_id": 1})

    # Get agent obses
    obs_0 = obs["agent_0"]
    obs_1 = obs["agent_1"]

    # Get available actions for agent 0 from environment state
    avail_actions = env.get_avail_actions(env_state.env_state)
    avail_actions = jax.lax.stop_gradient(avail_actions)
    avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
    avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)

    # Do one step to get a dummy info structure
    rng, act1_rng, act2_rng, step_rng = jax.random.split(rng, 4)
    
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
        act1_rng,
        aux_obs=None,
        env_state=env_state,
        test_mode=agent_0_test_mode
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
        init_hstate_1,  # shape of entry 0 is (1, 1, 8)
        act2_rng,
        aux_obs=None,
        env_state=env_state,
        test_mode=agent_1_test_mode
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
                act_rng,
                env_state=env_state,
                test_mode=agent_0_test_mode
            )
            act_0 = act_0.squeeze()

            # Get partner action with proper hidden state tracking
            act_1, hstate_1_next = agent_1_policy.get_action(
                agent_1_param, 
                obs_1_reshaped,
                done_1_reshaped,
                avail_actions_1,
                hstate_1,
                part_rng,
                env_state=env_state,
                test_mode=agent_1_test_mode
            )
            act_1 = act_1.squeeze()
            
            both_actions = [act_0, act_1]
            env_act = {k: both_actions[i] for i, k in enumerate(env.agents)}
            obs_next, env_state_next, reward, done_next, info_next = env.step(step_rng, env_state, env_act)

            return (ep_ts + 1, env_state_next, obs_next, rng, done_next, hstate_0_next, hstate_1_next, info_next), both_actions
                
        ep_ts, env_state, obs, rng, done, hstate_0, hstate_1, last_info = carry
        
        dummy_action = jnp.array(-1, dtype=jnp.int32)  
        new_carry, both_actions = jax.lax.cond(
            done["__all__"],
            # if done, execute true function(operand). else, execute false function(operand).
            lambda curr_carry: (curr_carry, [dummy_action, dummy_action]), # True fn, both_actions: have some degenerate action values
            take_step, # False fn
            operand=carry
        )

        trajectory_state = (new_carry[1], both_actions)

        return new_carry, trajectory_state

    final_carry, trajectory = jax.lax.scan(
        scan_step, init_carry, None, length=max_episode_steps)
    # Return the final info (which includes the episode return via LogWrapper).
    
    return final_carry[-1], trajectory

# similar to the common.run_episodes.run_episodes() function
# collects n trajectories for the agent pair
def collect_n_trajectories(rng, env, agent_0_param, agent_0_policy, 
                 agent_1_param, agent_1_policy, 
                 max_episode_steps, num_eps, agent_0_test_mode=False, agent_1_test_mode=False):
    
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
    _, all_trajectories = vmap_run_single_episode(ep_rngs)

    # Coalesce all_trajectories into a more readable format

    # TODO: coalesce the states part of the trajectory into a more readable format
    # currently we have a Log_Wrapper of the form where each leaf in the pytree is stacked
    # Leafs: ['agent_dir', 'agent_dir_idx', 'agent_inv', 'agent_pos', 'goal_pos', 'maze_map', 'pot_pos', 'replace', 'terminal', 'time', 'wall_map']
    # Ex: maze_map on cramped room has shape (6, 400, 4, 5)
    # @Johnny

    return_val = {
        "actions": jnp.transpose(jnp.stack([all_trajectories[1][0], all_trajectories[1][1]], axis=1), axes=(0, 2, 1)),
        # actions[i][j][k] is the action of trajectory # i (0-(n_episodes-1)), at step # j (0-399), of agent # k (0-1)
        "states": all_trajectories[0]
    }

    return return_val

# TODO: collect trajectories
# @ to-be-assigned

def get_events_from_trajectory(trajectory) -> jax.Array:
    return None

def aggregate_events(data: List[jax.Array]):
    return None


# sample usage
if __name__ == "__main__":

    a = jnp.array([[1, 2, 3], [4, 5, 6]])  
    b = jnp.array([[7, 8, 9], [10, 11, 12]])  

    print(a.shape)

    stacked = jnp.stack([a, b], axis=1)
    print(stacked.shape)  # (2, 2, 2)
    print(stacked)

    exit()

    print("sample trajectory collection for visualization with overcooked")

    # load the sample config
    from omegaconf import OmegaConf
    from hydra import initialize, compose
    from omegaconf import OmegaConf

    with initialize(version_base=None, config_path="../configs"):
        config = compose(config_name="heldout.yaml")

    config = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

    heldout_cfg = config["heldout_set"][config["TASK_NAME"]]

    from evaluation.heldout_evaluator import load_heldout_set
    from envs import make_env
    from envs.log_wrapper import LogWrapper

    rng = jax.random.PRNGKey(config["global_heldout_settings"]["EVAL_SEED"])
    heldout_init_rng, eval_rng = jax.random.split(rng, 2)

    env = make_env(config["ENV_NAME"], config["ENV_KWARGS"])
    env = LogWrapper(env)

    heldout_agents = load_heldout_set(heldout_cfg, env, config["TASK_NAME"], config["ENV_KWARGS"], heldout_init_rng)
    heldout_agent_list = list(heldout_agents.values())

    # just collect 6 trajectories with the first two held_out_agents

    policy1, params1, test_mode1 = heldout_agent_list[0]
    policy2, params2, test_mode2 = heldout_agent_list[1]

    print("running episodes")

    result = collect_n_trajectories(eval_rng, env,
        agent_0_policy=policy1, agent_0_param=params1,
        agent_1_policy=policy2, agent_1_param=params2,
        max_episode_steps=config["global_heldout_settings"]["MAX_EPISODE_STEPS"],
        num_eps=6, 
        agent_0_test_mode=test_mode1,
        agent_1_test_mode=test_mode2)


