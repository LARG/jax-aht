import jax
import jax.numpy as jnp
import os
from envs.overcooked.overcooked_visualizer_v2 import OvercookedVisualizerV2


def save_video(env, env_name, 
               agent_0_param, agent_0_policy, 
               agent_1_param, agent_1_policy, 
               max_episode_steps, num_eps, 
               savevideo: bool, save_dir: str, save_name: str):
    '''
    Render or save video of agent 0 and agent 1 playing against each other.
    
    Args:
        env: The environment instance
        env_name: Name of the environment ('lbf' or 'overcooked-v2')
        agent_0_param: Parameters for agent 0
        agent_0_policy: Policy for agent 0
        agent_1_param: Parameters for agent 1
        agent_1_policy: Policy for agent 1
        max_episode_steps: Maximum number of steps per episode
        num_eps: Number of episodes to run
        savevideo: Whether to save a video of the episode
        save_dir: Directory to save the video
        save_name: Name to use for the saved video
    '''
    assert env_name in ['lbf', 'overcooked-v2'], "Supported environments are lbf or overcooked-v2"
    
    # Step 1: run the episode and generate a list of env states 
    states = []
    rng = jax.random.PRNGKey(112358)
    
    for episode in range(num_eps):
        print(f"Running episode {episode+1}/{num_eps}")
        rng, episode_rng = jax.random.split(rng)
        
        # Run a single episode and collect states
        episode_states = run_episode_with_states(
            episode_rng, env, agent_0_param, agent_0_policy,
            agent_1_param, agent_1_policy, max_episode_steps
        )
        
        states.extend(episode_states)
    
    # Step 2: render or save video
    print(f"\nSaving video with {len(states)} frames...")
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    savepath = f"{save_dir}/{save_name}.mp4"
    if env_name == 'lbf':
        anim = env.animate(states, interval=150)
        anim.save(savepath, writer="ffmpeg")
        print(f"Video saved successfully at {savepath}")

    elif env_name == 'overcooked-v2':
        viz = OvercookedVisualizerV2()
        # Get layout from env kwargs if available, otherwise use default
        viz.animate_mp4(states, env.agent_view_size, 
            filename=savepath, 
            pixels_per_tile=32, fps=25)
        print(f"MP4 saved successfully at {savepath}")
    else:
        print(f"Unknown environment: {env_name}.")

    return savepath


def run_episode_with_states(rng, env, agent_0_param, agent_0_policy, 
                           agent_1_param, agent_1_policy, 
                           max_episode_steps):
    '''
    Run a single episode and collect states for rendering.
    Returns a list of states.
    '''
    # Reset the env.
    rng, reset_rng = jax.random.split(rng)
    obs, env_state = env.reset(reset_rng)
    done = {k: jnp.zeros((1), dtype=bool) for k in env.agents + ["__all__"]}
    
    # Initialize hidden states
    hstate_0 = agent_0_policy.init_hstate(1)
    hstate_1 = agent_1_policy.init_hstate(1)

    # Collect states for rendering
    ep_states = [env_state]
    
    # Run episode until done or max steps reached
    step = 0
    while not done["__all__"] and step < max_episode_steps:
        # Get available actions for each agent
        avail_actions = env.get_avail_actions(env_state)
        avail_actions = jax.lax.stop_gradient(avail_actions)
        avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
        avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)

        # Get agent obses
        obs_0, obs_1 = obs["agent_0"], obs["agent_1"]
        prev_done_0, prev_done_1 = done["agent_0"], done["agent_1"]
        
        # Reshape inputs for policies
        obs_0_reshaped = obs_0.reshape(1, 1, -1)
        done_0_reshaped = prev_done_0.reshape(1, 1)
        obs_1_reshaped = obs_1.reshape(1, 1, -1)
        done_1_reshaped = prev_done_1.reshape(1, 1)
        
        # Get actions for both agents
        rng, act_rng, part_rng, step_rng = jax.random.split(rng, 4)
        
        # Get ego action
        act_0, hstate_0 = agent_0_policy.get_action(
            agent_0_param,
            obs_0_reshaped,
            done_0_reshaped,
            avail_actions_0,
            hstate_0,
            act_rng
        )
        act_0 = act_0.squeeze()

        # Get partner action
        act_1, hstate_1 = agent_1_policy.get_action(
            agent_1_param, 
            obs_1_reshaped,
            done_1_reshaped,
            avail_actions_1,
            hstate_1,
            part_rng
        )
        act_1 = act_1.squeeze()
        
        # Take step in environment
        both_actions = [act_0, act_1]
        env_act = {k: both_actions[i] for i, k in enumerate(env.agents)}
        obs, env_state, reward, done, info = env.step(step_rng, env_state, env_act)

        # Add state to the list for rendering
        ep_states.append(env_state)
        
        step += 1
    
    return ep_states

if __name__ == "__main__":
    from envs import make_env
    from common.initialize_agents import initialize_mlp_agent
    from common.save_load_utils import load_checkpoints

    env = make_env("lbf", {})

    ego_ckpt_path = "results/lbf/ppo_ego_mlp/2025-04-13_23-19-15/saved_train_run" # mlp ego agent
    ego_agent_ckpt = load_checkpoints(ego_ckpt_path)
    # ego_agent_params = jax.tree.map(lambda x: x[0, -1][np.newaxis, ...], ego_agent_ckpt)
    ego_agent_params = jax.tree.map(lambda x: x[0, -1], ego_agent_ckpt)

    # Initialize policies
    base_rng = jax.random.PRNGKey(112358)
    rng, init1_rng, init2_rng = jax.random.split(base_rng, 3)
    
    # Initialize the policies with the loaded parameters
    agent_0_policy, _ = initialize_mlp_agent({}, env, init1_rng)
    agent_1_policy, _ = initialize_mlp_agent({}, env, init2_rng)
    
    # Make sure the policies are properly initialized with the parameters
    
    save_video(env, "lbf", 
        agent_0_param=ego_agent_params, agent_0_policy=agent_0_policy, 
        agent_1_param=ego_agent_params, agent_1_policy=agent_1_policy, 
        max_episode_steps=100, num_eps=1, 
        savevideo=True, 
        save_dir="results/lbf/videos/", save_name="ego-vs-ego-test")


