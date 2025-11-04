import numpy as np
from typing import Dict, Tuple

import jax
from envs.overcooked.adhoc_overcooked_visualizer import AdHocOvercookedVisualizer
from envs.jaxmarl_wrapper import JaxMARLWrapper
from envs.overcooked.overcooked_v1 import OvercookedV1
from envs.overcooked.augmented_layouts import augmented_layouts
from envs import make_env
from agents.overcooked import OnionAgent, PlateAgent, IndependentAgent, StaticAgent, RandomAgent
import time

def run_episode(env, agent0, agent1, key) -> Tuple[Dict[str, float], int]:
    """Run a single episode with two heuristic agents.
    
    Returns:
        Tuple containing:
        - Total rewards for each agent
        - Number of steps taken
    """
    # Reset environment
    print("Resetting environment...")
    key, subkey = jax.random.split(key)
    obs, state = env.reset(subkey)
    print("Environment reset complete.")
    
    # Initialize episode tracking
    done = {agent: False for agent in env.agents}
    done['__all__'] = False
    total_shaped_rewards = {agent: 0.0 for agent in env.agents}
    total_base_rewards = {agent: 0.0 for agent in env.agents}
    num_steps = 0
    
    # Initialize agent states
    agent0_state = agent0.init_agent_state(0)
    agent1_state = agent1.init_agent_state(1)

    agent0_name = agent0.get_name()
    agent1_name = agent1.get_name()

    # Initialize state sequence
    state_seq = []    
    while not done['__all__']:
        # Get actions from both agents with their states
        print(f"Step {num_steps}")
        action0, agent0_state = agent0.get_action(obs["agent_0"], state, agent0_state)
        action1, agent1_state = agent1.get_action(obs["agent_1"], state, agent1_state)
        
        actions = {"agent_0": action0, "agent_1": action1}
        
        # Step environment
        key, subkey = jax.random.split(key)
        obs, state, shaped_rewards, done, info = env.step(subkey, state, actions)
        base_return_so_far = info['base_return']
        base_reward = info['base_reward']

        # Add state to sequence and print debug info
        state_seq.append(state)
        
        # Update rewards
        for agent in env.agents:
            total_shaped_rewards[agent] += shaped_rewards[agent]
        num_steps += 1
                
        # Print progress every 10 steps
        if num_steps % 1 == 0:
            # print(f"Agent 0 {(agent0_name)} state: {agent0_state}")
            print(f"Agent 1 {(agent1_name)} state: {agent1_state}")
            print("Actions:", actions)
            # print("Base reward: ", base_reward)
            # print("Base return: ", base_return_so_far)

    total_base_rewards = {agent: info['base_return'][i] for i, agent in enumerate(env.agents)}

    print(f"Episode finished. Total states collected: {len(state_seq)}")
    return total_shaped_rewards, total_base_rewards, num_steps, state_seq

def main(num_episodes, 
         layout_name,
         random_reset=True,
         random_obj_state=True,
         do_reward_shaping=False,
         reward_shaping_params={},
         max_steps=100,
         visualize=False, 
         save_video=False):
    # Initialize environment
    print("Initializing environment...")
    layout = augmented_layouts[layout_name]
    # Initialize the environment via factory to keep consistent instantiation
    env = make_env(env_name="overcooked-v1", env_kwargs={
        "layout": layout_name,
        "random_reset": random_reset,
        "random_obj_state": random_obj_state,
        "max_steps": max_steps,
        "do_reward_shaping": do_reward_shaping,
        "reward_shaping_params": reward_shaping_params,
    })
    print("Environment initialized")
    
    # Initialize agents
    print("Initializing agents...")
    # agent0 = PlateAgent(layout=layout, p_plate_on_counter=0.) # red
    # agent1 = OnionAgent(layout=layout, p_onion_on_counter=0.) # blue
    agent0 = IndependentAgent(layout=layout, p_onion_on_counter=0., p_plate_on_counter=0.) # red
    # agent1 = IndependentAgent(layout=layout, p_onion_on_counter=0., p_plate_on_counter=0.) # blue

    agent1 = StaticAgent(layout=layout) # blue
    print("Agents initialized")
    
    print("Agent 0:", agent0.get_name())
    print("Agent 1:", agent1.get_name())
    
    # Run multiple episodes
    key = jax.random.PRNGKey(0)
    
    # Initialize returns lists for each agent
    base_returns_agent0 = []
    base_returns_agent1 = []
    
    state_seq_all = []
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        key, subkey = jax.random.split(key)
        total_shaped_rewards, total_base_rewards, num_steps, ep_states = run_episode(env, agent0, agent1, subkey)
        state_seq_all.extend(ep_states)
        print(f"Total states in sequence after episode: {len(state_seq_all)}")
        
        # Track returns for each agent separately
        episode_base_return_agent0 = total_base_rewards["agent_0"]
        episode_base_return_agent1 = total_base_rewards["agent_1"]
        base_returns_agent0.append(episode_base_return_agent0)
        base_returns_agent1.append(episode_base_return_agent1)
        
        print(f"\nEpisode {episode + 1} finished:")
        print(f"Total steps: {num_steps}")
        print(f"Agent 0 base return: {episode_base_return_agent0:.2f}")
        print(f"Agent 1 base return: {episode_base_return_agent1:.2f}")
    
    # Print statistics for each agent
    mean_return_agent0 = np.mean(base_returns_agent0)
    std_return_agent0 = np.std(base_returns_agent0)
    mean_return_agent1 = np.mean(base_returns_agent1)
    std_return_agent1 = np.std(base_returns_agent1)
    
    print(f"\nStatistics across {num_episodes} episodes:")
    print(f"Agent 0 - Mean return: {mean_return_agent0:.2f} ± {std_return_agent0:.2f}")
    print(f"Agent 1 - Mean return: {mean_return_agent1:.2f} ± {std_return_agent1:.2f}")
    print(f"Total mean return: {(mean_return_agent0 + mean_return_agent1):.2f} ± {np.sqrt(std_return_agent0**2 + std_return_agent1**2):.2f}")

    # Visualize state sequences
    if visualize:
        print("Visualizing state sequences...")
        viz = AdHocOvercookedVisualizer()
        for state in state_seq_all:
            viz.render(env.agent_view_size, state.env_state, highlight_agent_idx=0)
            time.sleep(.1)
    if save_video:
        print(f"\nSaving mp4 with {len(state_seq_all)} frames...")
        viz = AdHocOvercookedVisualizer()
        viz.animate_mp4([s.env_state for s in state_seq_all], env.agent_view_size, 
            highlight_agent_idx=0,
            filename=f'results/overcooked-v1/videos/{layout_name}/{agent0.get_name()}_vs_{agent1.get_name()}.mp4', 
            pixels_per_tile=32, fps=25)
        print("MP4 saved successfully!")

if __name__ == "__main__":
    DEBUG = False
    VISUALIZE = False
    SAVE_VIDEO = not VISUALIZE    
    NUM_EPISODES = 1

    layout_names = [
        # "cramped_room", 
        "asymm_advantages", 
        # "coord_ring", 
        # "counter_circuit", 
        # "forced_coord"
                    ]


    for layout_name in layout_names:
        with jax.disable_jit(DEBUG):
            main(num_episodes=NUM_EPISODES, 
                layout_name=layout_name,
                random_reset=True,
                random_obj_state=False,
                do_reward_shaping=True,
                reward_shaping_params={},
                max_steps=100,
                visualize=VISUALIZE, 
                save_video=SAVE_VIDEO) 