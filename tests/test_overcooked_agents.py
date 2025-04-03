import numpy as np
from typing import Dict, Tuple

import jax
from envs.overcooked.overcooked_wrapper import OvercookedWrapper
from envs.overcooked.overcooked_visualizer_v2 import OvercookedVisualizerV2
from envs.overcooked.augmented_layouts import augmented_layouts
from agents.overcooked import OnionAgent, StaticAgent, RandomAgent
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
    total_rewards = {agent: 0.0 for agent in env.agents}
    num_steps = 0
    
    # Initialize agent states
    agent0_state = agent0.initial_state
    agent1_state = agent1.initial_state
    
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
        obs, state, rewards, done, info = env.step(subkey, state, actions)
        
        # Add state to sequence and print debug info
        state_seq.append(state)
        
        # Update rewards
        for agent in env.agents:
            total_rewards[agent] += rewards[agent]
            
        num_steps += 1
                
        # Print progress every 10 steps
        if num_steps % 10 == 0:
            agent0_name = agent0.get_name()
            # agent1_name = agent1.get_name()
            print(f"Agent 0 {(agent0_name)} state: {agent0_state}")
            # print(f"Agent 1 {(agent1_name)} state: {agent1_state}")
            print("Actions:", actions)
    
    print(f"Episode finished. Total states collected: {len(state_seq)}")
    return total_rewards, num_steps, state_seq

def main(num_episodes, 
         layout_name,
         random_reset=True,
         random_obj_state=True,
         max_steps=100,
         visualize=False, 
         save_gif=False):
    # Initialize environment
    print("Initializing environment...")
    layout = augmented_layouts[layout_name]
    env = OvercookedWrapper(
        layout=layout,
        random_reset=random_reset,
        random_obj_state=random_obj_state,
        max_steps=max_steps,
    )
    print("Environment initialized")
    
    # Initialize agents
    print("Initializing agents...")
    agent0 = OnionAgent("agent_0", layout=layout) # red
    agent1 = StaticAgent("agent_1", layout=layout) # blue
    print("Agents initialized")
    
    print("Agent 0:", agent0.get_name())
    print("Agent 1:", agent1.get_name())
    
    # Run multiple episodes
    key = jax.random.PRNGKey(0)
    
    # Initialize returns list
    returns = []
    
    state_seq_all = []
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        key, subkey = jax.random.split(key)
        total_rewards, num_steps, ep_states = run_episode(env, agent0, agent1, subkey)
        state_seq_all.extend(ep_states)  # Changed from += to extend for better list handling
        print(f"Total states in sequence after episode: {len(state_seq_all)}")
        
        # Calculate episode return
        episode_return = sum(total_rewards.values())
        returns.append(episode_return)
        
        print(f"\nEpisode {episode + 1} finished:")
        print(f"Total steps: {num_steps}")
        print(f"Episode return: {episode_return:.2f}")
        print("Final rewards:")
        for agent in env.agents:
            print(f" {agent}: {total_rewards[agent]:.2f}")
    
    # Print statistics
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    print(f"\nStatistics across {num_episodes} episodes:")
    print(f"Mean return: {mean_return:.2f} ± {std_return:.2f}")

    # Visualize state sequences
    if visualize:
        print("Visualizing state sequences...")
        viz = OvercookedVisualizerV2()
        for state in state_seq_all:
            viz.render(env.agent_view_size, state, highlight=False)
            time.sleep(.1)
    if save_gif:
        print(f"\nSaving mp4 with {len(state_seq_all)} frames...")
        viz = OvercookedVisualizerV2()
        viz.animate_mp4(state_seq_all, env.agent_view_size, 
            filename=f'results/overcooked/mp4/{layout_name}_{agent0.get_name()}_vs_{agent1.get_name()}.mp4', 
            pixels_per_tile=32, fps=25)
        print("MP4 saved successfully!")

if __name__ == "__main__":
    DEBUG = False
    VISUALIZE = False
    SAVE_GIF = not VISUALIZE    
    NUM_EPISODES = 2

    layout_names = [
        # "asymm_advantages", "coord_ring", 
        # "counter_circuit", "cramped_room", 
        "forced_coord"
                    ]

    # TODO: remove debug print statements
    for layout_name in layout_names:
        with jax.disable_jit(DEBUG):
            main(num_episodes=NUM_EPISODES, 
                layout_name=layout_name,
                random_reset=True,
                random_obj_state=False,
                max_steps=30,
                visualize=VISUALIZE, 
                save_gif=SAVE_GIF) 