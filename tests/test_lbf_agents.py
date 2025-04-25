import os
import numpy as np
from typing import Dict, Tuple

import jax
from envs import make_env
from agents.lbf import RandomAgent, SequentialFruitAgent
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
    agent0_state = agent0.init_agent_state(0)
    agent1_state = agent1.init_agent_state(1)
    
    # Initialize state sequence
    state_seq = []    
    while not done['__all__']:
        # Get actions from both agents with their states
        key, act0_rng, act1_rng = jax.random.split(key, 3)

        print(f"Step {num_steps}")
        action0, agent0_state = agent0.get_action(obs["agent_0"], state, agent0_state, act0_rng)
        action1, agent1_state = agent1.get_action(obs["agent_1"], state, agent1_state, act1_rng)
        
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
            agent1_name = agent1.get_name()
            # print(f"Agent 0 {(agent0_name)} state: {agent0_state}")
            print(f"Agent 1 {(agent1_name)} state: {agent1_state}")
            print("Actions:", actions)
    
    print(f"Episode finished. Total states collected: {len(state_seq)}")
    return total_rewards, num_steps, state_seq

def main(num_episodes, 
         max_steps=100,
         visualize=False, 
         save_video=False):
    # Initialize environment
    print("Initializing environment...")
    # directly initialize the env
    # use the make_env function to initialize the env
    env = make_env(env_name="lbf", env_kwargs={"time_limit": max_steps})
    print("Environment initialized")
    
    # Initialize agents
    print("Initializing agents...")
    # choices: lexicographic, reverse_lexicographic, column_major, reverse_column_major, nearest_agent, farthest_agent
    agent0 = SequentialFruitAgent(grid_size=7, num_fruits=3, ordering_strategy='lexicographic') # boxed
    agent1 = SequentialFruitAgent(grid_size=7, num_fruits=3, ordering_strategy='lexicographic') # not boxed
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
        episode_return = np.mean(list(total_rewards.values()))
        returns.append(episode_return)
        
        print(f"\nEpisode {episode + 1} finished:")
        print(f"Total steps: {num_steps}")
        print(f"Mean episode return: {episode_return:.2f}")
        print("Episode returns per agent:")
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
        for state in state_seq_all:
            env.render(state)
            time.sleep(.1)
    if save_video:
        savedir = "results/lbf/videos"
        if not os.path.exists(savedir):
            os.makedirs(savedir, exist_ok=True)

        anim = env.animate(state_seq_all, interval=150)
        print(f"\nSaving mp4 with {len(state_seq_all)} frames...")
        savepath = os.path.join(savedir, f"{agent0.get_name()}_vs_{agent1.get_name()}.mp4")
        anim.save(savepath, writer="ffmpeg")
        print(f"MP4 saved successfully at {savepath}")

if __name__ == "__main__":
    DEBUG = False
    VISUALIZE = False
    SAVE_VIDEO = not VISUALIZE    
    NUM_EPISODES = 5


    with jax.disable_jit(DEBUG):
        main(num_episodes=NUM_EPISODES, 
             max_steps=30,
             visualize=VISUALIZE, 
             save_video=SAVE_VIDEO) 