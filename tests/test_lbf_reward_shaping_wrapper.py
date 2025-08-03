import jax
import jumanji
from jumanji.environments.routing.lbf.generator import RandomGenerator
from envs.log_wrapper import LogWrapper

from envs.lbf.lbf_wrapper import LBFWrapper
from envs.lbf.reward_shaping_lbf_wrapper import RewardShapingLBFWrapper

"""
The purpose of this file is to test the LBFWrapper wrapper for the LevelBasedForaging environment.
"""

agent_reward_shaping_params = {
    "agent_0": {
        "DISTANCE_TO_NEAREST_FOOD_REW": 1.5, # Reward for moving closer to food (H1)
        "DISTANCE_TO_FARTHEST_FOOD_REW": 0.0, # Reward for moving further from food (H2)
        # "SEQUENCE_REW": 0.0, # Reward for completing a sequence of actions (H3-H8)
        "FOLLOWING_TEAMMATE_REW": 0.0, # Reward for following another agent
        "CENTERED_FOOD_DISTANCE_REW": 0.0, # Reward for moving towards towards the food that is closest to the midpoint of the two agents
        "PROXIMITY_TO_TEAMMATE_REW": 0.1, # Reward for Proimity to teammate
        "COLLECT_FOOD_REW": 3.0},
    "agent_1": { 
        "DISTANCE_TO_NEAREST_FOOD_REW": 0.0, # Reward for moving closer to food (H1)
        "DISTANCE_TO_FARTHEST_FOOD_REW": 0.0, # Reward for moving further from food (H2)
        # "SEQUENCE_REW": 0.0, # Reward for completing a sequence of actions (H3-H8)
        "FOLLOWING_TEAMMATE_REW": 1.5, # Reward for following another agent
        "CENTERED_FOOD_DISTANCE_REW": 0.0, # Reward for moving towards towards the food that is closest to the midpoint of the two agents
        "PROXIMITY_TO_TEAMMATE_REW": 0.3, # Reward for Proimity to teammate
        "COLLECT_FOOD_REW": 3.0},
}

# Instantiate a Jumanji environment
env = jumanji.make('LevelBasedForaging-v0', 
                   generator=RandomGenerator(grid_size=7,
                                             fov=7,
                                             num_agents=2,
                                             num_food=3,
                                             force_coop=True,
                                            ),
                   time_limit=100, penalty=0.1)

wrapper = RewardShapingLBFWrapper(
    env,
    reward_shaping_params=agent_reward_shaping_params,
)
wrapper = LogWrapper(wrapper)

NUM_EPISODES = 4
key = jax.random.PRNGKey(20394)

# reset outside of for loop over episodes to test auto-reset behavior
key, subkey = jax.random.split(key)
obs, state = wrapper.reset(subkey)

for episode in range(NUM_EPISODES):
    done = {agent: False for agent in wrapper.agents}
    done['__all__'] = False
    total_rewards = {agent: 0.0 for agent in wrapper.agents}
    num_steps = 0
    while not done['__all__']:
        # Sample actions for each agent
        actions = {}
        for agent in wrapper.agents:
            action_space = wrapper.action_space(agent)
            key, action_key = jax.random.split(key)
            action = int(action_space.sample(action_key))
            actions[agent] = action
        
        # hardcoded actions
        # actions = {"agent_0": 1, "agent_1": 2}
        key, subkey = jax.random.split(key)
        obs, state, rewards, done, info = wrapper.step(subkey, state, actions)

        # Process observations, rewards, dones, and info as needed
        for agent in wrapper.agents:
            total_rewards[agent] += rewards[agent]

            print(f"\nEpisode {episode}, agent {agent}, timestep {wrapper.get_step_count(state.env_state)}")

            # print("action is ", actions[agent])
            # print("obs", obs[agent], "type", type(obs[agent]))
            # print("rewards", rewards[agent], "type", type(rewards[agent]))
            print("dones", done[agent], "type", type(done[agent]))
            print("avail actions are ", wrapper.get_avail_actions(state.env_state)[agent])

        print("info", info, "type", type(info))

        num_steps += 1

    print(f"Episode {episode} finished. Total rewards: {total_rewards}. Num steps: {num_steps}")
