import jax
from envs.log_wrapper import LogWrapper
from jaxmarl.environments.overcooked import overcooked_layouts

from envs.overcooked.overcooked_wrapper import OvercookedWrapper


"""
The purpose of this file is to test the OvercookedWrapper for the Overcooked environment.
"""

# Instantiate the Overcooked environment
env = OvercookedWrapper(
    layout=overcooked_layouts["cramped_room"],
    random_reset=True,
    max_steps=400,
    )
wrapper = LogWrapper(env)

NUM_EPISODES = 2
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
        
        key, subkey = jax.random.split(key)
        obs, state, rewards, done, info = wrapper.step(subkey, state, actions)

        # Process observations, rewards, dones, and info as needed
        for agent in wrapper.agents:
            total_rewards[agent] += rewards[agent]

            print(f"\nEpisode {episode}, agent {agent}, timestep {wrapper.get_step_count(state.env_state)}")

            print("obs shape is ", obs[agent].shape, "type", type(obs[agent]))
            print("action is ", actions[agent])
            print("rewards", rewards[agent], "type", type(rewards[agent]))
            print("info", info, "type", type(info))
            print("avail actions are ", wrapper.get_avail_actions(state.env_state)[agent])
            print("dones", done[agent], "type", type(done[agent]))

        num_steps += 1

    print(f"Episode {episode} finished. Total rewards: {total_rewards}. Num steps: {num_steps}")
