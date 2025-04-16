import jax
import jumanji
from jumanji.environments.routing.lbf.generator import RandomGenerator
from envs.lbf.adhoc_lbf_viewer import AdHocLBFViewer
from envs.log_wrapper import LogWrapper

from envs.jumanji_jaxmarl_wrapper import JumanjiToJaxMARL

"""
The purpose of this file is to test the JumanjiToJaxMARL wrapper for the LevelBasedForaging environment.
"""
# Instantiate custom viewer
grid_size = 8
viewer = AdHocLBFViewer(grid_size=grid_size, highlight_agent_idx=1)

# Instantiate a Jumanji environment
env = jumanji.make('LevelBasedForaging-v0', 
                    generator=RandomGenerator(grid_size=grid_size,
                                         fov=grid_size,
                                         num_agents=3,
                                         num_food=3,
                                         force_coop=True,
                                         ),
                    time_limit=100, penalty=0.1,
                    viewer=viewer)
                    
wrapper = JumanjiToJaxMARL(env)
wrapper = LogWrapper(wrapper)

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
        env.render(state.env_state.env_state)
        num_steps += 1

    print(f"Episode {episode} finished. Total rewards: {total_rewards}. Num steps: {num_steps}")
