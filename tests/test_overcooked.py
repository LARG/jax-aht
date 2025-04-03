"""
Short introduction to running the Overcooked environment and visualising it using random actions.
"""
import jax 
from envs.overcooked.overcooked_wrapper import OvercookedWrapper
from envs.overcooked.overcooked_visualizer_v2 import OvercookedVisualizerV2
from jaxmarl.environments.overcooked import overcooked_layouts, layout_grid_to_dict


# Parameters + random keys
episodes = 1
max_steps = 50
key = jax.random.PRNGKey(0)
key, key_r, key_a = jax.random.split(key, 3)

# Get one of the classic layouts (cramped_room, asymm_advantages, coord_ring, forced_coord, counter_circuit)
map_name = "cramped_room"
layout = overcooked_layouts[map_name]

# Or make your own!
# custom_layout_grid = """
# WWOWW
# WA  W
# B P X
# W  AW
# WWOWW
# """
# layout = layout_grid_to_dict(custom_layout_grid)

# Instantiate environment
# env = jaxmarl.make('overcooked', layout=layout, max_steps=max_steps)
env = OvercookedWrapper(
        layout=layout,
        random_reset=True,
        max_steps=max_steps,
    )

obs, state = env.reset(key_r)
print('list of agents in environment', env.agents)

# Sample random actions
key_a = jax.random.split(key_a, env.num_agents)
actions = {agent: env.action_space(agent).sample(key_a[i]) for i, agent in enumerate(env.agents)}
print('example action dict', actions)

for _ in range(episodes):
    state_seq = []
    for ts in range(max_steps):
        state_seq.append(state)
        # Iterate random keys and sample actions
        key, key_s, key_a = jax.random.split(key, 3)
        key_a = jax.random.split(key_a, env.num_agents)

        actions = {agent: env.action_space(agent).sample(key_a[i]) for i, agent in enumerate(env.agents)}
        
        # Step environment
        obs, state, rewards, dones, infos = env.step(key_s, state, actions)
        print("\n timestep", ts)
        # print("obs", obs["agent_0"], "type", type(obs["agent_0"]))
        # print("rewards", rewards["agent_0"], "type", type(rewards["agent_0"]))
        # print("dones", dones["agent_0"], "type", type(dones["agent_0"]))
        # print("infos", infos, "type", type(infos))

# Visualization code: 
viz = OvercookedVisualizerV2()
print("Saving MP4 video...")
viz.animate_mp4(state_seq, env.agent_view_size, 
                filename="results/overcooked/mp4/test.mp4", 
                pixels_per_tile=32, fps=5)
print("MP4 video saved!")

# # # Render to screen
# for s in state_seq:
#     viz.render(env.agent_view_size, s, highlight=False)
#     time.sleep(0.25)

