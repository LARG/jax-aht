import jax
import jumanji
from jumanji import specs
from jumanji.environments.routing.lbf.generator import RandomGenerator


# Define action sampling function
def sample_action(action_spec, key):
    if isinstance(action_spec, specs.MultiDiscreteArray):
        low = action_spec.minimum
        high = action_spec.maximum
        action = jax.random.randint(key, shape=action_spec.shape, minval=low, maxval=high)
    else:
        action = action_spec.generate_value()
    return action

# Instantiate a Jumanji environment using the registry
env = jumanji.make('LevelBasedForaging-v0',
                   generator=RandomGenerator(grid_size=8,
                                             fov=8,
                                             num_agents=3,
                                             num_food=5,
                                             force_coop=False,
                                            ))

NUM_EPISODES = 10
RENDER = True
SAVEVIDEO = False

reset_fn = jax.jit(env.reset)
step_fn = jax.jit(env.step)
states = []
key = jax.random.PRNGKey(20394)

for episode in range(NUM_EPISODES):
    key, reset_key = jax.random.split(key)
    state, timestep = reset_fn(reset_key)
    states.append(state)
    while not timestep.last():
        key, action_key = jax.random.split(key)
        observation = jax.tree_util.tree_map(lambda x: x[None], timestep.observation)
        action = sample_action(env.action_spec, action_key)
        state, timestep = step_fn(state, 
                                  action
                                  )
        print("action is ", action)
        print('timestep is ', timestep) # reward is stored within Timestep object

        if RENDER:         
            env.render(state)
        states.append(state)
    
        # import sys; sys.exit(0)

    # Rendering: Freeze the terminal frame to pause the GIF.
    if RENDER:
        for _ in range(3):
            states.append(state)
    
if RENDER and SAVEVIDEO:
    anim = env.animate(states, interval=150)
    anim.save("figures/lbf.gif", writer="imagemagick")
