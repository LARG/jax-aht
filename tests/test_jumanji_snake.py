import jax
import jumanji
from jumanji import specs
import matplotlib.pyplot as plt


# Define action sampling function
def sample_action(action_spec, key):
    if isinstance(action_spec, specs.BoundedArray):
        low = action_spec.minimum
        high = action_spec.maximum
        action = jax.random.randint(key, shape=(), minval=low, maxval=high)
    else:
        action = action_spec.generate_value()
    return action

# Instantiate a Jumanji environment using the registry
env = jumanji.make('Snake-v1')
NUM_EPISODES = 2

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
        # action = env.action_spec.generate_value()          # Action selection (dummy value here)
        # action, _ = policy(observation, action_key)
        # action = action.squeeze(axis=0)
        action = sample_action(env.action_spec, action_key)
        print('action is ', action)
        state, timestep = step_fn(state, 
                                  action
                                  )
        # env.render(state)
        states.append(state)
    # Freeze the terminal frame to pause the GIF.
    for _ in range(3):
        states.append(state)
    
anim = env.animate(states, interval=150)
anim.save("figures/snake.gif", writer="imagemagick")
