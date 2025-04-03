import jax
import jumanji
from jumanji import specs
from jumanji.wrappers import JumanjiToGymWrapper

'''
The purpose of this file is to test Jumanji's gym wrapper for the LevelBasedForaging environment.
'''

# Instantiate a Jumanji environment w/gym wrapper 
env = jumanji.make('LevelBasedForaging-v0')
env = JumanjiToGymWrapper(env, seed=103948)

NUM_EPISODES = 2

states = []
key = jax.random.PRNGKey(20394)

for episode in range(NUM_EPISODES):
    done = False
    state = env.reset()
    states.append(state)
    while not done:
        key, action_key = jax.random.split(key)
        action = env.action_space.sample()
        print('action is ', action, 'type is ', type(action))
        '''
        The gym wrapper does NOT work with LBF because LBF returns an array of rewards, 
        while the gym wrapper expects a single reward. We would need to write a wrapper 
        summing the rewards over all agents to get this to work. 
        '''
        state, rew, done, trunc, _ = env.step(action)
        # env.render(state)
        states.append(state)