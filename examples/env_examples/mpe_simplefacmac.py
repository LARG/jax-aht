import jax
from envs import make_env

'''
JaxMARL offers all of the PettingZoo environments in the MPE suite.
Of those, there are 3 fully cooperative variants of SimpleTag:

MPE_simple_facmac_3a_v1
MPE_simple_facmac_6a_v1
MPE_simple_facmac_9a_v1

In these environments, there are 3, 6, and 9 adversaries, respectively, and 
1 prey (good agent). The prey is controlled by a heuristic AI. 
For our purposes, we need to create a new environment that has 2 adversaries and 1 prey.
'''

env_id = "MPE_simple_facmac_3a_v1"
env = make_env(env_id)

key = jax.random.PRNGKey(0)
key, key_r, key_a = jax.random.split(key, 3)
obs, state = env.reset(key_r)

print(obs)
print(state)

print("ENV AGENTS ARE: ", env.agents)
actions = {agent: env.action_space(agent).sample(key_a) for agent in env.agents}
print("ACTIONS ARE: ", actions)

key, key_s = jax.random.split(key)
obs, state, rewards, dones, infos = env.step(key_s, state, actions)
