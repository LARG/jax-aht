import jax
from envs import make_env

env_id = "hanabi"
env = make_env(env_id)

key = jax.random.PRNGKey(0)
key, key_r, key_a = jax.random.split(key, 3)

obs, state = env.reset(key_r)

print("AGENTS are: ", env.agents)