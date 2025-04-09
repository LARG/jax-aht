from jaxmarl.environments.jaxnav.jaxnav_env import JaxNav
from jaxmarl.environments.jaxnav.jaxnav_viz import JaxNavVisualizer
import jax 

env = JaxNav(num_agents=2, act_type="Discrete")

rng = jax.random.PRNGKey(0)
rng, _rng = jax.random.split(rng)

obs, env_state = env.reset(_rng)

obs_list = [obs]
env_state_list = [env_state]

for _ in range(10):
    rng, act_rng, step_rng = jax.random.split(rng, 3)
    act_rngs = jax.random.split(act_rng, env.num_agents)
    actions = {a: env.action_space(a).sample(act_rngs[i]) for i, a in enumerate(env.action_spaces.keys())}
    obs, env_state, _, _, _ = env.step(step_rng, env_state, actions)
    obs_list.append(obs)
    env_state_list.append(env_state)

viz = JaxNavVisualizer(env, obs_list, env_state_list)
viz.animate("results/jaxnav/jaxnav.gif", view=True)