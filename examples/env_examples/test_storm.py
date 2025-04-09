import jax
import jax.numpy as jnp
from PIL import Image
from jaxmarl import make

# load environment
num_agents = 2
rng = jax.random.PRNGKey(18)
env = make('storm', 
        num_inner_steps=512, 
        num_outer_steps=1, 
        num_agents=num_agents, 
        fixed_coin_location=True,
        payoff_matrix=jnp.array([[[3, 0], [5, 1]], [[3, 5], [0, 1]]]),
        freeze_penalty=5,)
rng, _rng = jax.random.split(rng)
obs, old_state = env.reset(_rng)


# render each timestep
pics = []
for t in range(20):
    rng, *rngs = jax.random.split(rng, num_agents+1)
    actions = [jax.random.choice(
        rngs[a], a=env.action_space(0).n, p=jnp.array([0.1, 0.1, 0.5, 0.1, 0.2])
    ) for a in range(num_agents)]

    obs, state, reward, done, info = env.step_env(
        rng, old_state, [a for a in actions]
    )

    img = env.render(state)
    pics.append(img)

    old_state = state

# create and save gif
pics = [Image.fromarray(img) for img in pics]        
pics[0].save(
    "results/storm/storm.gif",
    format="GIF",
    save_all=True,
    optimize=False,
    append_images=pics[1:],
    duration=1000,
    loop=0,
)