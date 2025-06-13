import jax
from jax import numpy as jnp
from jaxmarl import make
from jaxmarl.wrappers.baselines import load_params
from obl_r2d2_agent import OBLAgentR2D2

weight_file = "./obl-r2d2-flax/icml_OBL1/OFF_BELIEF1_SHUFFLE_COLOR0_BZA0_BELIEF_a.safetensors"
params = load_params(weight_file)

agent = OBLAgentR2D2()
agent_carry = agent.initialize_carry(jax.random.PRNGKey(0), batch_dims=(2,))

rng = jax.random.PRNGKey(0)
env = make('hanabi')
obs, env_state = env.reset(rng)
env.render(env_state)

batchify = lambda x: jnp.stack([x[agent] for agent in env.agents])
unbatchify = lambda x: {agent:x[i] for i, agent in enumerate(env.agents)}

agent_input = (
    batchify(obs),
    batchify(env.get_legal_moves(env_state))
)
agent_carry, actions = agent.greedy_act(params, agent_carry, agent_input)
actions = unbatchify(actions)

obs, env_state, rewards, done, info = env.step(rng, env_state, actions)

print('actions:', {agent:env.action_encoding[int(a)] for agent, a in actions.items()})
env.render(env_state)
