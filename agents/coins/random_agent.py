import jax
import jax.numpy as jnp
from agents.coins.base_agent import BaseAgent, AgentState

class RandomAgent(BaseAgent):
    def __init__(self):
        super().__init__()

    def _get_action(self, obs: jnp.ndarray, env_state, agent_state: AgentState, rng: jax.random.PRNGKey):
        # Assume env_state has action_space.n or use a fixed number of actions (e.g., 5 for CoinGame)
        num_actions = 5  
        action = jax.random.randint(rng, (), 0, num_actions)
        return action, agent_state