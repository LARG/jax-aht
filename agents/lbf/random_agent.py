from typing import Tuple

import jax
import jax.numpy as jnp
from jumanji.environments.routing.lbf.types import State as LBFState

from agents.lbf.base_agent import BaseAgent, AgentState


class RandomAgent(BaseAgent):
    """A random agent that takes random actions."""
    
    def __init__(self):
        super().__init__()

    def _get_action(self, obs: jnp.ndarray, env_state: LBFState, 
                    agent_state: AgentState, rng: jax.random.PRNGKey) -> Tuple[int, AgentState]:
        """Return a random action and updated state.
        
        Args:
            obs: Flattened observation array (not used)
            agent_state: AgentState containing agent's internal state
            rng: jax.random.PRNGKey for any stochasticity. The rng key is not returned,
                so the user should split the key before passing it to an agent. 
            
        Returns:
            Tuple of (random_action, updated_agent_state)
        """
        # Generate random action (excluding Actions.done which is 6)
        action = jax.random.randint(rng, (), 0, 6)
        
        return action, agent_state
