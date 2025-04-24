from typing import Tuple

import jax
import jax.numpy as jnp
from jumanji.environments.routing.lbf.types import State as LBFState

from agents.lbf.base_agent import BaseAgent, AgentState


class RandomAgent(BaseAgent):
    """A random agent that takes random actions."""
    
    def __init__(self, agent_id: int):
        super().__init__(agent_id)

    def _get_action(self, obs: jnp.ndarray, env_state: LBFState, agent_state: AgentState) -> Tuple[int, AgentState]:
        """Return a random action and updated state.
        
        Args:
            obs: Flattened observation array (not used)
            agent_state: AgentState containing agent's internal state
            
        Returns:
            Tuple of (random_action, updated_agent_state)
        """
        # Split key for this step
        rng_key, subkey = jax.random.split(agent_state.rng_key)
        
        # Generate random action (excluding Actions.done which is 6)
        action = jax.random.randint(subkey, (), 0, 6)
        
        # Create new state with updated key
        updated_agent_state = AgentState(
            rng_key=rng_key
        )
        
        return action, updated_agent_state
