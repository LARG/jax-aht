from functools import partial
from typing import Tuple, Dict, Any

import jax
import jax.numpy as jnp

from agents.overcooked.base_agent import BaseAgent, AgentState


class RandomAgent(BaseAgent):
    """A random agent that takes random actions."""
    
    def __init__(self, layout: Dict[str, Any]):
        super().__init__(layout)

    @partial(jax.jit, static_argnums=(0,))
    def _get_action(self, obs: jnp.ndarray, agent_state: AgentState) -> Tuple[int, AgentState]:
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
        action = jax.random.randint(subkey, (), 0, 6)  # Random integer between 0 and 5
        
        # Create new state with updated key
        updated_agent_state = AgentState(
            agent_id=agent_state.agent_id,
            holding=agent_state.holding,
            goal=agent_state.goal,
            nonfull_pots=agent_state.nonfull_pots,
            soup_ready=agent_state.soup_ready,
            rng_key=rng_key
        )
        
        return action, updated_agent_state
