from functools import partial
from typing import Tuple, Dict, Any

import jax
import jax.numpy as jnp
from jax import lax
from jaxmarl.environments.overcooked.overcooked import Actions

from .base_agent import BaseAgent, AgentState, Holding, Goal

class OnionAgent(BaseAgent):
    """A heuristic agent for the Overcooked environment that gets onions 
    and places them on the counter with probability p_onion_on_counter 
    or in the pot with probability 1 - p_onion_on_counter.

    Currently, only the "nearest" preference works. 
    """
    
    def __init__(self, layout: Dict[str, Any], p_onion_on_counter: float = 0.1, pref: str="nearest"):
        super().__init__(layout)
        self.p_onion_on_counter = p_onion_on_counter
        self.pref = pref
        
    @partial(jax.jit, static_argnums=(0,))
    def _get_action(self, obs: jnp.ndarray, agent_state: AgentState) -> Tuple[int, AgentState]:
        """Get action based on observation and current agentstate.
        
        Args:
            obs: Flattened observation array
            
        Returns:
            Tuple of (action, updated_agent_state)
        """
        # Reshape flattened observation back to 3D
        obs_3d = jnp.reshape(obs, self.obs_shape)
        
        # Define helper functions for each state
        def get_onion(carry):
            '''Go to the nearest onion and pick it up. '''
            obs_3d, rng_key = carry
            action, new_rng_key = self._go_to_obj_and_interact(obs_3d, "onion", self.pref, rng_key)
            return (action, new_rng_key)
            
        def put_onion_in_pot(carry):
            '''Go to the nearest pot and put the onion in it. '''
            obs_3d, rng_key = carry
            action, new_rng_key = self._go_to_obj_and_interact(obs_3d, "nonfull_pot", self.pref, rng_key)
            return (action, new_rng_key)
            
        def put_onion_on_counter(carry):
            '''Go to the nearest free counter and put the onion on it. '''
            obs_3d, rng_key = carry
            action, new_rng_key = self._go_to_obj_and_interact(obs_3d, "counter", self.pref, rng_key)
            return (action, new_rng_key)
        
        # Get action and update RNG key based on current state
        def handle_holding_onion(carry):
            obs_3d, rng_key = carry
            # Generate random number to determine if we should put onion on counter
            rng_key, subkey = jax.random.split(rng_key)
            should_put_on_counter = jax.random.uniform(subkey) < self.p_onion_on_counter
            
            return lax.cond(
                should_put_on_counter,
                put_onion_on_counter,
                put_onion_in_pot,
                (obs_3d, rng_key)
            )
        
        action, rng_key = lax.cond(
            agent_state.holding == Holding.nothing,
            get_onion,
            lambda carry: lax.cond(
                agent_state.holding == Holding.onion,
                handle_holding_onion,
                lambda _: (Actions.stay, carry[1]),
                carry
            ),
            (obs_3d, agent_state.rng_key)
        )
        
        # Update goal based on what the agent is holding
        new_goal = lax.cond(
            agent_state.holding == Holding.nothing,
            lambda _: Goal.get_onion,
            lambda _: lax.cond(
                agent_state.holding == Holding.onion,
                lambda _: Goal.put_onion,
                lambda _: agent_state.goal,
                None
            ),
            None
        )

        # Create new state with updated key and goal, preserving other state values
        updated_agent_state = AgentState(
            agent_id=agent_state.agent_id,
            holding=agent_state.holding,
            goal=new_goal,
            nonfull_pots=agent_state.nonfull_pots,
            soup_ready=agent_state.soup_ready,
            rng_key=rng_key
        )

        return action, updated_agent_state