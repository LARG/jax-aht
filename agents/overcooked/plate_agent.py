from functools import partial
from typing import Tuple, Dict, Any

import jax
import jax.numpy as jnp
from jax import lax
from jaxmarl.environments.overcooked.overcooked import Actions

from .base_agent import BaseAgent, AgentState, Holding, Goal

class PlateAgent(BaseAgent):
    """A heuristic agent for the Overcooked environment that plates dishes
    when ready and delivers. With probability p_plate_on_counter, it will
    place plates on counters instead of plating soup.
    """
    
    def __init__(self, agent_name: str, layout: Dict[str, Any], p_plate_on_counter: float = 0.1):
        super().__init__(agent_name, layout)
        self.p_plate_on_counter = p_plate_on_counter
        
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
        def get_plate(carry):
            '''Go to the nearest plate and pick it up. '''
            obs_3d, rng_key = carry
            action, new_rng_key = self._go_to_obj_and_interact(obs_3d, "plate", rng_key)
            return (action, Goal.get_plate, new_rng_key)
            
        def put_plate_on_counter(carry):
            '''Go to the nearest free counter and put the plate on it. '''
            obs_3d, rng_key = carry
            action, new_rng_key = self._go_to_obj_and_interact(obs_3d, "counter", rng_key)
            return (action, Goal.put_plate, new_rng_key)
            
        def plate_soup(carry):
            '''Go to the nearest pot with ready soup and plate it. '''
            obs_3d, rng_key = carry
            action, new_rng_key = self._go_to_obj_and_interact(obs_3d, "ready_pot", rng_key)
            return (action, Goal.get_soup, new_rng_key)
            
        def deliver_dish(carry):
            '''Go to the delivery window and deliver the dish. '''
            obs_3d, rng_key = carry
            action, new_rng_key = self._go_to_obj_and_interact(obs_3d, "delivery", rng_key)
            return (action, Goal.deliver, new_rng_key)
        
        # Get action and update RNG key based on current state
        def handle_holding_plate(carry):
            obs_3d, rng_key = carry
            # Generate random number to determine if we should put plate on counter
            rng_key, subkey = jax.random.split(rng_key)
            should_put_on_counter = jax.random.uniform(subkey) < self.p_plate_on_counter
            
            # If we should put on counter, do that, otherwise check if soup is ready
            return lax.cond(
                should_put_on_counter,
                put_plate_on_counter,
                lambda carry: lax.cond(
                    agent_state.soup_ready,
                    plate_soup,
                    lambda _: (Actions.stay, Goal.get_soup, carry[1]),
                    carry
                ),
                (obs_3d, rng_key)
            )
        
        # Main action selection and goal update logic
        action, new_goal, rng_key = lax.cond(
            agent_state.holding == Holding.nothing,
            get_plate,
            lambda carry: lax.cond(
                agent_state.holding == Holding.plate,
                handle_holding_plate,
                lambda carry: lax.cond(
                    agent_state.holding == Holding.dish,
                    deliver_dish,
                    lambda _: (Actions.stay, agent_state.goal, carry[1]),
                    carry
                ),
                carry
            ),
            (obs_3d, agent_state.rng_key)
        )
        
        # Create new state with updated key and goal, preserving other state values
        updated_agent_state = AgentState(
            holding=agent_state.holding,
            goal=new_goal,
            nonfull_pots=agent_state.nonfull_pots,
            soup_ready=agent_state.soup_ready,
            rng_key=rng_key
        )

        return action, updated_agent_state