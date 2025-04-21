from functools import partial
from typing import Tuple, Dict, Any

import jax
import jax.numpy as jnp
from jax import lax
from jaxmarl.environments.overcooked.overcooked import Actions

from .base_agent import BaseAgent, AgentState, Holding, Goal

class IndependentAgent(BaseAgent):
    """A heuristic agent for the Overcooked environment that performs the entire pipeline of: 
    (1) Getting an onion and placing the onion in the soup pot until it has 3 onions
    (2) Getting a plate. 
    (3) Plating the soup when ready.
    (4) Delivering the soup.

    For the steps corresponding to getting an onion and getting a plate, the agent 
    has a probability p_onion_on_counter to place the onion on the counter instead of 
    in the soup pot, and p_plate_on_counter to place the plate on the counter instead of 
    delivering it. 

    This agent is suitable for: 
    - Asymmetric advantages
    - Cramped Room
    - Counter Circuit
    - Coord Ring

    It will fail on forced coordination layouts because it attempts to perform all 
    steps without coordination.
    """
    
    def __init__(self, agent_id: int, layout: Dict[str, Any], p_onion_on_counter: float = 0, p_plate_on_counter: float = 0.5):
        super().__init__(agent_id, layout)
        self.p_onion_on_counter = p_onion_on_counter
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
        def get_onion(carry):
            '''Go to the nearest onion and pick it up.'''
            obs_3d, agent_state = carry
            action, new_rng_key = self._go_to_obj_and_interact(obs_3d, "onion", agent_state.rng_key)
            return (action, Goal.get_onion, new_rng_key)
            
        def put_onion_in_pot(carry):
            '''Go to the nearest non-full pot and put the onion in it.'''
            obs_3d, agent_state = carry
            action, new_rng_key = self._go_to_obj_and_interact(obs_3d, "nonfull_pot", agent_state.rng_key)
            return (action, Goal.put_onion, new_rng_key)
            
        def put_onion_on_counter(carry):
            '''Go to the nearest free counter and put the onion on it.'''
            obs_3d, agent_state = carry
            action, new_rng_key = self._go_to_obj_and_interact(obs_3d, "counter", agent_state.rng_key)
            return (action, Goal.put_onion, new_rng_key)
            
        def get_plate(carry):
            '''Go to the nearest plate and pick it up.'''
            obs_3d, agent_state = carry
            action, new_rng_key = self._go_to_obj_and_interact(obs_3d, "plate", agent_state.rng_key)
            return (action, Goal.get_plate, new_rng_key)
            
        def put_plate_on_counter(carry):
            '''Go to the nearest free counter and put the plate on it.'''
            obs_3d, agent_state = carry
            action, new_rng_key = self._go_to_obj_and_interact(obs_3d, "counter", agent_state.rng_key)
            return (action, Goal.put_plate, new_rng_key)
            
        def plate_soup(carry):
            '''Go to the nearest pot with ready soup and plate it.'''
            obs_3d, agent_state = carry
            action, new_rng_key = self._go_to_obj_and_interact(obs_3d, "ready_pot", agent_state.rng_key)
            return (action, Goal.get_soup, new_rng_key)
            
        def deliver_dish(carry):
            '''Go to the delivery window and deliver the dish.'''
            obs_3d, agent_state = carry
            action, new_rng_key = self._go_to_obj_and_interact(obs_3d, "delivery", agent_state.rng_key)
            return (action, Goal.deliver, new_rng_key)
        
        # Handle onion-related actions
        def handle_holding_onion(carry):
            obs_3d, agent_state = carry
            
            # If no non-full pots available, always put onion on counter
            has_nonfull_pot = jnp.any(agent_state.nonfull_pots)
            
            # If there is a non-full pot available, randomly decide whether to put onion on counter or in pot
            # Otherwise, always put onion on counter
            rng_key, subkey = jax.random.split(agent_state.rng_key)
            should_put_on_counter = lax.cond(
                has_nonfull_pot,
                lambda _: jax.random.uniform(subkey) < self.p_onion_on_counter,  # 50% chance to put on counter if non-full pot exists
                lambda _: True,  # Always put on counter if all pots are full
                None
            )
            
            # Create temporary state with updated rng_key
            temp_state = AgentState(
                holding=agent_state.holding,
                goal=agent_state.goal,
                nonfull_pots=agent_state.nonfull_pots,
                soup_ready=agent_state.soup_ready,
                rng_key=rng_key
            )
            
            return lax.cond(
                should_put_on_counter,
                put_onion_on_counter,
                put_onion_in_pot,
                (obs_3d, temp_state)
            )
            
        # Handle plate-related actions
        def handle_holding_plate(carry):
            obs_3d, agent_state = carry
            # Generate random number to determine if we should put plate on counter
            rng_key, subkey = jax.random.split(agent_state.rng_key)
            should_put_on_counter = jax.random.uniform(subkey) < self.p_plate_on_counter
            
            # Create temporary state with updated rng_key
            temp_state = AgentState(
                holding=agent_state.holding,
                goal=agent_state.goal,
                nonfull_pots=agent_state.nonfull_pots,
                soup_ready=agent_state.soup_ready,
                rng_key=rng_key
            )
            
            return lax.cond(
                should_put_on_counter,
                put_plate_on_counter,
                lambda carry: lax.cond(
                    agent_state.soup_ready,
                    plate_soup,
                    lambda carry: (Actions.stay, Goal.get_soup, carry[1].rng_key),
                    carry
                ),
                (obs_3d, temp_state)
            )
        
        # Main action selection logic
        action, new_goal, rng_key = lax.cond(
            agent_state.holding == Holding.nothing,
            # If holding nothing:
            # - Get plate if soup is ready
            # - Get onion if any pot needs onions and no soup is ready
            # - Otherwise stay and wait
            lambda carry: lax.cond(
                agent_state.soup_ready,
                get_plate,  # If soup is ready, get a plate
                lambda carry: lax.cond(
                    agent_state.nonfull_pots.any(),  # If no soup ready and any pot needs onions
                    get_onion,
                    lambda carry: (Actions.stay, Goal.get_soup, carry[1].rng_key),  # Wait for soup to cook
                    carry
                ),
                carry
            ),
            # If holding something, handle based on what's being held
            lambda carry: lax.cond(
                agent_state.holding == Holding.onion,
                handle_holding_onion,
                lambda carry: lax.cond(
                    agent_state.holding == Holding.plate,
                    handle_holding_plate,
                    lambda carry: lax.cond(
                        agent_state.holding == Holding.dish,
                        deliver_dish,
                        lambda carry: (Actions.stay, carry[1].goal, carry[1].rng_key),  # Default action
                        carry
                    ),
                    carry
                ),
                carry
            ),
            (obs_3d, agent_state)
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