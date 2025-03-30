from functools import partial
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from flax import struct
from jax import lax
from jaxmarl.environments.overcooked.overcooked import Actions, State as OvercookedState
from jaxmarl.environments.overcooked.common import OBJECT_TO_INDEX

@struct.dataclass
class Holding:
    nothing = 0
    onion = 1
    plate = 2
    dish = 3 # Completed soup

@struct.dataclass
class Goal:
    get_onion = 0
    put_onion = 1
    get_plate = 2
    get_soup = 3
    deliver = 4


@struct.dataclass
class AgentState:
    """Agent state for the heuristic agent."""
    holding: int
    goal: int
    onions_in_pot: int
    soup_ready: bool  # Whether there is a ready soup in any pot
    rng_key: jax.random.PRNGKey

class BaseAgent:
    """A base heuristic agent for the Overcooked environment.
    
    TODO: 
    - extend go_to_obj to interact with obj once there
    - create simple child agent that simply tries to stay out of way of other agent
    - create simple child agent that gets onions and places them in pot
    - create simple child agent that gets plate and attempts to deliver soup when ready.
    - create more complex child agent that can do both of the above
    """
    
    def __init__(self, agent_name: str, layout: Dict[str, Any]):
        self.agent_id = int(agent_name[-1])
        self.map_width = layout["width"]
        self.map_height = layout["height"]

        self.num_onion_piles = layout["onion_pile_idx"].shape[0]
        self.num_plate_piles = layout["plate_pile_idx"].shape[0]
        self.num_pots = layout["pot_idx"].shape[0]
        
        self.obs_shape = (self.map_height, self.map_width, 26)  # Overcooked uses 26 channels

        # Initial state - will be passed into and returned from get_action
        self.initial_state = AgentState(
            holding=Holding.nothing,
            goal=Goal.get_onion,
            onions_in_pot=0,
            soup_ready=False,
            rng_key=jax.random.PRNGKey(self.agent_id)
        )
    def get_name(self):
        return self.__class__.__name__
    
    def get_action(self, 
                   obs: jnp.ndarray, env_state: OvercookedState, 
                   agent_state: AgentState = None) -> Tuple[int, AgentState]:
        """Update agent state based on observation and get action."""
        if agent_state is None:
            agent_state = self.initial_state
            
        # Update state based on observation before getting action
        agent_state = self._update_state(obs, env_state, agent_state)
        action, agent_state = self._get_action(obs, agent_state)

        return action, agent_state
        
    @partial(jax.jit, static_argnums=(0,))
    def _update_state(self, obs: jnp.ndarray, env_state: OvercookedState, agent_state: AgentState) -> AgentState:
        """Update agent state based on observation.
        
        Args:
            obs: Flattened observation array
            agent_state: Current agent state
            
        Returns:
            Updated agent state
        """
        # Reshape observation to 3D
        obs_3d = jnp.reshape(obs, self.obs_shape)
        
        # Update onions_in_pot based on pot status layer (channel 16)
        pot_layer = obs_3d[:, :, 10]  # Channel 10: pot locations
        pot_status = obs_3d[:, :, 16]  # Channel 16: number of onions in pot
        # TODO: we should really track the number of onions in each pot
        # and not just the total number of onions in pots...
        onions_in_pot = jnp.sum(pot_status * pot_layer)
        
        # Update soup_ready based on soup ready layer (channel 21)
        soup_ready_layer = obs_3d[:, :, 21]  # Channel 21: soup ready
        soup_ready = jnp.any(soup_ready_layer > 0)
        
        # Update holding based on agent inventory information
        inv_idx = env_state.agent_inv[self.agent_id] # an integer coding the object in the agent's inventory
        
        # Map inventory values (1, 3, 5, 9) to Holding enum values (0, 1, 2, 3)
        holding = lax.cond(
            inv_idx == OBJECT_TO_INDEX['empty'],
            lambda _: Holding.nothing,
            lambda _: lax.cond(
                inv_idx == OBJECT_TO_INDEX['onion'],
                lambda _: Holding.onion,
                lambda _: lax.cond(
                    inv_idx == OBJECT_TO_INDEX['plate'],
                    lambda _: Holding.plate,
                    lambda _: lax.cond(
                        inv_idx == OBJECT_TO_INDEX['dish'],
                        lambda _: Holding.dish,
                        lambda _: Holding.nothing,  # Default to nothing for unsupported indices
                        None),
                    None),
                None),
            None)
                    
        # Create updated state
        updated_agent_state = AgentState(
            holding=holding,
            goal=agent_state.goal,
            onions_in_pot=onions_in_pot,
            soup_ready=soup_ready,
            rng_key=agent_state.rng_key
        )
        
        return updated_agent_state
        
    @partial(jax.jit, static_argnums=(0,))
    def _get_action(self, obs: jnp.ndarray, state: AgentState) -> Tuple[int, AgentState]:
        """Get action and updated state based on observation and current state.
        
        Args:
            obs: Flattened observation array
            state: AgentState containing agent's internal state
            
        Returns:
            action
        """
        raise NotImplementedError("Subclasses must implement this method")

    def _get_agent_pos(self, obs: jnp.ndarray) -> Tuple[int, int]:
        """Get the position of the agent."""
        agent_pos_layer = obs[:, :, 0]
        agent_pos = jnp.argwhere(agent_pos_layer > 0, size=1)[0]
        agent_y, agent_x = agent_pos
        return agent_y, agent_x

    def _go_to_obj(self, obs: jnp.ndarray, obj_type: str, rng_key: jax.random.PRNGKey) -> Tuple[int, jax.random.PRNGKey]:
        """Go to the nearest object of the given type."""
        agent_y, agent_x = self._get_agent_pos(obs)
        
        if obj_type == "pot":
            target_y, target_x = self._get_nearest_pot_pos(obs, agent_y, agent_x)
        elif obj_type == "onion":
            target_y, target_x = self._get_nearest_onion_or_plate_pos(obs, agent_y, agent_x, "onion")
        elif obj_type == "plate":
            target_y, target_x = self._get_nearest_onion_or_plate_pos(obs, agent_y, agent_x, "plate")
        
        print(f"\t[_go_to_obj] Agent position: {agent_y}, {agent_x}")
        print(f"\t[_go_to_obj] Target position: {target_y}, {target_x}")
        # Move towards target
        nearest_free_y, nearest_free_x = self._get_nearest_free_space(target_y, target_x, obs)
        print(f"\t[_go_to_obj] Nearest free space: {nearest_free_y}, {nearest_free_x}")
        action, rng_key = self._move_towards(agent_y, agent_x, 
                nearest_free_y, nearest_free_x, obs, rng_key)
        print(f"\t[_go_to_obj] Action: {action}")
        return action, rng_key

    def _get_nearest_pot_pos(self, obs: jnp.ndarray, agent_y: int, agent_x: int) -> Tuple[int, int]:
        '''Returns position of the nearest pot.
        
        Args:
            obs: Observation array
            agent_y: Agent's y position
            agent_x: Agent's x position
            
        Returns:
            Tuple of (y, x) coordinates of nearest pot
        '''
        # Get all pot positions
        pot_layer = obs[:, :, 10]
        all_pot_positions = jnp.argwhere(
            pot_layer > 0,
            size=self.num_pots  # Use number of pots from layout
        )

        # Calculate Manhattan distances to each pot
        distances = jnp.sum(
            jnp.abs(all_pot_positions - jnp.array([agent_y, agent_x])),
            axis=1
        )
        
        # Find the position of the nearest pot
        nearest_idx = jnp.argmin(distances)
        nearest_pot_pos = all_pot_positions[nearest_idx]
        
        return nearest_pot_pos

    def _get_nearest_onion_or_plate_pos(self, obs: jnp.ndarray, agent_y: int, agent_x: int,
                                        obj_type: str) -> Tuple[int, int]:
        '''Returns position of the nearest onion or plate.
        Checks both onion piles (channel 12) and onions on counter (channel 23).
        
        Args:
            obs: Observation array
            agent_y: Agent's y position
            agent_x: Agent's x position
            
        Returns:
            Tuple of (y, x) coordinates of nearest onion
        '''
        if obj_type == "onion":
            obj_pile_layer = obs[:, :, 12] # location of onion piles
            obj_layer = obs[:, :, 23] # location of onions on counter
            max_piles = self.num_onion_piles
        elif obj_type == "plate":
            obj_pile_layer = obs[:, :, 14] # location of plate piles
            obj_layer = obs[:, :, 22] # location of plates on counter
            max_piles = self.num_plate_piles
        else:
            raise ValueError(f"Invalid object type: {obj_type}")
        
        # Get location of one pile 
        default_pile_pos = jnp.argwhere(obj_pile_layer > 0, size=1)[0]

        # Combine both layers to get all onion/plate positions
        all_obj_positions = jnp.argwhere(
            jnp.logical_or(obj_pile_layer > 0, obj_layer > 0),
            size=max_piles + 2 # consider at most 2 objects that are not the piles
        )
        # argwhere returns all-zero positions if fewer than max_piles + 2 objects found
        # Replace all-zero positions with default pile position
        is_zero_pos = jnp.all(all_obj_positions == 0, axis=1)
        all_obj_positions = jnp.where(
            jnp.expand_dims(is_zero_pos, axis=1),
            default_pile_pos,
            all_obj_positions
        )
        # Calculate Manhattan distances to each onion/plate
        distances = jnp.sum(
            jnp.abs(all_obj_positions - jnp.array([agent_y, agent_x])),
            axis=1
        )
        
        # Find the position of the nearest onion/plate
        nearest_idx = jnp.argmin(distances)
        nearest_obj_pos = all_obj_positions[nearest_idx]
        
        return nearest_obj_pos
            
    def _get_nearest_free_space(self, y: int, x: int, obs: jnp.ndarray) -> Tuple[int, int]:
        """Get the nearest free space to the target position to figure out where to move.
        Does not account for other agents, which is on purpose.
        
        Args:
            y: Target y position
            x: Target x position
            obs: Observation array
            
        Returns:
            Tuple of (y, x) coordinates of nearest free space
        """
        # Get occupied spaces (walls and counters)
        wall_mask = obs[:, :, 11] > 0       # Channel 11: counter/wall locations
        
        # Check bounds for all four directions
        up_valid = y > 0
        down_valid = y < self.map_height - 1
        right_valid = x < self.map_width - 1
        left_valid = x > 0
        
        # Check if each adjacent position is free
        up_free = up_valid & ~wall_mask[y - 1, x]
        down_free = down_valid & ~wall_mask[y + 1, x]
        right_free = right_valid & ~wall_mask[y, x + 1]
        left_free = left_valid & ~wall_mask[y, x - 1]
        
        # Return first free position found (prioritizing up, down, right, left)
        free_space = lax.cond(
            up_free,
            lambda _: (y - 1, x),
            lambda _: lax.cond(
                down_free,
                lambda _: (y + 1, x),
                lambda _: lax.cond(
                    right_free,
                    lambda _: (y, x + 1),
                    lambda _: lax.cond(
                        left_free,
                        lambda _: (y, x - 1),
                        lambda _: (y, x),  # Fallback to original position if all adjacent spaces are occupied
                        None),
                    None),
                None),
            None)
        return free_space

    def _move_towards(self, start_y: int, start_x: int, 
                      target_y: int, target_x: int, 
                      obs: jnp.ndarray, key: jax.random.PRNGKey) -> int:
        """Move towards target position while avoiding collisions."""
        # Calculate differences
        x_diff = start_x - target_x
        y_diff = start_y - target_y
        
        # Get occupied spaces (walls, other agent, and counters)
        wall_mask = obs[:, :, 11] > 0       # Channel 11: counter/wall locations
        other_agent_mask = obs[:, :, 1] > 0  # Channel 1: other agent position
        occupied_mask = jnp.logical_or(wall_mask, other_agent_mask)

        # Check if moving in each direction would lead to an occupied space
        up_valid = (start_y > 0) & (~occupied_mask[start_y - 1, start_x])
        down_valid = (start_y < self.map_height - 1) & (~occupied_mask[start_y + 1, start_x])
        right_valid = (start_x < self.map_width - 1) & (~occupied_mask[start_y, start_x + 1])
        left_valid = (start_x > 0) & (~occupied_mask[start_y, start_x - 1])
        stay_valid = True
        
        # Base scores: prefer directions that reduce distance to target (or stay if at target)
        # score encoding: [up, down, right, left, stay]
        up_score, down_score, right_score, left_score, stay_score = 0, 0, 0, 0, 0
        up_score = lax.cond(y_diff > 0, lambda _: 1, lambda _: 0, None)
        down_score = lax.cond(y_diff < 0, lambda _: 1, lambda _: 0, None)        
        right_score = lax.cond(
            y_diff == 0,
            lambda _: lax.cond(x_diff > 0, lambda _: 1, lambda _: 0, None),
            lambda _: 0,
            None)
        
        left_score = lax.cond(
            y_diff == 0,
            lambda _: lax.cond(x_diff < 0, lambda _: 1, lambda _: 0, None),
            lambda _: 0,
            None)
        
        stay_score = lax.cond(
            jnp.logical_and(y_diff == 0, x_diff == 0),
            lambda _: 1,
            lambda _: 0,
            None
        )

        scores = jnp.array([up_score, down_score, right_score, left_score, stay_score])
        # print(f"\t\t[_move_towards] Mmt scores: {scores}")
        # Set scores to large negative number for invalid moves
        scores = jnp.where(
            jnp.array([up_valid, down_valid, right_valid, left_valid, stay_valid]),
            scores,
            -1000.0
        )

        # Add small random noise to break ties
        key, subkey = jax.random.split(key)
        noise = jax.random.uniform(subkey, shape=(5,), minval=-0.01, maxval=0.01)

        scores += noise
                
        # Choose direction with highest score
        direction = jnp.argmax(scores)
        
        # Map direction to action
        action = lax.switch(
            direction,
            [
                lambda: Actions.up,
                lambda: Actions.down,
                lambda: Actions.right,
                lambda: Actions.left,
                lambda: Actions.stay
            ]
        )
        # print(f"\t\t[_move_towards] Mmt action: {action}")
        return action, key

    def _go_to_obj_and_interact(self, obs: jnp.ndarray, obj_type: str, rng_key: jax.random.PRNGKey) -> Tuple[int, jax.random.PRNGKey]:
        """Go to the nearest object of the given type. When adjacent to object, turn to face it
        and interact with it.
        
        Args:
            obs: 3D observation array
            obj_type: Type of object to go to ("pot", "onion", "plate")
            rng_key: Random key for action selection
            
        Returns:
            Tuple of (action, new_rng_key)
        """
        # Get agent position
        agent_y, agent_x = self._get_agent_pos(obs)
        
        # Get target position based on object type
        if obj_type == "pot":
            target_y, target_x = self._get_nearest_pot_pos(obs, agent_y, agent_x)
        elif obj_type == "onion":
            target_y, target_x = self._get_nearest_onion_or_plate_pos(obs, agent_y, agent_x, "onion")
        elif obj_type == "plate":
            target_y, target_x = self._get_nearest_onion_or_plate_pos(obs, agent_y, agent_x, "plate")
        else:
            raise ValueError(f"Invalid object type: {obj_type}")
        
        # print(f"Agent position: {agent_y}, {agent_x}")
        # print(f"Target {obj_type} position: {target_y}, {target_x}")

        # Check if agent is adjacent to target
        is_adjacent = jnp.logical_or(
            jnp.logical_and(jnp.abs(agent_y - target_y) == 1, agent_x == target_x),
            jnp.logical_and(jnp.abs(agent_x - target_x) == 1, agent_y == target_y)
        )

        # Determine required direction to face the target
        def _get_target_orientation_action(agent_y, agent_x, target_y, target_x):
            '''Assumes agent is adjacent to target, computes the direction action to face the target.
            '''
            y_diff = agent_y - target_y
            x_diff = agent_x - target_x
            action = lax.cond(
                jnp.abs(y_diff) > jnp.abs(x_diff),
                # If vertical distance is greater, face up or down
                lambda _: lax.cond(
                    y_diff > 0,
                    lambda _: Actions.up,
                    lambda _: Actions.down,
                    None
                ),
                # If horizontal distance is greater, face left or right
                lambda _: lax.cond(
                    x_diff > 0,
                    lambda _: Actions.left,  # Face left
                    lambda _: Actions.right,  # Face right
                    None
                ),
                None)
            return action

        # Get agent's current direction from observation
        agent_dir_layers = obs[:, :, 2:6]  # Layers 2-5 contain direction information for ego agent
        agent_dir_idx = jnp.argmax(agent_dir_layers[agent_y, agent_x])

        target_orientation_action = _get_target_orientation_action(agent_y, agent_x, target_y, target_x)
        
        print(f"\t[_go_to_obj_and_interact] Target orientation action: {target_orientation_action}")
        print(f"\t[_go_to_obj_and_interact] Agent direction index: {agent_dir_idx}")
        print(f"\t[_go_to_obj_and_interact] Is adjacent: {is_adjacent}")

        # If adjacent but not facing the right direction, turn to face it
        # if adjacent and facing the right direction, interact
        # If not adjacent, move towards the object
        action, rng_key = lax.cond(
            is_adjacent,
            lambda _: lax.cond(
                agent_dir_idx == target_orientation_action,
                lambda _: (Actions.interact, rng_key),
                lambda _: (target_orientation_action, rng_key),
                None
            ),
            lambda _: self._go_to_obj(obs, obj_type, rng_key),
            None
        )
        print(f"\t[_go_to_obj_and_interact] Action: {action}")
        return action, rng_key