'''BC-specific featurizer for Overcooked-v1 observations.

This module provides a standalone Featurizer class that converts Overcooked-v1 states
into the featurized observation format required by the BC agent, without needing
to wrap the entire environment.
'''
import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, Tuple
from enum import IntEnum
from jaxmarl.environments.overcooked.overcooked import (
    State,
    POT_EMPTY_STATUS,
    POT_FULL_STATUS,
    POT_READY_STATUS,
    MAX_ONIONS_IN_POT,
    URGENCY_CUTOFF,
)
from jaxmarl.environments.overcooked.common import OBJECT_TO_INDEX

import chex
from functools import partial
from flax import struct


@struct.dataclass
class ObjectType:
    """Object type indices for held items and targets for Overcooked-v1 state featurization."""
    ONION: int = 0
    SOUP: int = 1  
    PLATE: int = 2
    TOMATO_OR_NONE: int = 3
    
    # Special value for targets that don't correspond to held items
    NO_ITEM_TARGET: int = -1
    
    # Total number of object types (for one-hot encoding)
    NUM_TYPES: int = 4


# Direction enum for path planning
class Direction(IntEnum):
    UP = 0
    DOWN = 1
    RIGHT = 2
    LEFT = 3

# Position dataclass for path planning
@struct.dataclass
class Position:
    x: jnp.ndarray
    y: jnp.ndarray
    
    def checked_move(self, direction: int, width: int, height: int) -> Tuple['Position', bool]:
        """Move in a direction and check if the new position is valid."""
        new_y = jax.lax.select(
            direction == Direction.UP,
            self.y - 1,
            jax.lax.select(
                direction == Direction.DOWN,
                self.y + 1,
                self.y
            )
        )
        
        new_x = jax.lax.select(
            direction == Direction.RIGHT,
            self.x + 1,
            jax.lax.select(
                direction == Direction.LEFT,
                self.x - 1,
                self.x
            )
        )
        
        is_valid = (0 <= new_x) & (new_x < width) & (0 <= new_y) & (new_y < height)
        return Position(x=new_x, y=new_y), is_valid
    
    @staticmethod
    def opposite(direction: int) -> int:
        """Get the opposite direction."""
        return jax.lax.select(
            direction == Direction.UP,
            Direction.DOWN,
            jax.lax.select(
                direction == Direction.DOWN,
                Direction.UP,
                jax.lax.select(
                    direction == Direction.RIGHT,
                    Direction.LEFT,
                    jax.lax.select(
                        direction == Direction.LEFT,
                        Direction.RIGHT,
                        direction
                    )
                )
            )
        )

# All directions for vectorization
ALL_DIRECTIONS = jnp.array([Direction.UP, Direction.DOWN, Direction.RIGHT, Direction.LEFT])


class OvercookedV1PathPlanner:
    """True A* path planner with precomputed distances for Overcooked-v1"""
    
    def __init__(self, move_area: jnp.ndarray):
        self._precompute(move_area)
    
    @partial(jax.jit, static_argnums=(0,))
    def get_closest_target_pos(self, targets: jnp.ndarray, pos: jnp.ndarray, 
                              direction: int, move_area: jnp.ndarray, reachable_area: jnp.ndarray = None) -> Tuple[jnp.ndarray, bool]:
        """Find closest target position using precomputed A* distances."""
        y, x = pos[1], pos[0]
        
        # If current position is a target, return it
        is_current_target = targets[y, x]
        
        # Get position lookup index
        pos_lookup_idx = self.position_to_idx[y, x]
        
        def _compute_min_moves():
            min_moves = self.precomputed_min_moves[pos_lookup_idx, direction]
            return self._get_pos_from_min_moves_grid(min_moves, targets)
        
        is_allowed_pos = pos_lookup_idx != -1
        return jax.lax.cond(
            is_current_target | ~is_allowed_pos,
            lambda: (pos, is_allowed_pos),
            _compute_min_moves,
        )
    
    def _precompute(self, move_area: jnp.ndarray):
        """Precompute all possible path distances."""
        pos_idx = jnp.argwhere(move_area)
        num_pos = pos_idx.shape[0]
        positions = Position(y=pos_idx[:, 0], x=pos_idx[:, 1])
        
        # Precompute min moves for all positions and directions
        self.precomputed_min_moves = jax.vmap(
            jax.vmap(self._compute_min_moves, in_axes=(None, 0, None)),
            in_axes=(0, None, None),
        )(positions, ALL_DIRECTIONS, move_area)
        
        # Create position to index mapping
        self.position_to_idx = (
            jnp.full_like(move_area, -1, dtype=jnp.int32)
            .at[positions.y, positions.x]
            .set(jnp.arange(num_pos))
        )
    
    @staticmethod
    def _get_pos_from_min_moves_grid(min_moves: jnp.ndarray, targets: jnp.ndarray) -> Tuple[jnp.ndarray, bool]:
        """Get closest target position from precomputed distances."""
        min_moves_targets = jnp.where(targets, min_moves, jnp.inf)
        
        min_idx = jnp.argmin(min_moves_targets)
        min_y, min_x = jnp.divmod(min_idx, min_moves.shape[1])
        
        is_valid = jnp.any(jnp.isfinite(min_moves_targets))
        
        return jnp.array([min_x, min_y]), is_valid
    
    @staticmethod
    @jax.jit
    def _compute_min_moves(pos: Position, direction: int, mask: jnp.ndarray) -> jnp.ndarray:
        """Compute minimum moves from a position in a given direction using A* algorithm."""
        assert mask.ndim == 2 and mask.dtype == jnp.bool_
        H, W = mask.shape
        
        ys, xs = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing="ij")
        
        def _obstacle_ahead(pos, dir):
            new_pos, is_valid = pos.checked_move(dir, W, H)
            obstacle_ahead = ~mask[new_pos.y, new_pos.x]
            return ~is_valid | obstacle_ahead
        
        obstacle_ahead = jax.vmap(_obstacle_ahead, in_axes=(None, 0), out_axes=-1)(
            Position(x=xs, y=ys), ALL_DIRECTIONS
        )
        obstacle_ahead &= mask[..., jnp.newaxis]
        
        def cond_fun(loop_carry):
            _, changed = loop_carry
            return changed
        
        def body_fun(loop_carry):
            dist, _ = loop_carry
            
            min_across_last_dim = jnp.min(dist, axis=-1)
            
            def _move_dir(dir):
                moved_dist = jnp.full_like(min_across_last_dim, jnp.inf)
                
                moved_dist = jax.lax.select(
                    dir == Direction.UP,
                    moved_dist.at[:-1, :].set(min_across_last_dim[1:, :]),
                    jax.lax.select(
                        dir == Direction.DOWN,
                        moved_dist.at[1:, :].set(min_across_last_dim[:-1, :]),
                        jax.lax.select(
                            dir == Direction.RIGHT,
                            moved_dist.at[:, 1:].set(min_across_last_dim[:, :-1]),
                            jax.lax.select(
                                dir == Direction.LEFT,
                                moved_dist.at[:, :-1].set(min_across_last_dim[:, 1:]),
                                moved_dist
                            )
                        )
                    )
                )
                return jnp.where(mask, moved_dist, jnp.inf)
            
            dist_up = _move_dir(Direction.UP)
            dist_down = _move_dir(Direction.DOWN)
            dist_right = _move_dir(Direction.RIGHT)
            dist_left = _move_dir(Direction.LEFT)
            
            dist_new = jnp.stack([dist_up, dist_down, dist_right, dist_left], axis=-1)
            
            blocked_new_dist = jnp.where(
                obstacle_ahead, min_across_last_dim[..., jnp.newaxis], jnp.inf
            )
            
            dist_new = jnp.minimum(dist_new, blocked_new_dist)
            dist_new += 1
            dist_updated = jnp.minimum(dist, dist_new)
            
            changed = jnp.any(dist_updated != dist)
            
            return dist_updated, changed
        
        initial_dist = jnp.full((H, W, 4), jnp.inf, dtype=jnp.float32)
        initial_dist = initial_dist.at[pos.y, pos.x, direction].set(0)
        
        dist_final, _ = jax.lax.while_loop(cond_fun, body_fun, (initial_dist, True))
        
        def _compute_min_cost(pos, dir):
            new_pos, is_valid = pos.checked_move(dir, W, H)
            opposite_dir = Position.opposite(dir)
            
            return jnp.where(
                is_valid,
                dist_final[new_pos.y, new_pos.x, opposite_dir],
                jnp.inf,
            )
        
        min_cost_to_target = jax.vmap(
            _compute_min_cost, in_axes=(None, 0), out_axes=-1
        )(Position(x=xs, y=ys), ALL_DIRECTIONS)
        min_cost_to_target = jnp.min(min_cost_to_target, axis=-1)
        
        # We only care about target cells
        min_cost_to_target = jnp.where(mask, jnp.inf, min_cost_to_target)
        
        return min_cost_to_target


def compute_enclosed_spaces(empty_mask: jnp.ndarray) -> jnp.ndarray:
    """Compute the enclosed spaces in the environment."""
    height, width = empty_mask.shape
    id_grid = jnp.arange(empty_mask.size, dtype=jnp.int32).reshape(empty_mask.shape)
    id_grid = jnp.where(empty_mask, id_grid, -1)

    def _body_fun(val):
        _, curr = val

        def _next_val(pos):
            y, x = pos // width, pos % width
            
            # Check 4 neighbors using JAX-compatible operations
            offsets = jnp.array([(-1, 0), (1, 0), (0, -1), (0, 1)])
            
            def check_neighbor(offset):
                dy, dx = offset
                ny, nx = y + dy, x + dx
                
                in_bounds = (0 <= ny) & (ny < height) & (0 <= nx) & (nx < width)
                neighbor_val = jax.lax.select(in_bounds, curr[ny, nx], -1)
                return neighbor_val
            
            neighbor_vals = jax.vmap(check_neighbor)(offsets)
            self_val = curr[y, x]
            all_vals = jnp.concatenate([neighbor_vals, jnp.array([self_val])])
            
            new_val = jnp.max(all_vals)
            return jax.lax.select(self_val == -1, self_val, new_val)

        pos_indices = jnp.arange(height * width)
        next_vals = jax.vmap(_next_val)(pos_indices).reshape(height, width)
        stop = jnp.all(curr == next_vals)
        return stop, next_vals

    def _cond_fun(val):
        return ~val[0]

    initial_val = (False, id_grid)
    _, res = jax.lax.while_loop(_cond_fun, _body_fun, initial_val)
    return res


def mark_adjacent_cells(mask: jnp.ndarray) -> jnp.ndarray:
    """Mark cells adjacent to the given mask"""
    up = jnp.roll(mask, shift=-1, axis=0)
    down = jnp.roll(mask, shift=1, axis=0)
    left = jnp.roll(mask, shift=-1, axis=1)
    right = jnp.roll(mask, shift=1, axis=1)
    
    # Prevent wrapping
    up = up.at[-1, :].set(False)
    down = down.at[0, :].set(False)
    left = left.at[:, -1].set(False)
    right = right.at[:, 0].set(False)
    
    return mask | up | down | left | right


class BCFeaturizer:
    """
    BC-specific state featurizer for Overcooked-v1 states.
    
    Converts Overcooked-v1 states into the featurized observation format
    required by the BC agent. This is completely standalone and includes
    its own get_obs implementation copied from the original Overcooked environment.
    
    Example usage:
        layout = layouts["cramped_room"] 
        featurizer = BCFeaturizer(layout, num_pots=2)
        featurized_obs = featurizer.featurize_state(state)
    """
    
    def __init__(self, layout, num_pots: int = 2, max_steps: int = 400):
        """
        Args:
            layout: Layout dictionary containing environment layout information
            num_pots: Number of closest pots to encode for each player
            num_agents: Number of agents in the environment
            max_steps: Maximum number of steps in an episode (for urgency layer)
        """
        self.num_pots = num_pots
        self.num_agents = 2
        self.height = layout["height"]
        self.width = layout["width"]
        self.max_steps = max_steps

        # compute wall map
        wall_idx = layout.get("wall_idx")
        all_pos = jnp.arange(np.prod([self.height, self.width]), dtype=jnp.uint32)
        occupied_mask = jnp.zeros_like(all_pos)
        occupied_mask = occupied_mask.at[wall_idx].set(1)
        self.wall_map = occupied_mask.reshape(self.height, self.width).astype(jnp.bool_)
        
        # Initialize path planner (will be set during precomputing layout features)
        self.path_planner = None
        self._precompute_layout_features()
        
        self.feature_size = self._calculate_feature_size()
    
    def _precompute_layout_features(self):
        """Pre-compute layout-specific features like enclosed spaces and path planning."""

        # Compute move area (inverse of wall map)
        move_area = ~self.wall_map
        
        # Pre-compute enclosed spaces
        self.enclosed_spaces = compute_enclosed_spaces(move_area)
        
        # Initialize path planner with move area
        self.path_planner = OvercookedV1PathPlanner(move_area)
    
    def _calculate_feature_size(self) -> int:
        """Calculate the size of the feature vector for each agent."""
        # Player features: orientation(4) + held_obj(4) + onion(2) + tomato(2) + dish(2) + 
        # soup(2) + soup_onions(1) + soup_tomatoes(1) + serving(2) + empty_counter(2) + 
        # pot_features(num_pots*10) + wall_features(4)
        player_features = 4 + ObjectType.NUM_TYPES + 2 + 2 + 2 + 2 + 1 + 1 + 2 + 2 + self.num_pots*10 + 4
        
        # Other player features (for 2 agents, this is 1 other player with full feature vector)
        other_player_features = player_features * (self.num_agents - 1)
        
        # Relative positions to other players (for 2 agents, this is 1 other player * 2 coordinates)
        relative_positions = 2 * (self.num_agents - 1)
        
        # Absolute position
        absolute_position = 2
        
        return player_features + other_player_features + relative_positions + absolute_position
    
    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """Return a full observation, of size (height x width x n_layers), where n_layers = 26.
        Layers are of shape (height x width) and  are binary (0/1) except where indicated otherwise.
        The obs is very sparse (most elements are 0), which prob. contributes to generalization problems in Overcooked.
        A v2 of this environment should have much more efficient observations, e.g. using item embeddings

        The list of channels is below. Agent-specific layers are ordered so that an agent perceives its layers first.
        Env layers are the same (and in same order) for both agents.

        Agent positions :
        0. position of agent i (1 at agent loc, 0 otherwise)
        1. position of agent (1-i)

        Agent orientations :
        2-5. agent_{i}_orientation_0 to agent_{i}_orientation_3 (layers are entirely zero except for the one orientation
        layer that matches the agent orientation. That orientation has a single 1 at the agent coordinates.)
        6-9. agent_{i-1}_orientation_{dir}

        Static env positions (1 where object of type X is located, 0 otherwise.):
        10. pot locations
        11. counter locations (table)
        12. onion pile locations
        13. tomato pile locations (tomato layers are included for consistency, but this env does not support tomatoes)
        14. plate pile locations
        15. delivery locations (goal)

        Pot and soup specific layers. These are non-binary layers:
        16. number of onions in pot (0,1,2,3) for elements corresponding to pot locations. Nonzero only for pots that
        have NOT started cooking yet. When a pot starts cooking (or is ready), the corresponding element is set to 0
        17. number of tomatoes in pot.
        18. number of onions in soup (0,3) for elements corresponding to either a cooking/done pot or to a soup (dish)
        ready to be served. This is a useless feature since all soups have exactly 3 onions, but it made sense in the
        full Overcooked where recipes can be a mix of tomatoes and onions
        19. number of tomatoes in soup
        20. pot cooking time remaining. [19 -> 1] for pots that are cooking. 0 for pots that are not cooking or done
        21. soup done. (Binary) 1 for pots done cooking and for locations containing a soup (dish). O otherwise.

        Variable env layers (binary):
        22. plate locations
        23. onion locations
        24. tomato locations

        Urgency:
        25. Urgency. The entire layer is 1 there are 40 or fewer remaining time steps. 0 otherwise
        """
        width, height, n_channels = (self.width, self.height, 26)
        padding = (state.maze_map.shape[0]-height) // 2

        maze_map = state.maze_map[padding:-padding, padding:-padding, 0]
        soup_loc = jnp.array(maze_map == OBJECT_TO_INDEX["dish"], dtype=jnp.uint8)

        pot_loc_layer = jnp.array(maze_map == OBJECT_TO_INDEX["pot"], dtype=jnp.uint8)
        pot_status = state.maze_map[padding:-padding, padding:-padding, 2] * pot_loc_layer
        onions_in_pot_layer = jnp.minimum(POT_EMPTY_STATUS - pot_status, MAX_ONIONS_IN_POT) * (pot_status >= POT_FULL_STATUS)    # 0/1/2/3, as long as not cooking or not done
        onions_in_soup_layer = jnp.minimum(POT_EMPTY_STATUS - pot_status, MAX_ONIONS_IN_POT) * (pot_status < POT_FULL_STATUS) \
                               * pot_loc_layer + MAX_ONIONS_IN_POT * soup_loc   # 0/3, as long as cooking or done
        pot_cooking_time_layer = pot_status * (pot_status < POT_FULL_STATUS)                           # Timer: 19 to 0
        soup_ready_layer = pot_loc_layer * (pot_status == POT_READY_STATUS) + soup_loc                 # Ready soups, plated or not
        urgency_layer = jnp.ones(maze_map.shape, dtype=jnp.uint8) * ((self.max_steps - state.time) < URGENCY_CUTOFF)

        agent_pos_layers = jnp.zeros((2, height, width), dtype=jnp.uint8)
        agent_pos_layers = agent_pos_layers.at[0, state.agent_pos[0, 1], state.agent_pos[0, 0]].set(1)
        agent_pos_layers = agent_pos_layers.at[1, state.agent_pos[1, 1], state.agent_pos[1, 0]].set(1)

        # Add agent inv: This works because loose items and agent cannot overlap
        agent_inv_items = jnp.expand_dims(state.agent_inv,(1,2)) * agent_pos_layers
        maze_map = jnp.where(jnp.sum(agent_pos_layers,0), agent_inv_items.sum(0), maze_map)
        soup_ready_layer = soup_ready_layer \
                           + (jnp.sum(agent_inv_items,0) == OBJECT_TO_INDEX["dish"]) * jnp.sum(agent_pos_layers,0)
        onions_in_soup_layer = onions_in_soup_layer \
                               + (jnp.sum(agent_inv_items,0) == OBJECT_TO_INDEX["dish"]) * 3 * jnp.sum(agent_pos_layers,0)

        env_layers = [
            jnp.array(maze_map == OBJECT_TO_INDEX["pot"], dtype=jnp.uint8),       # Channel 10
            jnp.array(maze_map == OBJECT_TO_INDEX["wall"], dtype=jnp.uint8),
            jnp.array(maze_map == OBJECT_TO_INDEX["onion_pile"], dtype=jnp.uint8),
            jnp.zeros(maze_map.shape, dtype=jnp.uint8),                           # tomato pile
            jnp.array(maze_map == OBJECT_TO_INDEX["plate_pile"], dtype=jnp.uint8),
            jnp.array(maze_map == OBJECT_TO_INDEX["goal"], dtype=jnp.uint8),        # 15
            jnp.array(onions_in_pot_layer, dtype=jnp.uint8),
            jnp.zeros(maze_map.shape, dtype=jnp.uint8),                           # tomatoes in pot
            jnp.array(onions_in_soup_layer, dtype=jnp.uint8),
            jnp.zeros(maze_map.shape, dtype=jnp.uint8),                           # tomatoes in soup
            jnp.array(pot_cooking_time_layer, dtype=jnp.uint8),                     # 20
            jnp.array(soup_ready_layer, dtype=jnp.uint8),
            jnp.array(maze_map == OBJECT_TO_INDEX["plate"], dtype=jnp.uint8),
            jnp.array(maze_map == OBJECT_TO_INDEX["onion"], dtype=jnp.uint8),
            jnp.zeros(maze_map.shape, dtype=jnp.uint8),                           # tomatoes
            urgency_layer,                                                          # 25
        ]

        # Agent related layers
        agent_direction_layers = jnp.zeros((8, height, width), dtype=jnp.uint8)
        dir_layer_idx = state.agent_dir_idx+jnp.array([0,4])
        agent_direction_layers = agent_direction_layers.at[dir_layer_idx,:,:].set(agent_pos_layers)

        # Both agent see their layers first, then the other layer
        alice_obs = jnp.zeros((n_channels, height, width), dtype=jnp.uint8)
        alice_obs = alice_obs.at[0:2].set(agent_pos_layers)

        alice_obs = alice_obs.at[2:10].set(agent_direction_layers)
        alice_obs = alice_obs.at[10:].set(jnp.stack(env_layers))

        bob_obs = jnp.zeros((n_channels, height, width), dtype=jnp.uint8)
        bob_obs = bob_obs.at[0].set(agent_pos_layers[1]).at[1].set(agent_pos_layers[0])
        bob_obs = bob_obs.at[2:6].set(agent_direction_layers[4:]).at[6:10].set(agent_direction_layers[0:4])
        bob_obs = bob_obs.at[10:].set(jnp.stack(env_layers))

        alice_obs = jnp.transpose(alice_obs, (1, 2, 0))
        bob_obs = jnp.transpose(bob_obs, (1, 2, 0))

        return {"agent_0" : alice_obs, "agent_1" : bob_obs}
        
    def featurize_state(self, state: State) -> Dict[str, jnp.ndarray]:
        """
        Convert an Overcooked-v1 state to featurized observations for all agents.
        
        Args:
            state: Overcooked-v1 state
            
        Returns:
            Dict mapping agent_id to featurized observation array
        """
        # Get original spatial observations using internal get_obs
        spatial_obs = self.get_obs(state)
        
        featurized_obs = {}
        
        # Get agent positions for relative positioning
        agent_positions = []
        for agent_id in range(self.num_agents):
            agent_obs = spatial_obs[f"agent_{agent_id}"]
            agent_pos, _ = self._get_agent_position_and_orientation(agent_obs)
            agent_positions.append(agent_pos)
        
        # Featurize each agent's observation
        for agent_id in range(self.num_agents):
            agent_obs = spatial_obs[f"agent_{agent_id}"]
            agent_features = self._featurize_agent_observation(agent_obs)
            
            # Get other player features (FULL feature vectors)
            other_player_features = []
            for other_id in range(self.num_agents):
                if other_id != agent_id:
                    other_obs = spatial_obs[f"agent_{other_id}"]
                    other_features = self._featurize_agent_observation(other_obs)
                    other_player_features.append(other_features)
            
            # Calculate relative positions to other players
            relative_positions = []
            for other_id in range(self.num_agents):
                if other_id != agent_id:
                    rel_pos = agent_positions[other_id] - agent_positions[agent_id]
                    relative_positions.append(rel_pos)
            
            # Concatenate all features
            all_features = jnp.concatenate([
                agent_features,
                jnp.concatenate(other_player_features) if other_player_features else jnp.array([]),
                jnp.concatenate(relative_positions) if relative_positions else jnp.array([]),
                agent_positions[agent_id]  # absolute position
            ])
            
            featurized_obs[f"agent_{agent_id}"] = all_features
        
        return featurized_obs
    
    def featurize_state_as_array(self, state: State) -> jnp.ndarray:
        """
        Convert an Overcooked-v1 state to featurized observations as a stacked array.
        
        Args:
            state: Overcooked-v1 state
            
        Returns:
            Array of shape (num_agents, feature_size) containing featurized observations
        """
        featurized_obs_dict = self.featurize_state(state)
        
        # Stack observations into array indexed by agent_id
        obs_array = jnp.stack([
            featurized_obs_dict[f"agent_{i}"] for i in range(self.num_agents)
        ])
        
        return obs_array
    
    def _get_agent_position_and_orientation(self, obs: chex.Array) -> Tuple[jnp.ndarray, int]:
        """Extract agent position and orientation from observation."""
                # Find agent position using argmax (JAX-compatible)
        # Channel 0 always contains the current agent's position
        agent_pos_layer = obs[:, :, 0]
        agent_flat_idx = jnp.argmax(agent_pos_layer.flatten())
        agent_y, agent_x = jnp.divmod(agent_flat_idx, self.width)
        
        # Find agent orientation
        # Channels 2-5 are always current agent's orientation
        orientation_layers = obs[:, :, 2:6]
        orientation_flat_idx = jnp.argmax(orientation_layers[agent_y, agent_x])
        
        return jnp.array([agent_x, agent_y]), orientation_flat_idx
    
    def _get_held_object_idx(self, obs: chex.Array) -> int:
        """Extract what object the agent is holding (returns index)."""
        # Channel 0 always contains the current agent's position
        agent_pos_layer = obs[:, :, 0]
        agent_flat_idx = jnp.argmax(agent_pos_layer.flatten())
        agent_y, agent_x = jnp.divmod(agent_flat_idx, self.width)
        
        # Check if agent is holding a dish (soup)
        soup_ready_layer = obs[:, :, 21]  # Channel 21: soup done
        has_soup = soup_ready_layer[agent_y, agent_x] == 1
        
        # Check for other objects on agent position
        plate_layer = obs[:, :, 22]  # Channel 22: plate locations
        onion_layer = obs[:, :, 23]  # Channel 23: onion locations
        
        has_plate = plate_layer[agent_y, agent_x] == 1
        has_onion = onion_layer[agent_y, agent_x] == 1
        
        return jax.lax.select(has_onion, ObjectType.ONION, 
                jax.lax.select(has_soup, ObjectType.SOUP,
                jax.lax.select(has_plate, ObjectType.PLATE, ObjectType.TOMATO_OR_NONE)))
    
    def _get_closest_feature_distance(self, agent_pos: jnp.ndarray, 
                                    feature_layer: jnp.ndarray, move_area: jnp.ndarray,
                                    direction: int, reachable_area: jnp.ndarray = None,
                                    held_obj_idx: int = None, target_obj_idx: int = ObjectType.NO_ITEM_TARGET) -> jnp.ndarray:
        """Find the closest feature using path planning with reachability analysis."""
        
        target_pos, is_valid = self.path_planner.get_closest_target_pos(
            feature_layer, agent_pos, direction, move_area, reachable_area
        )
        
        delta = target_pos - agent_pos
        computed_distance = jax.lax.select(is_valid, delta, jnp.array([0, 0]))
        
        # If agent is holding the target item, distance is [0, 0]
        is_holding_target = held_obj_idx == target_obj_idx
        return jax.lax.select(is_holding_target, jnp.array([0, 0]), computed_distance)
    
    def _get_multiple_pot_features(self, obs: chex.Array, agent_pos: jnp.ndarray, 
                                 move_area: jnp.ndarray, direction: int, reachable_area: jnp.ndarray = None) -> jnp.ndarray:
        """Get features for multiple pots using path planning with reachability."""
        height, width, channels = obs.shape
        pot_layer = obs[:, :, 10]  # Channel 10: pot locations
        
        # Create a mask for reachable pots
        pot_mask = pot_layer & reachable_area
        
        # Initialize pot features array
        all_pot_features = jnp.zeros((self.num_pots, 10), dtype=jnp.int32)
        
        # Find pots in order of distance using path planning
        remaining_pots = pot_mask.copy()
        
        for pot_idx in range(self.num_pots):
            # Find closest remaining pot using path planning
            target_pos, is_valid = self.path_planner.get_closest_target_pos(
                remaining_pots, agent_pos, direction, move_area, reachable_area
            )
            
            # Get features for this pot
            pot_features = self._get_pot_features_at_position(obs, agent_pos, target_pos, is_valid)
            all_pot_features = all_pot_features.at[pot_idx].set(pot_features)
            
            # Remove this pot from remaining pots
            remaining_pots = remaining_pots.at[target_pos[1], target_pos[0]].set(False)
        
        return all_pot_features.flatten()
    
    def _get_pot_features_at_position(self, obs: chex.Array, agent_pos: jnp.ndarray, 
                                    pot_pos: jnp.ndarray, is_valid: bool) -> jnp.ndarray:
        """Get pot features at a specific position."""
        height, width, channels = obs.shape
        
        # If pot is not valid, return zeros
        pot_features = jnp.zeros(10, dtype=jnp.int32)
        
        def _compute_pot_features():
            pot_y, pot_x = pot_pos[1], pot_pos[0]
            pot_distance = pot_pos - agent_pos
            
            # Get pot state information
            onions_in_pot_layer = obs[:, :, 16]  # Channel 16: onions in pot
            onions_in_soup_layer = obs[:, :, 18]  # Channel 18: onions in soup
            pot_cooking_time_layer = obs[:, :, 20]  # Channel 20: pot cooking time
            soup_ready_layer = obs[:, :, 21]  # Channel 21: soup done
            
            onions_in_pot = onions_in_pot_layer[pot_y, pot_x]
            onions_in_soup = onions_in_soup_layer[pot_y, pot_x]
            cooking_time = pot_cooking_time_layer[pot_y, pot_x]
            is_ready = soup_ready_layer[pot_y, pot_x]
            
            # Determine pot state
            total_onions = jnp.maximum(onions_in_pot, onions_in_soup)
            is_empty = total_onions == 0
            is_full = total_onions == 3  # 3 onions total
            is_cooking = cooking_time > 0
            is_ready = is_ready == 1
            
            return jnp.array([
                1,  # exists
                is_empty.astype(jnp.int32),
                is_full.astype(jnp.int32),
                is_cooking.astype(jnp.int32),
                is_ready.astype(jnp.int32),
                total_onions,  # num_onions
                0,  # num_tomatoes (no tomatoes in v1)
                cooking_time,
                pot_distance[0],  # dx
                pot_distance[1]   # dy
            ], dtype=jnp.int32)
        
        return jax.lax.select(is_valid, _compute_pot_features(), pot_features)
    
    def _get_wall_features(self, obs: chex.Array, agent_pos: jnp.ndarray) -> jnp.ndarray:
        """Get wall adjacency features (4 directions): [NORTH, SOUTH, EAST, WEST]."""
        height, width, channels = obs.shape
        wall_layer = obs[:, :, 11]  # Channel 11: wall locations
        
        # Check 4 directions: up, down, right, left
        directions = jnp.array([(-1, 0), (1, 0), (0, 1), (0, -1)])
        
        def check_direction(direction):
            dy, dx = direction
            check_x = agent_pos[0] + dx
            check_y = agent_pos[1] + dy
            
            in_bounds = (0 <= check_x) & (check_x < width) & (0 <= check_y) & (check_y < height)
            has_wall = jax.lax.select(in_bounds, wall_layer[check_y, check_x] == 1, True)
            
            return has_wall.astype(jnp.int32)
        
        wall_features = jax.vmap(check_direction)(directions)
        return wall_features
    
    def _featurize_agent_observation(self, obs: chex.Array) -> jnp.ndarray:
        """Convert spatial observation to feature vector for a single agent."""
        height, width, channels = obs.shape
        
        # Get agent position and orientation
        agent_pos, orientation_idx = self._get_agent_position_and_orientation(obs)
        
        # Create orientation one-hot encoding
        orientation_onehot = jnp.eye(4)[orientation_idx]
        
        # Get held object
        held_obj_idx = self._get_held_object_idx(obs)
        held_obj_onehot = jnp.eye(ObjectType.NUM_TYPES)[held_obj_idx]
        
        # Compute move area and reachability
        wall_layer = obs[:, :, 11]  # Channel 11: wall locations
        move_area = ~wall_layer.astype(jnp.bool_)
        
        # Use pre-computed enclosed spaces for reachability
        reachable_area = self.enclosed_spaces == self.enclosed_spaces[agent_pos[1], agent_pos[0]]
        reachable_area = mark_adjacent_cells(reachable_area)
        
        # Get closest feature distances using path planning with reachability
        # Onion dispenser (pile) vs loose onion
        onion_pile_layer = obs[:, :, 12]  # Channel 12: onion pile locations
        onion_layer = obs[:, :, 23]  # Channel 23: onion locations
        onion_targets = onion_pile_layer | onion_layer
        onion_distance = self._get_closest_feature_distance(agent_pos, onion_targets, move_area, orientation_idx, reachable_area, held_obj_idx, ObjectType.ONION)
        
        # Dish dispenser (pile) vs loose dish
        plate_pile_layer = obs[:, :, 14]  # Channel 14: plate pile locations
        plate_layer = obs[:, :, 22]  # Channel 22: plate locations
        plate_targets = plate_pile_layer | plate_layer
        plate_distance = self._get_closest_feature_distance(agent_pos, plate_targets, move_area, orientation_idx, reachable_area, held_obj_idx, ObjectType.PLATE)
        
        # Tomato (always zero in v1, but calculate for consistency)
        tomato_pile_layer = obs[:, :, 13]  # Channel 13: tomato pile locations (always empty)
        tomato_layer = obs[:, :, 24]  # Channel 24: tomato locations (always empty)
        tomato_targets = tomato_pile_layer | tomato_layer
        tomato_distance = self._get_closest_feature_distance(agent_pos, tomato_targets, move_area, orientation_idx, reachable_area, held_obj_idx, ObjectType.TOMATO_OR_NONE)
        
        # Soup (ready soup in pots or as dishes on counters)
        soup_ready_layer = obs[:, :, 21]  # Channel 21: soup done
        soup_distance = self._get_closest_feature_distance(agent_pos, soup_ready_layer, move_area, orientation_idx, reachable_area, held_obj_idx, ObjectType.SOUP)
        
        # Serving location
        goal_layer = obs[:, :, 15]  # Channel 15: delivery locations
        goal_distance = self._get_closest_feature_distance(agent_pos, goal_layer, move_area, orientation_idx, reachable_area, held_obj_idx, ObjectType.NO_ITEM_TARGET)
        
        # Empty counter locations
        wall_layer = obs[:, :, 11]  # Channel 11: wall locations
        pot_layer = obs[:, :, 10]  # Channel 10: pot locations
        
        # Empty counters are wall tiles that are not pots and don't have items on them
        empty_counter_mask = wall_layer & ~pot_layer & ~(onion_layer | plate_layer | soup_ready_layer)
        
        empty_counter_distance = self._get_closest_feature_distance(agent_pos, empty_counter_mask, move_area, 
                                                                    orientation_idx, reachable_area, held_obj_idx, ObjectType.NO_ITEM_TARGET)
        
        # Get soup ingredient counts (always 3 onions in v1 if any soup exists)
        soup_exists_on_grid = jnp.any(soup_ready_layer == 1)
        agent_holding_soup = held_obj_idx == ObjectType.SOUP
        any_soup_exists = soup_exists_on_grid | agent_holding_soup
        
        soup_onions = jnp.array([jax.lax.select(any_soup_exists, 3, 0)])
        soup_tomatoes = jnp.array([0])  # No tomatoes in v1
        
        # Get features for multiple pots using path planning with reachability
        all_pot_features = self._get_multiple_pot_features(obs, agent_pos, move_area, orientation_idx, reachable_area)
        
        # Get wall adjacency features
        wall_features = self._get_wall_features(obs, agent_pos)
        
        # Concatenate all features
        features = jnp.concatenate([
            orientation_onehot,  # 4 - dir_features: [NORTH, SOUTH, EAST, WEST]
            held_obj_onehot,     # 4 - inv_features: [ONION, SOUP, PLATE, TOMATO_OR_NONE]  
            onion_distance,      # 2 - onion_features: (dx, dy) to closest onion/pile
            tomato_distance,     # 2 - tomato_features: (dx, dy) to closest tomato/pile (always [0,0] in v1)
            plate_distance,      # 2 - dish_features: (dx, dy) to closest plate/pile
            soup_distance,       # 2 - soup_features: (dx, dy) to closest soup
            soup_onions,         # 1 - soup_onions: number of onions in closest soup (0 or 3)
            soup_tomatoes,       # 1 - soup_tomatoes: number of tomatoes in closest soup (always 0 in v1)
            goal_distance,       # 2 - serving_features: (dx, dy) to closest delivery location
            empty_counter_distance,  # 2 - empty_counter_features: (dx, dy) to closest empty counter
            all_pot_features,    # num_pots * 10 - pot_features: features for each of the closest pots
            wall_features        # 4 - wall_features: [NORTH, SOUTH, EAST, WEST] wall adjacency
        ]).astype(jnp.float32)
        
        return features