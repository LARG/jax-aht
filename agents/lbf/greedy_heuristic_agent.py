from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct
from jumanji.environments.routing.lbf.types import State as LBFState
from agents.lbf.base_agent import BaseAgent

class GreedyHeuristicAgent(BaseAgent):
    """
    Goes greedily to the some fruit based on some condition

    Conditions (passed to __init__ as strings):
        'closest_self' - goes to the closest fruit
        'closest_avg' - goes to the fruit closest to the average position of all agents
        'closest_level' - goes to the closest fruit that the agent can consume by itself
        'closest_combined_level' - goes to the closest fruit that the agent can consume by itself or with help from other agents
    """

    @struct.dataclass
    class GreedyAgentState:
        """Internal state for the GreedyHeuristicAgent."""
        agent_id: int                   # The unique ID of this agent.

    VALID_HEURISTICS = [
        'closest_self',
        # 'closest_avg', 
        # 'closest_level', 
        # 'closest_combined_level',
    ]

    def __init__(self, grid_size: int = 7, num_fruits: int = 3, heuristic: str = 'closest_self'):

        super().__init__()
        self.grid_size = grid_size
        self.num_fruits = num_fruits
        if heuristic not in self.VALID_HEURISTICS:
             raise ValueError(f"Invalid heuristic: '{heuristic}'. Must be one of {self.VALID_HEURISTICS}")
        self.heuristic = heuristic # Store the chosen heuristic string
  
    def init_agent_state(self, agent_id: int):
        return GreedyHeuristicAgent.GreedyAgentState(agent_id=agent_id)

    def _create_distance_map(
        self,
        target: jnp.ndarray,
        obstacles: jnp.ndarray,
    ) -> jnp.ndarray:
        """Creates a distance map from target to all positions, avoiding obstacles.

        Args:
            target: Target position (row, col)
            obstacles: Array of potential obstacle positions (N, 2).

        Returns:
            Distance map grid.
        """
        grid = jnp.full((self.grid_size, self.grid_size), jnp.inf, dtype=jnp.float32)
        grid = jax.lax.cond(
            jnp.all(jnp.logical_and(target >= 0, target < self.grid_size)),
            lambda g: g.at[target[0], target[1]].set(0.0),
            lambda g: g,
            grid
        )

        obstacle_mask = jnp.zeros_like(grid, dtype=bool)
        obstacles = jnp.atleast_2d(obstacles)
        # Check if obstacles array has content before proceeding
        if obstacles.shape[0] > 0 and obstacles.shape[-1] == 2:
            valid_obstacles_mask = jnp.all((obstacles >= 0) & (obstacles < self.grid_size), axis=1)
            def update_obstacle_mask(carry_mask, i):
                is_valid = valid_obstacles_mask[i]
                pos = obstacles[i]
                # Conditionally set the mask to True at pos if the obstacle is valid
                new_mask = jax.lax.cond(
                    is_valid,
                    lambda m: m.at[pos[0], pos[1]].set(True),
                    lambda m: m, # No change if invalid
                    carry_mask
                )
                return new_mask, None # Return updated mask and None carry for scan

            # Iterate through potential obstacles and update the mask conditionally
            obstacle_mask = jax.lax.fori_loop(
                 0, obstacles.shape[0], lambda i, current_mask: update_obstacle_mask(current_mask, i)[0], obstacle_mask
            )

        grid = jnp.where(obstacle_mask, jnp.inf, grid)

        max_iterations = self.grid_size * self.grid_size
        def body_fn(i, current_grid):
            padded_grid = jnp.pad(current_grid, 1, constant_values=jnp.inf)
            up    = padded_grid[:-2, 1:-1]
            down  = padded_grid[2:, 1:-1]
            left  = padded_grid[1:-1, :-2]
            right = padded_grid[1:-1, 2:]
            min_neighbor_dist = jnp.minimum(jnp.minimum(up, down), jnp.minimum(left, right)) + 1
            new_grid = jnp.minimum(current_grid, min_neighbor_dist)
            new_grid = jnp.where(obstacle_mask, jnp.inf, new_grid)
            # Ensure target is only set if it was within bounds initially
            new_grid = jax.lax.cond(
                 jnp.all(jnp.logical_and(target >= 0, target < self.grid_size)),
                 lambda g: g.at[target[0], target[1]].set(0.0),
                 lambda g: g,
                 new_grid
            )
            return new_grid
        final_grid = jax.lax.fori_loop(0, max_iterations, body_fn, grid)
        return final_grid

    def _get_best_move(
        self,
        agent_pos: jnp.ndarray,
        distance_map: jnp.ndarray,
        rng_key: jax.random.PRNGKey # Added rng_key for tie-breaking
    ) -> Tuple[jnp.ndarray, jax.random.PRNGKey]: # Return action and new key
        """Finds the best move action (1-4) using random tie-breaking.

        Args:
            agent_pos: Current position (row, col)
            distance_map: Distance map from target
            rng_key: JAX random key for tie-breaking.

        Returns:
            Tuple: (Action index (0-4) as jnp.ndarray, new_rng_key)
        """
        r, c = agent_pos
        current_dist = jax.lax.select(
            jnp.all((agent_pos >= 0) & (agent_pos < self.grid_size)),
            distance_map[r, c],
            jnp.inf
        )

        actions = jnp.array([1, 2, 3, 4], dtype=jnp.int32)
        neighbor_coords = jnp.array([
            [r - 1, c], [r + 1, c], [r, c - 1], [r, c + 1]
        ])
        neighbor_dists = jnp.full(4, jnp.inf)

        def update_dist(index, coords, dists):
            return jax.lax.cond(
                jnp.all((coords >= 0) & (coords < self.grid_size)),
                lambda d: d.at[index].set(distance_map[coords[0], coords[1]]),
                lambda d: d,
                dists
            )
        neighbor_dists = update_dist(0, neighbor_coords[0], neighbor_dists)
        neighbor_dists = update_dist(1, neighbor_coords[1], neighbor_dists)
        neighbor_dists = update_dist(2, neighbor_coords[2], neighbor_dists)
        neighbor_dists = update_dist(3, neighbor_coords[3], neighbor_dists)

        min_neighbor_dist = jnp.min(neighbor_dists)

        # --- Random Tie-breaking ---
        key, subkey = jax.random.split(rng_key)
        noise = jax.random.uniform(subkey, shape=(4,), maxval=1e-5)
        min_mask = (neighbor_dists == min_neighbor_dist) & jnp.isfinite(neighbor_dists)
        noisy_dists = jnp.where(min_mask, neighbor_dists + noise, neighbor_dists)
        best_neighbor_action_idx = jnp.argmin(noisy_dists)
        # --- End Random Tie-breaking ---

        best_action = jnp.where(
            jnp.logical_and(min_neighbor_dist < current_dist, jnp.isfinite(min_neighbor_dist)),
            actions[best_neighbor_action_idx],
            jnp.array(0, dtype=jnp.int32)
        )

        return jnp.array(best_action, dtype=jnp.int32), key # Return action and the new key

    def _get_action(
        self,
        obs: jnp.ndarray,
        env_state: LBFState,
        agent_state: GreedyAgentState,
        rng: jax.random.PRNGKey
    ) -> Tuple[jnp.ndarray, GreedyAgentState]:

        agent_pos = env_state.agents.position[agent_state.agent_id]

        target = self.closest_self(
            agent_pos, 
            env_state.food_items
        )

        # check if we can load the target fruit
        manhattan_dist = jnp.sum(jnp.abs(agent_pos - target))
        should_load = manhattan_dist <= 1
        action = jnp.array(0, dtype=jnp.int32)
        action = jnp.where(should_load, jnp.array(5, dtype=jnp.int32), action)

        def calculate_move(key_in):
            all_agent_pos = env_state.agents.position
            num_agents = all_agent_pos.shape[0]
            agent_id_indices = jnp.arange(num_agents)
            obstacles = jnp.where(
                (agent_id_indices == agent_state.agent_id)[:, None],
                jnp.array([[-1, -1]], dtype=all_agent_pos.dtype),
                all_agent_pos
            )
            obstacles = obstacles.reshape(-1, 2)
            distance_map = self._create_distance_map(target, obstacles)
            move_action_val, key_out = self._get_best_move(agent_pos, distance_map, key_in)
            return move_action_val, key_out

        def no_move(key_in):
            return jnp.array(0, dtype=jnp.int32), key_in

        move_action, rng_key = jax.lax.cond(
            should_load,
            no_move, # if we should load, we don't move
            calculate_move, # otherwise we want to move
            rng
        )

        action = jnp.where(action == 0, move_action, action)
        return action, agent_state

    def closest_self(self, agent_pos: jnp.ndarray, food_items):
        """Return the index of the closest fruit to the agent."""
        
        positions = food_items.position # Shape (F, 2)
        eaten = food_items.eaten # Shape (F,)

        distances = jnp.sum(jnp.abs(positions - agent_pos), axis=1)
        distances = jnp.where(eaten == 0, distances, jnp.inf)

        # return the location of the closest fruit
        closest_idx = jnp.argmin(distances)
        return positions[closest_idx]
    