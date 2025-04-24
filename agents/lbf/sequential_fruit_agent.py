from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct
from jumanji.environments.routing.lbf.types import State as LBFState
from agents.lbf.base_agent import BaseAgent


class SequentialFruitAgent(BaseAgent):
    """
    Goes fruit-by-fruit in a predetermined order specified during initialization.
    Initializes the fruit sequence lazily on the first call to get_action within an episode.
    Uses random tie-breaking for moves with equal distance.

    Ordering Strategies (passed to __init__ as strings):
        'lexicographic': Sort fruits by rows first (top to bottom), then columns (left to right).
        'reverse_lexicographic': Sort fruits by rows first (bottom to top), then columns (right to left).
        'column_major': Sort fruits by columns first (left to right), then rows (top to bottom).
        'reverse_column_major': Sort fruits by columns first (right to left), then rows (bottom to top).
        'nearest_agent': Sort fruits by Manhattan distance from agent's initial position (closest first).
        'farthest_agent': Sort fruits by Manhattan distance from agent's initial position (farthest first).
    """

    @struct.dataclass
    class SeqAgentState:
        """Internal state for the SequentialFruitAgent."""
        rng_key: jax.random.PRNGKey      # For any stochasticity
        sequence: jnp.ndarray           # Sorted fruit positions (num_fruits, 2)
        idx: jnp.ndarray                # Current index in the sequence (scalar int32)
        initialized: jnp.ndarray        # Flag (scalar bool) indicating if sequence is set

    # Define valid ordering strategy strings
    VALID_ORDERING_STRATEGIES = [
        'lexicographic',
        'reverse_lexicographic',
        'column_major',
        'reverse_column_major',
        'nearest_agent',
        'farthest_agent'
    ]

    def __init__(self, agent_id: int, grid_size: int = 7, num_fruits: int = 3, ordering_strategy: str = 'lexicographic'):
        """
        Initializes the agent.

        Args:
            agent_id: The unique ID of this agent.
            grid_size: The size of the grid (assuming square).
            num_fruits: The maximum number of fruits expected (for placeholder shape).
            ordering_strategy: String selecting the fruit ordering (default: 'lexicographic').
                               See class docstring for options.
        """
        super().__init__(agent_id)
        self.grid_size = grid_size
        self.num_fruits = num_fruits
        if ordering_strategy not in self.VALID_ORDERING_STRATEGIES:
             raise ValueError(f"Invalid ordering_strategy: '{ordering_strategy}'. Must be one of {self.VALID_ORDERING_STRATEGIES}")
        self.ordering_strategy = ordering_strategy # Store the chosen strategy string
        self.init_rng = jax.random.PRNGKey(self.agent_id)
        # Initialize the state structure here
        self.initial_state = self.init(self.init_rng, self.num_fruits)


    @staticmethod
    def init(rng_key: jax.random.PRNGKey, num_fruits: int) -> 'SequentialFruitAgent.SeqAgentState':
        """
        Creates an initial, uninitialized agent state structure.

        Args:
            rng_key: A JAX random key for the agent's state.
            num_fruits: The maximum number of fruits expected (for placeholder shape).

        Returns:
            An initial SeqAgentState with initialized=False.
        """
        return SequentialFruitAgent.SeqAgentState(
            rng_key=rng_key,
            sequence=jnp.zeros((num_fruits, 2), dtype=jnp.int32),
            idx=jnp.array(0, dtype=jnp.int32),
            initialized=jnp.array(False)
        )

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
        agent_state: SeqAgentState,
    ) -> Tuple[jnp.ndarray, SeqAgentState]:
        """Calculates the agent's action for the current step."""

        # --- 0) LAZY INITIALIZATION ---
        def initialize_state_sequence(current_agent_state, current_env_state):
            """Computes the fruit sequence based on self.ordering_strategy."""
            positions = current_env_state.food_items.position # Shape (F, 2)
            agent_start_pos = current_env_state.agents.position[self.agent_id]

            # --- Select ordering based on strategy string ---
            # Use standard Python conditional logic here, as this runs only once
            # during the trace triggered by the lax.cond for initialization.
            if self.ordering_strategy == 'lexicographic':
                # Rows (top-bottom), then Cols (left-right)
                order = jnp.lexsort((positions[:, 1], positions[:, 0]))
                sorted_positions = positions[order]
            elif self.ordering_strategy == 'reverse_lexicographic':
                # Rows (bottom-top), then Cols (right-left)
                order = jnp.lexsort((-positions[:, 1], -positions[:, 0]))
                sorted_positions = positions[order]
            elif self.ordering_strategy == 'column_major':
                # Cols (left-right), then Rows (top-bottom)
                order = jnp.lexsort((positions[:, 0], positions[:, 1]))
                sorted_positions = positions[order]
            elif self.ordering_strategy == 'reverse_column_major':
                # Cols (right-left), then Rows (bottom-top)
                order = jnp.lexsort((-positions[:, 0], -positions[:, 1]))
                sorted_positions = positions[order]
            elif self.ordering_strategy == 'nearest_agent':
                # Manhattan distance from agent start (closest first)
                distances = jnp.sum(jnp.abs(positions - agent_start_pos), axis=1)
                order = jnp.argsort(distances, stable=True)
                sorted_positions = positions[order]
            elif self.ordering_strategy == 'farthest_agent':
                # Manhattan distance from agent start (farthest first)
                distances = jnp.sum(jnp.abs(positions - agent_start_pos), axis=1)
                order = jnp.argsort(-distances, stable=True)
                sorted_positions = positions[order]
            else:
                # Should not happen due to __init__ check, but good practice
                # Defaulting to lexicographic in case of unexpected state
                order = jnp.lexsort((positions[:, 1], positions[:, 0]))
                sorted_positions = positions[order]
            # --- End ordering selection ---

            # Pad sequence if fewer fruits than self.num_fruits exist
            padding_needed = self.num_fruits - sorted_positions.shape[0]
            padded_seq = jnp.pad(sorted_positions, ((0, padding_needed), (0, 0)), constant_values=-1)

            return current_agent_state.replace(
                sequence=padded_seq,
                idx=jnp.array(0, dtype=jnp.int32),
                initialized=jnp.array(True)
            )

        # Conditionally initialize if the flag is False
        agent_state = jax.lax.cond(
            agent_state.initialized,
            lambda: agent_state,
            lambda: initialize_state_sequence(agent_state, env_state)
        )
        # Now agent_state is guaranteed to be initialized

        # --- Get current state information ---
        seq = agent_state.sequence
        i = agent_state.idx
        agent_pos = env_state.agents.position[self.agent_id]
        rng_key = agent_state.rng_key # Get key from state

        # --- 1) Check target status and advance index ---
        current_target = seq[i]
        is_target_valid = jnp.all(current_target >= 0)
        food_pos = env_state.food_items.position
        eaten = env_state.food_items.eaten
        is_target_present_and_uneaten = jnp.any(
            jnp.logical_and(jnp.all(food_pos == current_target, axis=1), ~eaten)
        )
        is_target_eaten_or_gone = jnp.logical_or(~is_target_valid, ~is_target_present_and_uneaten)
        should_advance = jnp.logical_and(is_target_eaten_or_gone, is_target_valid)
        new_idx = jnp.where(should_advance, i + 1, i)
        target = seq[new_idx]
        is_new_target_valid = jnp.all(target >= 0)

        # --- 2) Check LOAD action ---
        manhattan_dist = jnp.sum(jnp.abs(agent_pos - target))
        can_load_original_target = jnp.logical_and(is_target_valid, is_target_present_and_uneaten)
        should_load = jnp.logical_and(manhattan_dist <= 1, can_load_original_target)
        action = jnp.array(0, dtype=jnp.int32)
        action = jnp.where(should_load, jnp.array(5, dtype=jnp.int32), action)

        # --- 3) If not loading and target is valid, calculate move action ---
        def calculate_move(key_in):
            all_agent_pos = env_state.agents.position
            num_agents = all_agent_pos.shape[0]
            agent_id_indices = jnp.arange(num_agents)
            obstacles = jnp.where(
                (agent_id_indices == self.agent_id)[:, None],
                jnp.array([[-1, -1]], dtype=all_agent_pos.dtype),
                all_agent_pos
            )
            obstacles = obstacles.reshape(-1, 2)
            distance_map = self._create_distance_map(target, obstacles)
            move_action_val, key_out = self._get_best_move(agent_pos, distance_map, key_in)
            return move_action_val, key_out

        def no_move(key_in):
             return jnp.array(0, dtype=jnp.int32), key_in

        should_calculate_move = jnp.logical_and(action == 0, is_new_target_valid)
        move_action, rng_key = jax.lax.cond(
            should_calculate_move,
            calculate_move,
            no_move,
            rng_key
        )

        action = jnp.where(action == 0, move_action, action)

        # --- 4) Update agent state ---
        new_agent_state = agent_state.replace(
            rng_key=rng_key,
            idx=new_idx,
        )

        return action, new_agent_state
