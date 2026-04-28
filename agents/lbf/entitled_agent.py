from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct
from jumanji.environments.routing.lbf.types import State as LBFState
from agents.lbf.base_agent import BaseAgent


class EntitledAgent(BaseAgent):
    """
    An agent that waits for its teammate to position adjacent to a target fruit
    before moving to collect it together.

    For the closest uneaten fruit, the agent designates two adjacent spots (either
    N/S or E/W of the fruit) as candidate positions for the teammate. N/S is
    preferred when both spots are in bounds; otherwise E/W is used.

    Behavior:
      - If the teammate is at one of the two candidate spots, the entitled agent
        moves toward the other spot and loads when adjacent to the fruit.
      - Otherwise, the entitled agent does not move (action = 0 / NOOP).
    """

    @struct.dataclass
    class EntitledAgentState:
        """Internal state for the EntitledAgent."""
        agent_id: int
        orientations: jnp.ndarray  # (num_fruits,) bool: True = N/S, False = E/W
        initialized: jnp.ndarray   # scalar bool

    def __init__(self, grid_size: int = 7, num_fruits: int = 3):
        super().__init__()
        self.grid_size = grid_size
        self.num_fruits = num_fruits
        self._jit_key = (type(self).__name__, grid_size, num_fruits)

    def __hash__(self):
        return hash(self._jit_key)

    def __eq__(self, other):
        return isinstance(other, EntitledAgent) and self._jit_key == other._jit_key

    def get_name(self):
        return "EntitledAgent"

    def init_agent_state(self, agent_id: int) -> 'EntitledAgent.EntitledAgentState':
        return EntitledAgent.EntitledAgentState(
            agent_id=agent_id,
            orientations=jnp.zeros(self.num_fruits, dtype=bool),
            initialized=jnp.array(False),
        )

    def _create_distance_map(
        self,
        target: jnp.ndarray,
        obstacles: jnp.ndarray,
    ) -> jnp.ndarray:
        """Creates a BFS distance map from target to all positions, avoiding obstacles."""
        grid = jnp.full((self.grid_size, self.grid_size), jnp.inf, dtype=jnp.float32)
        grid = jax.lax.cond(
            jnp.all(jnp.logical_and(target >= 0, target < self.grid_size)),
            lambda g: g.at[target[0], target[1]].set(0.0),
            lambda g: g,
            grid
        )

        obstacle_mask = jnp.zeros_like(grid, dtype=bool)
        obstacles = jnp.atleast_2d(obstacles)
        if obstacles.shape[0] > 0 and obstacles.shape[-1] == 2:
            valid_obstacles_mask = jnp.all((obstacles >= 0) & (obstacles < self.grid_size), axis=1)

            def update_obstacle_mask(carry_mask, i):
                is_valid = valid_obstacles_mask[i]
                pos = obstacles[i]
                new_mask = jax.lax.cond(
                    is_valid,
                    lambda m: m.at[pos[0], pos[1]].set(True),
                    lambda m: m,
                    carry_mask
                )
                return new_mask, None

            obstacle_mask = jax.lax.fori_loop(
                0, obstacles.shape[0],
                lambda i, current_mask: update_obstacle_mask(current_mask, i)[0],
                obstacle_mask
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
            new_grid = jax.lax.cond(
                jnp.all(jnp.logical_and(target >= 0, target < self.grid_size)),
                lambda g: g.at[target[0], target[1]].set(0.0),
                lambda g: g,
                new_grid
            )
            return new_grid

        return jax.lax.fori_loop(0, max_iterations, body_fn, grid)

    def _get_best_move(
        self,
        agent_pos: jnp.ndarray,
        distance_map: jnp.ndarray,
        rng_key: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, jax.random.PRNGKey]:
        """Finds the best move action (1-4) using random tie-breaking."""
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

        key, subkey = jax.random.split(rng_key)
        noise = jax.random.uniform(subkey, shape=(4,), maxval=1e-5)
        min_mask = (neighbor_dists == min_neighbor_dist) & jnp.isfinite(neighbor_dists)
        noisy_dists = jnp.where(min_mask, neighbor_dists + noise, neighbor_dists)
        best_neighbor_action_idx = jnp.argmin(noisy_dists)

        best_action = jnp.where(
            jnp.logical_and(min_neighbor_dist < current_dist, jnp.isfinite(min_neighbor_dist)),
            actions[best_neighbor_action_idx],
            jnp.array(0, dtype=jnp.int32)
        )

        return jnp.array(best_action, dtype=jnp.int32), key

    def _get_action(
        self,
        obs: jnp.ndarray,
        env_state: LBFState,
        agent_state: 'EntitledAgent.EntitledAgentState',
        rng: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, 'EntitledAgent.EntitledAgentState']:

        # Lazily initialize per-fruit orientations on the first step of each episode.
        # True = use N/S spots, False = use E/W spots. Chosen randomly and held fixed.
        agent_state = jax.lax.cond(
            agent_state.initialized,
            lambda: agent_state,
            lambda: agent_state.replace(
                orientations=jax.random.bernoulli(rng, shape=(self.num_fruits,)),
                initialized=jnp.array(True),
            ),
        )

        agent_pos = env_state.agents.position[agent_state.agent_id]
        teammate_pos = env_state.agents.position[1 - agent_state.agent_id]

        food_positions = env_state.food_items.position  # (F, 2)
        food_eaten = env_state.food_items.eaten          # (F,)

        # For every fruit simultaneously, compute N/S and E/W candidate spots.
        # Use the per-fruit orientation stored in agent_state, with a bounds-based
        # fallback: if the chosen orientation has a spot out of bounds, use the other.
        rs = food_positions[:, 0]  # (F,)
        cs = food_positions[:, 1]  # (F,)

        ns_spot0s = jnp.stack([rs - 1, cs], axis=1)  # (F, 2) — North
        ns_spot1s = jnp.stack([rs + 1, cs], axis=1)  # (F, 2) — South
        ew_spot0s = jnp.stack([rs, cs - 1], axis=1)  # (F, 2) — West
        ew_spot1s = jnp.stack([rs, cs + 1], axis=1)  # (F, 2) — East

        ns_both_valid = jnp.logical_and(rs - 1 >= 0, rs + 1 < self.grid_size)  # (F,)
        ew_both_valid = jnp.logical_and(cs - 1 >= 0, cs + 1 < self.grid_size)  # (F,)

        # Use the randomly chosen orientation; fall back to the other if out of bounds
        want_ns = agent_state.orientations  # (F,) bool
        use_ns = jnp.logical_or(
            jnp.logical_and(want_ns, ns_both_valid),   # wanted N/S and it's valid
            jnp.logical_and(~want_ns, ~ew_both_valid), # wanted E/W but it's invalid, so use N/S
        )  # (F,)
        spot0s = jnp.where(use_ns[:, None], ns_spot0s, ew_spot0s)  # (F, 2)
        spot1s = jnp.where(use_ns[:, None], ns_spot1s, ew_spot1s)  # (F, 2)

        # Check if the teammate is at either candidate spot for each fruit
        teammate_at_spot0s = jnp.all(teammate_pos[None, :] == spot0s, axis=1)  # (F,)
        teammate_at_spot1s = jnp.all(teammate_pos[None, :] == spot1s, axis=1)  # (F,)

        # A fruit is "active" if it is uneaten and the teammate is at one of its spots
        active = jnp.logical_and(
            jnp.logical_or(teammate_at_spot0s, teammate_at_spot1s),
            food_eaten == 0
        )  # (F,)
        any_active = jnp.any(active)

        # Among active fruits, pick the one closest to the entitled agent
        dists = jnp.sum(jnp.abs(food_positions - agent_pos), axis=1)  # (F,)
        dists_active = jnp.where(active, dists, jnp.inf)
        best_idx = jnp.argmin(dists_active)

        fruit_pos = food_positions[best_idx]
        spot0 = spot0s[best_idx]
        spot1 = spot1s[best_idx]
        teammate_at_spot0 = teammate_at_spot0s[best_idx]

        # Target the spot the teammate is NOT occupying
        my_target = jnp.where(teammate_at_spot0, spot1, spot0)

        # Load when adjacent to the chosen fruit
        adjacent_to_fruit = jnp.sum(jnp.abs(agent_pos - fruit_pos)) <= 1
        should_load = jnp.logical_and(any_active, adjacent_to_fruit)

        # Move toward my_target when not yet loading
        should_move = jnp.logical_and(any_active, ~should_load)

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
            distance_map = self._create_distance_map(my_target, obstacles)
            move_action_val, key_out = self._get_best_move(agent_pos, distance_map, key_in)
            return move_action_val, key_out

        def no_move(key_in):
            return jnp.array(0, dtype=jnp.int32), key_in

        move_action, _ = jax.lax.cond(
            should_move,
            calculate_move,
            no_move,
            rng
        )

        action = jnp.where(action == 0, move_action, action)
        return action, agent_state  # agent_state carries the initialized orientations
