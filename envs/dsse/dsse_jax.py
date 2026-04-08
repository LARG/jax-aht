"""JAX reimplementation of the Drone Swarm Search Environment.

N drones search a grid_size x grid_size map for K targets using 9 discrete actions
(8 moves plus SEARCH). A rescue requires n_drones_to_rescue drones to
SEARCH the same cell in the same step; the POD roll applies after the
coordination check.
"""

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import chex
from flax.struct import dataclass


# Action constants
LEFT, RIGHT, UP, DOWN = 0, 1, 2, 3
UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT = 4, 5, 6, 7
SEARCH = 8
NUM_ACTIONS = 9

# Movement deltas: (dx, dy) for actions 0-7. Action 8 (SEARCH) = no movement.
MOVE_DX = jnp.array([-1, 1, 0, 0, -1, 1, -1, 1, 0], dtype=jnp.int32)
MOVE_DY = jnp.array([0, 0, -1, 1, -1, -1, 1, 1, 0], dtype=jnp.int32)

# 3x3 neighborhood offsets for target movement: (dy, dx) in row-major order
# Index 0=(y-1,x-1), 1=(y-1,x), 2=(y-1,x+1), 3=(y,x-1), 4=(y,x), ...
NEIGHBOR_DY = jnp.array([-1, -1, -1, 0, 0, 0, 1, 1, 1], dtype=jnp.int32)
NEIGHBOR_DX = jnp.array([-1, 0, 1, -1, 0, 1, -1, 0, 1], dtype=jnp.int32)


@dataclass
class DSSEState:
    """Internal environment state (all JAX arrays, JIT-compatible)."""
    drone_positions: chex.Array       # (n_drones, 2) int32 - (x, y)
    target_positions: chex.Array      # (n_targets, 2) int32 - (x, y)
    targets_found: chex.Array         # (n_targets,) bool
    target_detection_mult: chex.Array # (n_targets,) float32 - per-target detection multiplier
    target_inc_x: chex.Array          # (n_targets,) float32 - fractional x movement accumulator
    target_inc_y: chex.Array          # (n_targets,) float32 - fractional y movement accumulator
    prob_matrix: chex.Array           # (grid_size, grid_size) float32
    prob_center_x: chex.Array         # float32 - Gaussian center x (continuous)
    prob_center_y: chex.Array         # float32 - Gaussian center y (continuous)
    prob_inc_x: chex.Array            # float32 - fractional center x accumulator
    prob_inc_y: chex.Array            # float32 - fractional center y accumulator
    dispersion: chex.Array            # float32 - Gaussian sigma
    timestep: chex.Array              # int32


class DSSEJax:
    """JAX DSSE backend. reset / step / get_obs are jit-safe."""

    def __init__(
        self,
        grid_size: int = 15,
        n_drones: int = 4,
        n_targets: int = 2,
        timestep_limit: int = 100,
        probability_of_detection: float = 0.9,
        vector_x: float = 1.1,
        vector_y: float = 1.0,
        dispersion_start: float = 0.5,
        dispersion_inc: float = 0.1,
        cell_size: float = 130.0,
        drone_speed: float = 10.0,
        disaster_position: Tuple[int, int] = None,
        target_cluster_radius: int = 2,
        n_drones_to_rescue: int = 1,
        leave_grid_penalty: float = 0.0,
        pre_render_steps: int = 0,
        share_rewards: bool = True,
        # Legacy params (mapped to vector_x/y for backward compat)
        drift_x: float = None,
        drift_y: float = None,
        **kwargs,
    ):
        self.grid_size = grid_size
        self.n_drones = n_drones
        self.n_targets = n_targets
        self.timestep_limit = timestep_limit
        self.pod = probability_of_detection
        self.dispersion_start = dispersion_start
        self.dispersion_inc = dispersion_inc
        self.n_drones_to_rescue = n_drones_to_rescue
        self.leave_grid_penalty = leave_grid_penalty
        self.pre_render_steps = pre_render_steps
        self.share_rewards = share_rewards
        self.target_cluster_radius = target_cluster_radius

        # Drift vector and time step relation (matching original physics)
        if drift_x is not None:
            # Legacy: direct drift per step
            self.vector_x = drift_x
            self.vector_y = drift_y if drift_y is not None else drift_x
            self.time_step_relation = 1.0
        else:
            self.vector_x = vector_x
            self.vector_y = vector_y
            # time_step_relation = cell_size / (drone_speed - wind_resistance)
            # Original: cell_size=130, drone_speed=10 → 13.0 seconds per step
            # Then for vector magnitude: sqrt(1.1^2 + 1.0^2) = 1.487
            # time_step_relation = cell_size / (magnitude * time_step) = 130 / (1.487 * 13) = 6.72
            time_step = cell_size / drone_speed
            vector_magnitude = jnp.sqrt(vector_x**2 + vector_y**2)
            self.time_step_relation = float(
                cell_size / (vector_magnitude * time_step + 1e-8)
            )

        # Disaster position (center of target clustering and probability)
        if disaster_position is None:
            self.disaster_x = grid_size // 2
            self.disaster_y = grid_size // 2
        else:
            self.disaster_x = disaster_position[0]
            self.disaster_y = disaster_position[1]

        # Precompute grid coordinate meshes for Gaussian calculation
        xs = jnp.arange(grid_size, dtype=jnp.float32)
        self._grid_x, self._grid_y = jnp.meshgrid(xs, xs)  # (gs, gs)

        # Precompute "other agent" indices for observations
        all_idx = jnp.arange(n_drones)
        self._other_idx = jnp.array([
            jnp.concatenate([all_idx[:i], all_idx[i+1:]])
            for i in range(n_drones)
        ])  # (n_drones, n_drones-1)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: chex.PRNGKey) -> DSSEState:
        """Reset environment to initial state."""
        rng_drones, rng_targets, rng_pre = jax.random.split(rng, 3)

        # Place drones in a row along top edge (matching original default_drones_positions)
        drone_x = jnp.arange(self.n_drones, dtype=jnp.int32) % self.grid_size
        drone_y = jnp.arange(self.n_drones, dtype=jnp.int32) // self.grid_size
        drone_positions = jnp.stack([drone_x, drone_y], axis=-1)

        # Place targets clustered near disaster position (within target_cluster_radius)
        target_positions = self._cluster_targets(rng_targets)

        # Initial probability matrix centered on disaster position
        center_x = jnp.float32(self.disaster_x)
        center_y = jnp.float32(self.disaster_y)
        dispersion = jnp.float32(self.dispersion_start)
        prob_matrix = self._compute_prob_matrix(center_x, center_y, dispersion)

        state = DSSEState(
            drone_positions=drone_positions,
            target_positions=target_positions,
            targets_found=jnp.zeros(self.n_targets, dtype=jnp.bool_),
            target_detection_mult=jnp.ones(self.n_targets, dtype=jnp.float32),
            target_inc_x=jnp.zeros(self.n_targets, dtype=jnp.float32),
            target_inc_y=jnp.zeros(self.n_targets, dtype=jnp.float32),
            prob_matrix=prob_matrix,
            prob_center_x=center_x,
            prob_center_y=center_y,
            prob_inc_x=jnp.float32(0.0),
            prob_inc_y=jnp.float32(0.0),
            dispersion=dispersion,
            timestep=jnp.int32(0),
        )

        # Pre-render simulation: evolve probability matrix and targets before agents start
        def _pre_step(carry, _):
            st, rng = carry
            rng, rng_t = jax.random.split(rng)
            st = self._evolve_world(rng_t, st)
            return (st, rng), None

        (state, _), _ = jax.lax.scan(
            _pre_step, (state, rng_pre), None, length=self.pre_render_steps,
        )

        return state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, rng: chex.PRNGKey, state: DSSEState, actions: chex.Array,
    ) -> Tuple[DSSEState, chex.Array, chex.Array, dict]:
        """Execute one step."""
        rng_det, rng_world = jax.random.split(rng)

        # --- Move drones ---
        is_search = (actions == SEARCH)
        dx = MOVE_DX[actions]
        dy = MOVE_DY[actions]
        new_x = state.drone_positions[:, 0] + dx
        new_y = state.drone_positions[:, 1] + dy

        # Boundary check
        valid = (
            (new_x >= 0) & (new_x < self.grid_size) &
            (new_y >= 0) & (new_y < self.grid_size)
        )
        new_x = jnp.where(valid, new_x, state.drone_positions[:, 0])
        new_y = jnp.where(valid, new_y, state.drone_positions[:, 1])
        new_positions = jnp.stack([new_x, new_y], axis=-1)

        # Boundary penalty (for drones that tried to leave)
        boundary_penalty = jnp.where(
            ~valid & ~is_search, self.leave_grid_penalty, 0.0,
        )

        # --- Check for target detection ---
        # (n_drones, n_targets) match matrices
        drone_at_target_x = (new_positions[:, 0:1] == state.target_positions[None, :, 0])
        drone_at_target_y = (new_positions[:, 1:2] == state.target_positions[None, :, 1])
        drone_at_target = drone_at_target_x & drone_at_target_y

        drone_searching = is_search[:, None]
        target_not_found = ~state.targets_found[None, :]
        can_detect = drone_at_target & drone_searching & target_not_found

        # Count how many drones are searching at each target's cell
        # n_drones_to_rescue > 1 creates a "stag hunt" requiring joint action
        drones_searching_per_target = can_detect.astype(jnp.int32).sum(axis=0)  # (n_targets,)
        enough_drones = drones_searching_per_target >= self.n_drones_to_rescue  # (n_targets,)

        # Per-target detection probability: min(mult * pod, 1.0)
        det_prob = jnp.minimum(
            state.target_detection_mult * self.pod, 1.0,
        )  # (n_targets,)

        det_rolls = jax.random.uniform(rng_det, shape=(self.n_targets,))
        detected_per_target = enough_drones & (det_rolls < det_prob)  # (n_targets,)

        newly_found = detected_per_target
        new_targets_found = state.targets_found | newly_found

        # --- Rewards ---
        time_decay = 1.0 - (state.timestep.astype(jnp.float32) / self.timestep_limit)
        n_newly_found = newly_found.astype(jnp.float32).sum()
        team_reward = n_newly_found * (1.0 + time_decay)

        if self.share_rewards:
            rewards = jnp.full(self.n_drones, team_reward / self.n_drones) + boundary_penalty
        else:
            # Per-drone: reward goes to drones that participated in a rescue
            drone_participated = (can_detect & newly_found[None, :]).any(axis=1).astype(jnp.float32)
            rewards = drone_participated * (1.0 + time_decay) + boundary_penalty

        # --- Evolve world (probability matrix + target movement) ---
        state_evolved = self._evolve_world(rng_world, state)

        # --- Done ---
        new_timestep = state.timestep + 1
        all_found = new_targets_found.all()
        time_up = new_timestep >= self.timestep_limit
        done = all_found | time_up

        new_state = DSSEState(
            drone_positions=new_positions,
            target_positions=state_evolved.target_positions,
            targets_found=new_targets_found,
            target_detection_mult=state.target_detection_mult,
            target_inc_x=state_evolved.target_inc_x,
            target_inc_y=state_evolved.target_inc_y,
            prob_matrix=state_evolved.prob_matrix,
            prob_center_x=state_evolved.prob_center_x,
            prob_center_y=state_evolved.prob_center_y,
            prob_inc_x=state_evolved.prob_inc_x,
            prob_inc_y=state_evolved.prob_inc_y,
            dispersion=state_evolved.dispersion,
            timestep=new_timestep,
        )

        info = {
            "targets_found": new_targets_found.sum(),
            "all_found": all_found,
        }

        return new_state, rewards, done, info

    def _evolve_world(self, rng: chex.PRNGKey, state: DSSEState) -> DSSEState:
        """Evolve probability matrix and move targets (no drone interaction)."""
        rng_targets = rng

        # Update probability matrix center (fractional accumulation) 
        new_prob_inc_x = state.prob_inc_x + self.vector_x / self.time_step_relation
        new_prob_inc_y = state.prob_inc_y + self.vector_y / self.time_step_relation

        # Move center by integer amount when accumulated >= 1
        move_cx = jnp.sign(new_prob_inc_x) * (jnp.abs(new_prob_inc_x) >= 1.0)
        move_cy = jnp.sign(new_prob_inc_y) * (jnp.abs(new_prob_inc_y) >= 1.0)

        new_center_x = state.prob_center_x + move_cx
        new_center_y = state.prob_center_y + move_cy

        # Subtract integer part from accumulator
        new_prob_inc_x = new_prob_inc_x - move_cx
        new_prob_inc_y = new_prob_inc_y - move_cy

        # Clip center to grid
        new_center_x = jnp.clip(new_center_x, 0.0, self.grid_size - 1.0)
        new_center_y = jnp.clip(new_center_y, 0.0, self.grid_size - 1.0)

        # Update dispersion and recompute probability matrix
        new_dispersion = state.dispersion + self.dispersion_inc
        new_prob_matrix = self._compute_prob_matrix(new_center_x, new_center_y, new_dispersion)

        # --- Move targets (stochastic, probability-weighted) ---
        new_target_inc_x = state.target_inc_x + self.vector_x / self.time_step_relation
        new_target_inc_y = state.target_inc_y + self.vector_y / self.time_step_relation

        # Check which targets should move this step
        should_move = (jnp.abs(new_target_inc_x) >= 1.0) | (jnp.abs(new_target_inc_y) >= 1.0)

        # Move each target using probability-weighted 3x3 neighborhood sampling
        new_target_positions, new_target_inc_x, new_target_inc_y = self._move_targets(
            rng_targets, state.target_positions,
            new_target_inc_x, new_target_inc_y,
            should_move, state.targets_found, new_prob_matrix,
        )

        return state.replace(
            target_positions=new_target_positions,
            target_inc_x=new_target_inc_x,
            target_inc_y=new_target_inc_y,
            prob_matrix=new_prob_matrix,
            prob_center_x=new_center_x,
            prob_center_y=new_center_y,
            prob_inc_x=new_prob_inc_x,
            prob_inc_y=new_prob_inc_y,
            dispersion=new_dispersion,
        )

    def _move_targets(
        self, rng, positions, inc_x, inc_y, should_move, found, prob_matrix,
    ):
        """Move targets using probability-weighted 3x3 neighborhood."""
        gs = self.grid_size

        def _move_single(carry, target_idx):
            rng = carry
            rng, rng_move = jax.random.split(rng)

            x = positions[target_idx, 0]
            y = positions[target_idx, 1]
            do_move = should_move[target_idx] & ~found[target_idx]

            # Extract 3x3 neighborhood from probability matrix using padded indexing
            # Pad the probability matrix with zeros on all sides
            padded = jnp.pad(prob_matrix, 1, mode='constant', constant_values=0.0)
            # Extract 3x3 slice (offset by 1 due to padding)
            slice_3x3 = jax.lax.dynamic_slice(
                padded,
                (y.astype(jnp.int32), x.astype(jnp.int32)),  # y+1-1=y, x+1-1=x in padded
                (3, 3),
            ).flatten()  # (9,)

            # Sample direction from categorical distribution.
            # The 1e-10 floor avoids log(0) on cells the heuristic
            # has assigned zero density to. The floor is small enough
            # that any nonzero cell still dominates the categorical;
            # we have run this at grid_size up to 15 with ndr=2 and
            # seen no NaNs in either training or held-out evaluation.
            logits = jnp.log(slice_3x3 + 1e-10)
            sampled_idx = jax.random.categorical(rng_move, logits)

            new_x = x + NEIGHBOR_DX[sampled_idx]
            new_y = y + NEIGHBOR_DY[sampled_idx]

            # Clip to grid bounds
            new_x = jnp.clip(new_x, 0, gs - 1)
            new_y = jnp.clip(new_y, 0, gs - 1)

            # Only apply movement if should_move
            final_x = jnp.where(do_move, new_x, x)
            final_y = jnp.where(do_move, new_y, y)

            # Reset increment accumulators after movement
            new_ix = jnp.where(do_move, inc_x[target_idx] - jnp.sign(inc_x[target_idx]), inc_x[target_idx])
            new_iy = jnp.where(do_move, inc_y[target_idx] - jnp.sign(inc_y[target_idx]), inc_y[target_idx])

            return rng, (final_x, final_y, new_ix, new_iy)

        rng, results = jax.lax.scan(
            _move_single, rng, jnp.arange(self.n_targets),
        )
        new_xs, new_ys, new_inc_xs, new_inc_ys = results

        new_positions = jnp.stack([new_xs, new_ys], axis=-1).astype(jnp.int32)
        return new_positions, new_inc_xs, new_inc_ys

    def _cluster_targets(self, rng: chex.PRNGKey) -> chex.Array:
        """Place targets clustered near disaster position."""
        gs = self.grid_size
        r = self.target_cluster_radius

        # Generate random offsets within cluster radius
        rng_dx, rng_dy = jax.random.split(rng)
        offsets_x = jax.random.randint(rng_dx, (self.n_targets,), -r, r + 1)
        offsets_y = jax.random.randint(rng_dy, (self.n_targets,), -r, r + 1)

        target_x = jnp.clip(self.disaster_x + offsets_x, 0, gs - 1).astype(jnp.int32)
        target_y = jnp.clip(self.disaster_y + offsets_y, 0, gs - 1).astype(jnp.int32)

        return jnp.stack([target_x, target_y], axis=-1)

    @partial(jax.jit, static_argnums=(0,))
    def get_obs(self, state: DSSEState) -> chex.Array:
        """Get observations for all drones.

        Per-drone: [x_norm, y_norm, prob_matrix_flat, other_positions_norm]
        """
        gs = jnp.float32(self.grid_size)
        pos_norm = state.drone_positions.astype(jnp.float32) / gs
        prob_flat = state.prob_matrix.flatten()

        def _build_obs(drone_idx):
            own_pos = pos_norm[drone_idx]
            others = self._other_idx[drone_idx]
            other_pos = pos_norm[others].flatten()
            return jnp.concatenate([own_pos, prob_flat, other_pos])

        obs = jax.vmap(_build_obs)(jnp.arange(self.n_drones))
        return obs

    @property
    def obs_size(self) -> int:
        return 2 + self.grid_size * self.grid_size + (self.n_drones - 1) * 2

    def _compute_prob_matrix(
        self, center_x: chex.Array, center_y: chex.Array, sigma: chex.Array,
    ) -> chex.Array:
        """Compute normalized 2D Gaussian probability matrix."""
        sigma_sq = sigma * sigma + 1e-6
        exponent = -(
            (self._grid_x - center_x) ** 2 + (self._grid_y - center_y) ** 2
        ) / (2.0 * sigma_sq)
        prob = jnp.exp(exponent)
        total = prob.sum()
        prob = jnp.where(total > 0, prob / total, prob)
        return prob
