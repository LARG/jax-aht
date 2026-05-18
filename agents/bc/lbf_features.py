from __future__ import annotations

import jax
import jax.numpy as jnp


def _scatter_obstacles(
    grid_size: int,
    positions: jnp.ndarray,
    valid: jnp.ndarray,
) -> jnp.ndarray:
    batch_size = positions.shape[0]
    mask = jnp.zeros((batch_size, grid_size, grid_size), dtype=bool)
    clipped = jnp.clip(positions, 0, grid_size - 1).astype(jnp.int32)
    batch_idx = jnp.arange(batch_size)
    for i in range(positions.shape[1]):
        pos = clipped[:, i]
        mask = mask.at[batch_idx, pos[:, 0], pos[:, 1]].set(valid[:, i])
    return mask


def _distance_map(
    grid_size: int,
    starts: jnp.ndarray,
    obstacles: jnp.ndarray,
    obstacle_valid: jnp.ndarray,
) -> jnp.ndarray:
    batch_size = starts.shape[0]
    start_valid = jnp.all((starts >= 0) & (starts < grid_size), axis=-1)
    start_clipped = jnp.clip(starts, 0, grid_size - 1).astype(jnp.int32)
    batch_idx = jnp.arange(batch_size)

    obstacle_mask = _scatter_obstacles(grid_size, obstacles, obstacle_valid)
    obstacle_mask = obstacle_mask.at[
        batch_idx, start_clipped[:, 0], start_clipped[:, 1]
    ].set(False)

    dist = jnp.full((batch_size, grid_size, grid_size), jnp.inf, dtype=jnp.float32)
    dist = dist.at[batch_idx, start_clipped[:, 0], start_clipped[:, 1]].set(
        jnp.where(start_valid, 0.0, jnp.inf)
    )
    dist = jnp.where(obstacle_mask, jnp.inf, dist)

    def body_fn(_, current):
        padded = jnp.pad(
            current,
            ((0, 0), (1, 1), (1, 1)),
            constant_values=jnp.inf,
        )
        up = padded[:, :-2, 1:-1]
        down = padded[:, 2:, 1:-1]
        left = padded[:, 1:-1, :-2]
        right = padded[:, 1:-1, 2:]
        neighbor = jnp.minimum(jnp.minimum(up, down), jnp.minimum(left, right)) + 1.0
        updated = jnp.minimum(current, neighbor)
        updated = jnp.where(obstacle_mask, jnp.inf, updated)
        return updated.at[batch_idx, start_clipped[:, 0], start_clipped[:, 1]].set(
            jnp.where(start_valid, 0.0, jnp.inf)
        )

    return jax.lax.fori_loop(0, grid_size * grid_size, body_fn, dist)


def _min_adjacent_dist(
    dist_map: jnp.ndarray,
    food_pos: jnp.ndarray,
    food_alive: jnp.ndarray,
    grid_size: int,
) -> jnp.ndarray:
    batch_size, num_food, _ = food_pos.shape
    offsets = jnp.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=jnp.float32)
    targets = food_pos[:, :, None, :] + offsets
    valid = jnp.all((targets >= 0) & (targets < grid_size), axis=-1)
    clipped = jnp.clip(targets, 0, grid_size - 1).astype(jnp.int32)

    batch_idx = jnp.arange(batch_size)[:, None, None]
    gathered = dist_map[
        batch_idx,
        clipped[..., 0],
        clipped[..., 1],
    ]
    gathered = jnp.where(valid, gathered, jnp.inf)
    best = jnp.min(gathered, axis=-1)
    missing = jnp.full((batch_size, num_food), 2.0, dtype=jnp.float32)
    return jnp.where(
        food_alive,
        jnp.where(jnp.isfinite(best), best / grid_size, 2.0),
        missing,
    )


def augment_lbf_obs(obs: jnp.ndarray, grid_size: int, num_food: int) -> jnp.ndarray:
    original_shape = obs.shape[:-1]
    flat_obs = obs.reshape((-1, obs.shape[-1])).astype(jnp.float32)

    food_raw = flat_obs[:, : 3 * num_food].reshape((-1, num_food, 3))
    self_raw = flat_obs[:, 3 * num_food : 3 * num_food + 3]
    teammate_raw = flat_obs[:, 3 * num_food + 3 : 3 * num_food + 6]

    food_pos = food_raw[..., :2]
    food_level = food_raw[..., 2:3]
    self_pos = self_raw[..., :2]
    teammate_pos = teammate_raw[..., :2]
    self_level = self_raw[..., 2:3]
    teammate_level = teammate_raw[..., 2:3]

    food_alive = jnp.all(food_pos >= 0, axis=-1)
    max_coord = jnp.maximum(float(grid_size - 1), 1.0)
    max_path = float(grid_size)
    level_scale = float(grid_size)

    self_to_food = food_pos - self_pos[:, None, :]
    teammate_to_food = food_pos - teammate_pos[:, None, :]
    self_manhattan = jnp.sum(jnp.abs(self_to_food), axis=-1)
    teammate_manhattan = jnp.sum(jnp.abs(teammate_to_food), axis=-1)
    teammate_delta = (teammate_pos - self_pos) / max_coord
    teammate_distance = jnp.sum(
        jnp.abs(teammate_pos - self_pos),
        axis=-1,
        keepdims=True,
    )

    food_obstacles = food_pos
    food_obstacle_valid = food_alive
    self_obstacles = jnp.concatenate([food_obstacles, teammate_pos[:, None, :]], axis=1)
    self_obstacle_valid = jnp.concatenate(
        [food_obstacle_valid, jnp.ones((flat_obs.shape[0], 1), dtype=bool)],
        axis=1,
    )
    teammate_obstacles = jnp.concatenate([food_obstacles, self_pos[:, None, :]], axis=1)
    teammate_obstacle_valid = jnp.concatenate(
        [food_obstacle_valid, jnp.ones((flat_obs.shape[0], 1), dtype=bool)],
        axis=1,
    )

    self_dist_map = _distance_map(grid_size, self_pos, self_obstacles, self_obstacle_valid)
    teammate_dist_map = _distance_map(
        grid_size, teammate_pos, teammate_obstacles, teammate_obstacle_valid
    )
    self_path_to_food = _min_adjacent_dist(self_dist_map, food_pos, food_alive, grid_size)
    teammate_path_to_food = _min_adjacent_dist(
        teammate_dist_map, food_pos, food_alive, grid_size
    )
    self_can_load = (
        (self_manhattan == 1)
        & (self_level[:, 0:1] >= food_level[..., 0])
        & food_alive
    )
    pair_can_load = (
        ((self_level[:, 0:1] + teammate_level[:, 0:1]) >= food_level[..., 0])
        & food_alive
    )

    food_features = jnp.concatenate(
        [
            food_alive[..., None].astype(jnp.float32),
            jnp.where(food_alive[..., None], food_pos / max_coord, 0.0),
            food_level / level_scale,
            jnp.where(food_alive[..., None], self_to_food / max_coord, 0.0),
            jnp.where(food_alive[..., None], teammate_to_food / max_coord, 0.0),
            jnp.where(food_alive, self_manhattan / max_path, 2.0)[..., None],
            jnp.where(food_alive, teammate_manhattan / max_path, 2.0)[..., None],
            self_path_to_food[..., None],
            teammate_path_to_food[..., None],
            ((self_manhattan == 1) & food_alive)[..., None].astype(jnp.float32),
            ((teammate_manhattan == 1) & food_alive)[..., None].astype(jnp.float32),
            self_can_load[..., None].astype(jnp.float32),
            pair_can_load[..., None].astype(jnp.float32),
        ],
        axis=-1,
    ).reshape((flat_obs.shape[0], -1))

    global_features = jnp.concatenate(
        [
            self_pos / max_coord,
            teammate_pos / max_coord,
            self_level / level_scale,
            teammate_level / level_scale,
            teammate_delta,
            teammate_distance / max_path,
            jnp.sum(food_alive, axis=-1, keepdims=True) / float(num_food),
        ],
        axis=-1,
    )

    augmented = jnp.concatenate([flat_obs, global_features, food_features], axis=-1)
    return augmented.reshape((*original_shape, augmented.shape[-1]))
