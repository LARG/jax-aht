"""Greedy search agent: moves toward the highest-probability cell, then searches."""

from typing import Tuple

import jax
import jax.numpy as jnp

from agents.dsse.base_agent import BaseAgent, AgentState
from envs.dsse.dsse_jax import DSSEState, SEARCH, NUM_ACTIONS


class GreedySearchAgent(BaseAgent):
    """Moves toward the cell with highest probability, searches when on it.

    Observation layout: [x_norm, y_norm, prob_matrix_flat, other_positions_norm]
    The probability matrix starts at index 2 and has grid_size^2 elements.
    """

    def __init__(self, grid_size: int = 15):
        super().__init__()
        self.grid_size = grid_size

    def _get_action(
        self,
        obs: jnp.ndarray,
        env_state: DSSEState,
        agent_state: AgentState,
        rng: jax.random.PRNGKey,
    ) -> Tuple[int, AgentState]:
        gs = self.grid_size
        # Extract position and probability matrix from obs
        pos_x = jnp.round(obs[0] * gs).astype(jnp.int32)
        pos_y = jnp.round(obs[1] * gs).astype(jnp.int32)
        prob_flat = obs[2 : 2 + gs * gs]
        prob_matrix = prob_flat.reshape(gs, gs)

        # Find the cell with highest probability
        best_idx = jnp.argmax(prob_flat)
        best_x = best_idx % gs
        best_y = best_idx // gs

        # Compute direction to target
        dx = jnp.sign(best_x - pos_x)
        dy = jnp.sign(best_y - pos_y)

        # Map (dx, dy) to action
        # At target: SEARCH
        at_target = (dx == 0) & (dy == 0)
        # Movement: LEFT=0, RIGHT=1, UP=2, DOWN=3, UL=4, UR=5, DL=6, DR=7
        action = jnp.where(
            at_target, SEARCH,
            jnp.where(dx == -1,
                jnp.where(dy == -1, 4,  # UP_LEFT
                jnp.where(dy == 1, 6,   # DOWN_LEFT
                0)),                     # LEFT
            jnp.where(dx == 1,
                jnp.where(dy == -1, 5,  # UP_RIGHT
                jnp.where(dy == 1, 7,   # DOWN_RIGHT
                1)),                     # RIGHT
            jnp.where(dy == -1, 2,       # UP
                3)))                      # DOWN
        )

        new_agent_state = AgentState(
            agent_id=agent_state.agent_id,
            step_count=agent_state.step_count + 1,
        )
        return action, new_agent_state
