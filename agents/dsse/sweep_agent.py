"""Sweep agent: systematically sweeps the grid in a boustrophedon (zigzag) pattern."""

from typing import Tuple

import jax
import jax.numpy as jnp

from agents.dsse.base_agent import BaseAgent, AgentState
from envs.dsse.dsse_jax import DSSEState, SEARCH, RIGHT, LEFT, DOWN


class SweepAgent(BaseAgent):
    """Sweeps the grid row by row, searching at each cell.

    Pattern: move right across a row, step down, move left, step down, repeat.
    Searches at every cell visited. This provides a baseline coverage strategy.
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
        step = agent_state.step_count

        # Alternate: search, then move. 2 actions per cell.
        is_search_step = (step % 2 == 0)

        # Which cell in the sweep pattern are we heading to?
        cell_idx = step // 2
        row = cell_idx // gs
        col_in_row = cell_idx % gs
        # Boustrophedon: even rows go right, odd rows go left
        is_even_row = (row % 2 == 0)

        # Determine movement direction
        at_row_end = jnp.where(is_even_row, col_in_row == gs - 1, col_in_row == gs - 1)
        # Within a row: move right (even) or left (odd)
        row_action = jnp.where(is_even_row, RIGHT, LEFT)
        # At end of row: move down
        move_action = jnp.where(at_row_end & (col_in_row == 0), DOWN, row_action)

        action = jnp.where(is_search_step, SEARCH, move_action)

        new_agent_state = AgentState(
            agent_id=agent_state.agent_id,
            step_count=agent_state.step_count + 1,
        )
        return action, new_agent_state
