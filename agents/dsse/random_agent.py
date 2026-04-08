from typing import Tuple

import jax
import jax.numpy as jnp

from agents.dsse.base_agent import BaseAgent, AgentState
from envs.dsse.dsse_jax import DSSEState, NUM_ACTIONS


class RandomAgent(BaseAgent):
    """Random agent that takes uniformly random actions."""

    def __init__(self):
        super().__init__()

    def _get_action(
        self,
        obs: jnp.ndarray,
        env_state: DSSEState,
        agent_state: AgentState,
        rng: jax.random.PRNGKey,
    ) -> Tuple[int, AgentState]:
        action = jax.random.randint(rng, (), 0, NUM_ACTIONS)
        return action, agent_state
