from typing import Tuple

import jax
import jax.numpy as jnp

from agents.hanabi.base_agent import BaseAgent, AgentState


class RandomAgent(BaseAgent):

    def __init__(self, num_actions: int = 20, **kwargs):
        super().__init__(num_actions=num_actions, **kwargs)

    def _get_action(
        self,
        obs: jnp.ndarray,
        env_state,
        avail_mask: jnp.ndarray,
        agent_state: AgentState,
        rng: jax.random.PRNGKey,
    ) -> Tuple[int, AgentState]:
        # uniform sample over legal actions via Gumbel trick
        logits = jnp.where(avail_mask > 0, 0.0, -1e9)
        action = jax.random.categorical(rng, logits)
        return action, agent_state
