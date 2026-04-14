"""Flawed agent: IGGI + configurable random-mistake probability.

mistake_prob=0 is pure IGGI, =1 is pure random. Good for creating a
spectrum of partner competence in held-out eval.
"""
from typing import Tuple

import jax
import jax.numpy as jnp

from agents.hanabi.iggi_agent import IGGIAgent
from agents.hanabi.base_agent import AgentState


class FlawedAgent(IGGIAgent):

    def __init__(self, mistake_prob: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.mistake_prob = mistake_prob

    def _get_action(
        self,
        obs: jnp.ndarray,
        env_state,
        avail_mask: jnp.ndarray,
        agent_state: AgentState,
        rng: jax.random.PRNGKey,
    ) -> Tuple[int, AgentState]:
        rng, mistake_rng, iggi_rng, random_rng = jax.random.split(rng, 4)

        iggi_action, _ = super()._get_action(
            obs, env_state, avail_mask, agent_state, iggi_rng
        )

        random_logits = jnp.where(avail_mask > 0, 0.0, -1e9)
        random_action = jax.random.categorical(random_rng, random_logits)

        # coin flip: IGGI or random?
        make_mistake = jax.random.uniform(mistake_rng) < self.mistake_prob
        action = jnp.where(make_mistake, random_action, iggi_action)

        return action, agent_state
