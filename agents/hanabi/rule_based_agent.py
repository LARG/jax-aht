"""Simple rule-based Hanabi agent with parameterized play/hint/discard priority.

Four strategies: cautious (hint >> discard >> play), aggressive (play first),
communicative (hint >> play >> discard), frugal (discard >> hint >> play).
"""
from typing import Tuple

import jax
import jax.numpy as jnp

from agents.hanabi.base_agent import BaseAgent, AgentState

# (play_weight, discard_weight, hint_weight): used as logits, so soft priority
STRATEGY_WEIGHTS = {
    "cautious":       (1.0,   10.0,  1000.0),   # hint >> discard >> play
    "aggressive":     (1000.0, 1.0,  10.0),      # play >> hint >> discard
    "communicative":  (10.0,   1.0,  1000.0),    # hint >> play >> discard
    "frugal":         (1.0,   1000.0, 10.0),      # discard >> hint >> play
}

VALID_STRATEGIES = tuple(STRATEGY_WEIGHTS.keys())


class RuleBasedAgent(BaseAgent):
    """Weighted-category sampling over play/discard/hint actions."""

    def __init__(
        self,
        strategy: str = "cautious",
        hand_size: int = 5,
        num_colors: int = 5,
        num_ranks: int = 5,
        num_actions: int = 21,
        **kwargs,
    ):
        super().__init__(num_actions=num_actions, **kwargs)
        if strategy not in STRATEGY_WEIGHTS:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Valid strategies: {VALID_STRATEGIES}"
            )
        self.strategy = strategy
        self.hand_size = hand_size
        self.num_colors = num_colors
        self.num_ranks = num_ranks

        play_w, discard_w, hint_w = STRATEGY_WEIGHTS[strategy]
        self.play_weight = play_w
        self.discard_weight = discard_w
        self.hint_weight = hint_w

        # per-action weight vector, concrete at trace time
        weights = jnp.zeros(num_actions)
        discard_end = hand_size
        play_end = 2 * hand_size
        hint_end = play_end + num_colors + num_ranks
        weights = weights.at[:discard_end].set(discard_w)
        weights = weights.at[discard_end:play_end].set(play_w)
        weights = weights.at[play_end:hint_end].set(hint_w)
        self._logits = jnp.log(weights + 1e-10)

    def _get_action(
        self,
        obs: jnp.ndarray,
        env_state,
        avail_mask: jnp.ndarray,
        agent_state: AgentState,
        rng: jax.random.PRNGKey,
    ) -> Tuple[int, AgentState]:
        masked_logits = jnp.where(avail_mask > 0, self._logits, -1e9)
        action = jax.random.categorical(rng, masked_logits)
        return action, agent_state
