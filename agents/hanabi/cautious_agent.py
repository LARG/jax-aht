import jax
import jax.numpy as jnp
from agents.hanabi.iggi_agent import IGGIAgent
from agents.hanabi.base_agent import AgentState
from typing import Tuple


class CautiousAgent(IGGIAgent):
    def _get_action(
        self,
        obs: jnp.ndarray,
        env_state,
        avail_mask: jnp.ndarray,
        agent_state: AgentState,
        rng: jax.random.PRNGKey,
    ) -> Tuple[int, AgentState]:

        my_id = agent_state.agent_id
        partner_id = 1 - my_id

        fireworks = env_state.fireworks
        next_playable_rank = jnp.sum(fireworks, axis=1).astype(jnp.int32)

        my_knowledge = env_state.card_knowledge[my_id]
        my_knowledge_2d = my_knowledge.reshape(
            self.hand_size, self.num_colors, self.num_ranks
        )

        partner_hand = env_state.player_hands[partner_id]
        info_tokens = jnp.sum(env_state.info_tokens)

        is_playable = self._is_certainly_playable(my_knowledge_2d, next_playable_rank)
        safe_play_mask = jnp.zeros(self.num_actions)
        safe_play_mask = safe_play_mask.at[self.play_start:self.play_end].set(
            is_playable.astype(jnp.float32)
        )
        safe_play_mask = safe_play_mask * avail_mask
        has_safe_play = safe_play_mask.sum() > 0

        hint_playable_mask = self._find_playable_hint_mask(
            partner_hand, next_playable_rank, avail_mask
        )
        can_hint_playable = (hint_playable_mask.sum() > 0) & (info_tokens > 0)

        is_useless = self._is_certainly_useless(my_knowledge_2d, next_playable_rank, env_state)
        useless_discard_mask = jnp.zeros(self.num_actions)
        useless_discard_mask = useless_discard_mask.at[self.discard_start:self.discard_end].set(
            is_useless.astype(jnp.float32)
        )
        useless_discard_mask = useless_discard_mask * avail_mask
        has_useless_discard = useless_discard_mask.sum() > 0

        random_discard_mask = self._random_discard_mask(avail_mask)
        has_random_discard = random_discard_mask.sum() > 0

        random_hint_mask = jnp.zeros(self.num_actions)
        random_hint_mask = random_hint_mask.at[self.hint_color_start:self.hint_rank_end].set(1.0)
        random_hint_mask = random_hint_mask * avail_mask
        has_random_hint = (random_hint_mask.sum() > 0) & (info_tokens > 0)

        m2l = self._mask_to_logits
        rules = [
            (has_safe_play,       m2l(safe_play_mask)),
            (can_hint_playable,   m2l(hint_playable_mask)),
            (has_useless_discard, m2l(useless_discard_mask)),
            (has_random_discard,  m2l(random_discard_mask)),
            (has_random_hint,     m2l(random_hint_mask)),
        ]
        action = self._select_priority_action(rules, avail_mask, rng)
        return action, agent_state
