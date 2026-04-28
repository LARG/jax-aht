"""Outer agent (Walton-Rivers 2017). Hints before discarding.

Canonical priority:
  1. PlaySafeCard
  2. OsawaDiscard (discard certainly-useless; frees info tokens for hinting)
  3. TellAnyoneAboutPlayableCard (hint about a playable card)
  4. TellUnknown (hint about anything the partner doesn't fully know yet)
  5. DiscardRandomly

The earlier version here skipped rule 2 and rule 4, so Outer was
effectively "play > hint playable > hint random > discard random",
which burned info tokens fast and scored 0/25 on full Hanabi. Adding
OsawaDiscard and TellUnknown should bring it closer to the paper's
numbers.
"""
from typing import Tuple

import jax
import jax.numpy as jnp

from agents.hanabi.iggi_agent import IGGIAgent
from agents.hanabi.base_agent import AgentState


class OuterAgent(IGGIAgent):
    """play > osawa > hint playable > tell unknown > discard random"""

    def _find_tell_unknown_mask(self, partner_hand, partner_knowledge_2d, avail_mask):
        """Hints that reveal new information about the partner's cards.

        Any color or rank the partner doesn't fully know yet in at least
        one of their cards. Color hint is valid if some partner card has
        that color AND the partner's belief over colors for that card is
        not yet narrowed to one color.
        """
        partner_colors = jnp.argmax(partner_hand.sum(axis=2), axis=1)  # (hand_size,)
        partner_ranks = jnp.argmax(partner_hand.sum(axis=1), axis=1)  # (hand_size,)

        # For each card, is the color "known" (narrowed to 1)?
        color_possibilities = partner_knowledge_2d.sum(axis=2)  # (hand_size, num_colors)
        n_colors_possible = (color_possibilities > 0).sum(axis=1)  # (hand_size,)
        color_unknown_per_card = n_colors_possible > 1  # (hand_size,)

        rank_possibilities = partner_knowledge_2d.sum(axis=1)  # (hand_size, num_ranks)
        n_ranks_possible = (rank_possibilities > 0).sum(axis=1)  # (hand_size,)
        rank_unknown_per_card = n_ranks_possible > 1  # (hand_size,)

        # Color hint c is informative if some card actually has color c
        # AND that card's color isn't yet fully known.
        colors_with_unknown = jnp.zeros(self.num_colors)
        for c in range(self.num_colors):
            has_color = (partner_colors == c)
            informative = (has_color & color_unknown_per_card).any()
            colors_with_unknown = colors_with_unknown.at[c].set(
                informative.astype(jnp.float32)
            )

        ranks_with_unknown = jnp.zeros(self.num_ranks)
        for r in range(self.num_ranks):
            has_rank = (partner_ranks == r)
            informative = (has_rank & rank_unknown_per_card).any()
            ranks_with_unknown = ranks_with_unknown.at[r].set(
                informative.astype(jnp.float32)
            )

        hint_mask = jnp.zeros(self.num_actions)
        hint_mask = hint_mask.at[self.hint_color_start:self.hint_color_end].set(colors_with_unknown)
        hint_mask = hint_mask.at[self.hint_rank_start:self.hint_rank_end].set(ranks_with_unknown)
        return hint_mask * avail_mask

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

        partner_knowledge = env_state.card_knowledge[partner_id]
        partner_knowledge_2d = partner_knowledge.reshape(
            self.hand_size, self.num_colors, self.num_ranks
        )

        partner_hand = env_state.player_hands[partner_id]
        info_tokens = jnp.sum(env_state.info_tokens)

        # 1. PlaySafeCard
        is_playable = self._is_certainly_playable(my_knowledge_2d, next_playable_rank)
        safe_play_mask = jnp.zeros(self.num_actions)
        safe_play_mask = safe_play_mask.at[self.play_start:self.play_end].set(
            is_playable.astype(jnp.float32)
        )
        safe_play_mask = safe_play_mask * avail_mask
        has_safe_play = safe_play_mask.sum() > 0

        # 2. OsawaDiscard (canonical rule 2, missing from the old impl)
        is_useless = self._is_certainly_useless(my_knowledge_2d, next_playable_rank)
        useless_discard_mask = jnp.zeros(self.num_actions)
        useless_discard_mask = useless_discard_mask.at[self.discard_start:self.discard_end].set(
            is_useless.astype(jnp.float32)
        )
        useless_discard_mask = useless_discard_mask * avail_mask
        has_useless_discard = useless_discard_mask.sum() > 0

        # 3. TellPlayable
        hint_playable_mask = self._find_playable_hint_mask(
            partner_hand, next_playable_rank, avail_mask
        )
        can_hint_playable = (hint_playable_mask.sum() > 0) & (info_tokens > 0)

        # 4. TellUnknown (canonical rule 4, hint anything the partner doesn't know yet)
        tell_unknown_mask = self._find_tell_unknown_mask(
            partner_hand, partner_knowledge_2d, avail_mask
        )
        can_tell_unknown = (tell_unknown_mask.sum() > 0) & (info_tokens > 0)

        # 5. DiscardRandomly
        discard_only_mask = jnp.zeros(self.num_actions)
        discard_only_mask = discard_only_mask.at[self.discard_start:self.discard_end].set(1.0)
        discard_only_mask = discard_only_mask * avail_mask

        action_logits = jnp.where(
            has_safe_play,
            jnp.where(safe_play_mask > 0, 0.0, -1e9),
            jnp.where(
                has_useless_discard,
                jnp.where(useless_discard_mask > 0, 0.0, -1e9),
                jnp.where(
                    can_hint_playable,
                    jnp.where(hint_playable_mask > 0, 0.0, -1e9),
                    jnp.where(
                        can_tell_unknown,
                        jnp.where(tell_unknown_mask > 0, 0.0, -1e9),
                        # DiscardRandomly (terminal rule; restrict to discard range)
                        jnp.where(
                            (jnp.arange(self.num_actions) >= self.discard_start) &
                            (jnp.arange(self.num_actions) < self.discard_end) &
                            (avail_mask > 0),
                            0.0, -1e9
                        )
                    )
                )
            )
        )

        action = jax.random.categorical(rng, action_logits)
        return action, agent_state
