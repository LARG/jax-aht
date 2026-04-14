"""IGGI agent (Walton-Rivers 2017). Uses card knowledge and partner hand visibility.

Priority: play safe > hint playable > discard useless > discard oldest > hint random.
"""
from typing import Tuple

import jax
import jax.numpy as jnp

from agents.hanabi.base_agent import BaseAgent, AgentState


class IGGIAgent(BaseAgent):

    def __init__(
        self,
        hand_size: int = 5,
        num_colors: int = 5,
        num_ranks: int = 5,
        num_actions: int = 21,
        **kwargs,
    ):
        super().__init__(num_actions=num_actions, **kwargs)
        self.hand_size = hand_size
        self.num_colors = num_colors
        self.num_ranks = num_ranks

        # JaxMARL action encoding: [0,h) discard, [h,2h) play, [2h,2h+c) hint color, etc.
        # Note: discard is FIRST, not play. Got this wrong initially.
        self.discard_start = 0
        self.discard_end = hand_size
        self.play_start = hand_size
        self.play_end = 2 * hand_size
        self.hint_color_start = 2 * hand_size
        self.hint_color_end = 2 * hand_size + num_colors
        self.hint_rank_start = 2 * hand_size + num_colors
        self.hint_rank_end = 2 * hand_size + num_colors + num_ranks

    def _is_certainly_playable(self, my_knowledge_2d, next_playable_rank):
        """Card is certainly playable if ALL its possibilities are playable."""
        rank_idx = jnp.arange(self.num_ranks)
        playable_matrix = (rank_idx[None, :] == next_playable_rank[:, None]).astype(jnp.float32)

        # playable if no non-playable possibilities remain
        non_playable_possibilities = my_knowledge_2d * (1.0 - playable_matrix[None, :, :])
        has_any_possibility = my_knowledge_2d.sum(axis=(1, 2)) > 0
        is_playable = (non_playable_possibilities.sum(axis=(1, 2)) == 0) & has_any_possibility
        return is_playable

    def _is_certainly_useless(self, my_knowledge_2d, next_playable_rank):
        """Card is certainly useless if ALL its possibilities are already completed."""
        rank_idx = jnp.arange(self.num_ranks)
        completed_matrix = (rank_idx[None, :] < next_playable_rank[:, None]).astype(jnp.float32)

        # useless if no useful possibilities remain
        useful_possibilities = my_knowledge_2d * (1.0 - completed_matrix[None, :, :])
        has_any_possibility = my_knowledge_2d.sum(axis=(1, 2)) > 0
        is_useless = (useful_possibilities.sum(axis=(1, 2)) == 0) & has_any_possibility
        return is_useless

    def _find_playable_hint_mask(self, partner_hand, next_playable_rank, avail_mask):
        """Mask of hint actions that touch a playable card in partner's hand."""
        partner_colors = jnp.argmax(partner_hand.sum(axis=2), axis=1)  # (hand_size,)
        partner_ranks = jnp.argmax(partner_hand.sum(axis=1), axis=1)  # (hand_size,)

        partner_playable = (
            partner_ranks == next_playable_rank[partner_colors]
        ).astype(jnp.float32)  # (hand_size,)

        playable_cards = partner_hand * partner_playable[:, None, None]
        colors_with_playable = (playable_cards.sum(axis=(0, 2)) > 0).astype(jnp.float32)
        ranks_with_playable = (playable_cards.sum(axis=(0, 1)) > 0).astype(jnp.float32)
        hint_mask = jnp.zeros(self.num_actions)
        hint_mask = hint_mask.at[self.hint_color_start:self.hint_color_end].set(colors_with_playable)
        hint_mask = hint_mask.at[self.hint_rank_start:self.hint_rank_end].set(ranks_with_playable)
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

        # decode game state
        fireworks = env_state.fireworks  # (num_colors, num_ranks) one-hot
        next_playable_rank = jnp.sum(fireworks, axis=1).astype(jnp.int32)  # (num_colors,)

        my_knowledge = env_state.card_knowledge[my_id]  # (hand_size, 25)
        my_knowledge_2d = my_knowledge.reshape(
            self.hand_size, self.num_colors, self.num_ranks
        )

        partner_hand = env_state.player_hands[partner_id]  # (hand_size, colors, ranks)
        info_tokens = jnp.sum(env_state.info_tokens)  # scalar int

        # 1. PlaySafeCard
        is_playable = self._is_certainly_playable(my_knowledge_2d, next_playable_rank)
        safe_play_mask = jnp.zeros(self.num_actions)
        safe_play_mask = safe_play_mask.at[self.play_start:self.play_end].set(
            is_playable.astype(jnp.float32)
        )
        safe_play_mask = safe_play_mask * avail_mask
        has_safe_play = safe_play_mask.sum() > 0

        # 2. TellPlayable
        hint_playable_mask = self._find_playable_hint_mask(
            partner_hand, next_playable_rank, avail_mask
        )
        can_hint_playable = (hint_playable_mask.sum() > 0) & (info_tokens > 0)

        # 3. OsawaDiscard
        is_useless = self._is_certainly_useless(my_knowledge_2d, next_playable_rank)
        useless_discard_mask = jnp.zeros(self.num_actions)
        useless_discard_mask = useless_discard_mask.at[self.discard_start:self.discard_end].set(
            is_useless.astype(jnp.float32)
        )
        useless_discard_mask = useless_discard_mask * avail_mask
        has_useless_discard = useless_discard_mask.sum() > 0

        # 4. DiscardOldest (slot 0)
        oldest_discard_mask = jnp.zeros(self.num_actions)
        oldest_discard_mask = oldest_discard_mask.at[self.discard_start].set(1.0)
        oldest_discard_mask = oldest_discard_mask * avail_mask
        has_oldest_discard = oldest_discard_mask.sum() > 0

        # 5. TellRandom
        random_hint_mask = jnp.zeros(self.num_actions)
        random_hint_mask = random_hint_mask.at[self.hint_color_start:self.hint_rank_end].set(1.0)
        random_hint_mask = random_hint_mask * avail_mask
        has_random_hint = (random_hint_mask.sum() > 0) & (info_tokens > 0)

        # priority cascade
        # TODO: refactor this nested jnp.where into a table-driven lookup.
        # It's readable once you know the rule order but it's a mess to
        # extend, since the Piers/VDB/Outer variants all copy this pattern.
        action_logits = jnp.where(
            has_safe_play,
            jnp.where(safe_play_mask > 0, 0.0, -1e9),
            jnp.where(
                can_hint_playable,
                jnp.where(hint_playable_mask > 0, 0.0, -1e9),
                jnp.where(
                    has_useless_discard,
                    jnp.where(useless_discard_mask > 0, 0.0, -1e9),
                    jnp.where(
                        has_oldest_discard,
                        jnp.where(oldest_discard_mask > 0, 0.0, -1e9),
                        jnp.where(
                            has_random_hint,
                            jnp.where(random_hint_mask > 0, 0.0, -1e9),
                            # 6. DiscardRandomly (canonical terminal rule)
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
        )

        action = jax.random.categorical(rng, action_logits)
        return action, agent_state
