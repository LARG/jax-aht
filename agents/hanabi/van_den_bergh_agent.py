"""Van Den Bergh agent (Walton-Rivers 2017).

Canonical priority from the paper:
  1. If lives > 1: PlayProbablySafe(0.6). Else: PlaySafeCard.
  2. DiscardProbablyUseless(1.0): certain useless cards first
  3. TellAnyoneAboutUsefulCard
  4. TellAnyoneAboutUselessCard (TellDispensable)
  5. TellMostInformation (hint that reveals max new info)
  6. DiscardProbablyUseless(0.0): probabilistic fallback

Old implementation was missing the probabilistic play (rule 1's
PlayProbablySafe) and TellMostInformation (rule 5), and had the order
of TellPlayable and OsawaDiscard swapped. Fixed here.
"""
from typing import Tuple

import jax
import jax.numpy as jnp

from agents.hanabi.iggi_agent import IGGIAgent
from agents.hanabi.base_agent import AgentState


class VanDenBerghAgent(IGGIAgent):

    def __init__(self, play_threshold: float = 0.6, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.play_threshold = play_threshold

    def _probability_playable(self, my_knowledge_2d, next_playable_rank):
        """Per-card probability of being playable = playable_poss / total_poss."""
        rank_idx = jnp.arange(self.num_ranks)
        playable_matrix = (rank_idx[None, :] == next_playable_rank[:, None]).astype(jnp.float32)
        playable_possibilities = (my_knowledge_2d * playable_matrix[None, :, :]).sum(axis=(1, 2))
        total_possibilities = my_knowledge_2d.sum(axis=(1, 2))
        prob = playable_possibilities / jnp.maximum(total_possibilities, 1e-8)
        return prob

    def _probability_useless(self, my_knowledge_2d, next_playable_rank):
        """Per-card probability of being useless = completed_poss / total_poss."""
        rank_idx = jnp.arange(self.num_ranks)
        completed_matrix = (rank_idx[None, :] < next_playable_rank[:, None]).astype(jnp.float32)
        useless_possibilities = (my_knowledge_2d * completed_matrix[None, :, :]).sum(axis=(1, 2))
        total_possibilities = my_knowledge_2d.sum(axis=(1, 2))
        prob = useless_possibilities / jnp.maximum(total_possibilities, 1e-8)
        return prob

    def _find_dispensable_hint_mask(self, partner_hand, next_playable_rank, avail_mask):
        partner_colors = jnp.argmax(partner_hand.sum(axis=2), axis=1)
        partner_ranks = jnp.argmax(partner_hand.sum(axis=1), axis=1)
        partner_dispensable = (
            partner_ranks < next_playable_rank[partner_colors]
        ).astype(jnp.float32)
        dispensable_cards = partner_hand * partner_dispensable[:, None, None]
        colors_with_dispensable = (dispensable_cards.sum(axis=(0, 2)) > 0).astype(jnp.float32)
        ranks_with_dispensable = (dispensable_cards.sum(axis=(0, 1)) > 0).astype(jnp.float32)
        hint_mask = jnp.zeros(self.num_actions)
        hint_mask = hint_mask.at[self.hint_color_start:self.hint_color_end].set(colors_with_dispensable)
        hint_mask = hint_mask.at[self.hint_rank_start:self.hint_rank_end].set(ranks_with_dispensable)
        return hint_mask * avail_mask

    def _find_most_informative_hint_mask(self, partner_hand, partner_knowledge_2d, avail_mask):
        """Hint that reveals max number of new facts about partner's cards.

        Score each possible hint by how many partner cards it newly identifies.
        A color hint for color c "reveals" cards whose belief has c among
        multiple possibilities and whose actual color is c.
        """
        partner_colors = jnp.argmax(partner_hand.sum(axis=2), axis=1)  # (hand_size,)
        partner_ranks = jnp.argmax(partner_hand.sum(axis=1), axis=1)

        color_poss = partner_knowledge_2d.sum(axis=2)  # (hand_size, num_colors)
        n_colors_possible = (color_poss > 0).sum(axis=1)
        color_unknown = n_colors_possible > 1

        rank_poss = partner_knowledge_2d.sum(axis=1)
        n_ranks_possible = (rank_poss > 0).sum(axis=1)
        rank_unknown = n_ranks_possible > 1

        # Count cards each color hint would reveal
        color_scores = jnp.zeros(self.num_colors)
        for c in range(self.num_colors):
            count = ((partner_colors == c) & color_unknown).sum().astype(jnp.float32)
            color_scores = color_scores.at[c].set(count)

        rank_scores = jnp.zeros(self.num_ranks)
        for r in range(self.num_ranks):
            count = ((partner_ranks == r) & rank_unknown).sum().astype(jnp.float32)
            rank_scores = rank_scores.at[r].set(count)

        hint_mask = jnp.zeros(self.num_actions)
        hint_mask = hint_mask.at[self.hint_color_start:self.hint_color_end].set(color_scores)
        hint_mask = hint_mask.at[self.hint_rank_start:self.hint_rank_end].set(rank_scores)
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
        life_tokens = jnp.sum(env_state.life_tokens)

        # 1. PlayProbablySafe(0.6) if lives > 1, else PlaySafeCard
        prob_playable = self._probability_playable(my_knowledge_2d, next_playable_rank)
        is_certainly_playable = self._is_certainly_playable(my_knowledge_2d, next_playable_rank)
        # When we have extra lives, accept any card with p(playable) >= threshold.
        # Otherwise, only play certain ones.
        can_gamble = life_tokens > 1
        play_cond = jnp.where(
            can_gamble,
            prob_playable >= self.play_threshold,
            is_certainly_playable,
        )
        safe_play_mask = jnp.zeros(self.num_actions)
        safe_play_mask = safe_play_mask.at[self.play_start:self.play_end].set(
            play_cond.astype(jnp.float32)
        )
        safe_play_mask = safe_play_mask * avail_mask
        has_safe_play = safe_play_mask.sum() > 0

        # 2. DiscardProbablyUseless(1.0) = OsawaDiscard
        is_useless = self._is_certainly_useless(my_knowledge_2d, next_playable_rank)
        useless_discard_mask = jnp.zeros(self.num_actions)
        useless_discard_mask = useless_discard_mask.at[self.discard_start:self.discard_end].set(
            is_useless.astype(jnp.float32)
        )
        useless_discard_mask = useless_discard_mask * avail_mask
        has_useless_discard = useless_discard_mask.sum() > 0

        # 3. TellAnyoneAboutUsefulCard
        hint_playable_mask = self._find_playable_hint_mask(
            partner_hand, next_playable_rank, avail_mask
        )
        can_hint_playable = (hint_playable_mask.sum() > 0) & (info_tokens > 0)

        # 4. TellAnyoneAboutUselessCard (dispensable)
        hint_dispensable_mask = self._find_dispensable_hint_mask(
            partner_hand, next_playable_rank, avail_mask
        )
        can_hint_dispensable = (hint_dispensable_mask.sum() > 0) & (info_tokens > 0)

        # 5. TellMostInformation
        most_info_mask = self._find_most_informative_hint_mask(
            partner_hand, partner_knowledge_2d, avail_mask
        )
        # Pick the single highest-score hint (argmax with mask)
        most_info_best = (most_info_mask == jnp.max(most_info_mask)).astype(jnp.float32) * (most_info_mask > 0)
        can_tell_most_info = (most_info_mask.sum() > 0) & (info_tokens > 0)

        # 6. DiscardProbablyUseless(0.0) for probabilistic fallback
        prob_useless = self._probability_useless(my_knowledge_2d, next_playable_rank)
        prob_discard_weights = jnp.zeros(self.num_actions)
        prob_discard_weights = prob_discard_weights.at[self.discard_start:self.discard_end].set(
            prob_useless
        )
        prob_discard_weights = prob_discard_weights * avail_mask
        prob_discard_logits = jnp.where(
            prob_discard_weights > 0, jnp.log(prob_discard_weights + 1e-10), -1e9
        )
        has_prob_discard = (prob_discard_weights.sum() > 0)

        # Priority cascade matching canonical order
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
                        can_hint_dispensable,
                        jnp.where(hint_dispensable_mask > 0, 0.0, -1e9),
                        jnp.where(
                            can_tell_most_info,
                            jnp.where(most_info_best > 0, 0.0, -1e9),
                            # DiscardProbablyUseless(0.0) for probabilistic discard
                            # is already the canonical terminal rule for VDB.
                            jnp.where(
                                has_prob_discard,
                                prob_discard_logits,
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
        )

        action = jax.random.categorical(rng, action_logits)
        return action, agent_state
