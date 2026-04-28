import jax
import jax.numpy as jnp
from agents.hanabi.iggi_agent import IGGIAgent
from agents.hanabi.base_agent import AgentState
from typing import Tuple


class PiersAgent(IGGIAgent):
    def __init__(
        self,
        play_threshold: float = 0.6,
        hint_threshold: int = 4,
        hand_size: int = 5,
        num_colors: int = 5,
        num_ranks: int = 5,
        num_actions: int = 21,
        **kwargs,
    ):
        super().__init__(
            hand_size=hand_size, num_colors=num_colors,
            num_ranks=num_ranks, num_actions=num_actions, **kwargs,
        )
        self.play_threshold = play_threshold
        self.hint_threshold = hint_threshold

    def _probability_playable(self, my_knowledge_2d, next_playable_rank):
        rank_idx = jnp.arange(self.num_ranks)
        playable_matrix = (rank_idx[None, :] == next_playable_rank[:, None]).astype(jnp.float32)

        playable_possibilities = (my_knowledge_2d * playable_matrix[None, :, :]).sum(axis=(1, 2))
        total_possibilities = my_knowledge_2d.sum(axis=(1, 2))
        prob = playable_possibilities / jnp.maximum(total_possibilities, 1e-8)
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
        life_tokens = jnp.sum(env_state.life_tokens)
        deck_empty = env_state.remaining_deck_size[0] == 0

        prob_playable = self._probability_playable(my_knowledge_2d, next_playable_rank)

        is_playable = self._is_certainly_playable(my_knowledge_2d, next_playable_rank)
        safe_play_mask = jnp.zeros(self.num_actions)
        safe_play_mask = safe_play_mask.at[self.play_start:self.play_end].set(
            is_playable.astype(jnp.float32)
        )
        safe_play_mask = safe_play_mask * avail_mask
        has_safe_play = safe_play_mask.sum() > 0

        is_probably_playable = prob_playable >= self.play_threshold
        prob_play_mask = jnp.zeros(self.num_actions)
        prob_play_mask = prob_play_mask.at[self.play_start:self.play_end].set(
            is_probably_playable.astype(jnp.float32)
        )
        prob_play_mask = prob_play_mask * avail_mask
        can_prob_play = (prob_play_mask.sum() > 0) & (life_tokens > 1)

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

        hint_dispensable_mask = self._find_dispensable_hint_mask(
            partner_hand, next_playable_rank, avail_mask
        )
        can_hint_dispensable = (
            (hint_dispensable_mask.sum() > 0) &
            (info_tokens > 0)
        )

        colors_rev = env_state.colors_revealed[my_id]
        ranks_rev = env_state.ranks_revealed[my_id]
        has_hint = (colors_rev.sum(axis=1) > 0) | (ranks_rev.sum(axis=1) > 0)
        unhinted = (~has_hint).astype(jnp.float32)

        unhinted_avail = unhinted * avail_mask[self.discard_start:self.discard_end]
        has_unhinted = unhinted_avail.sum() > 0
        slot_indices = jnp.arange(self.hand_size).astype(jnp.float32)
        unhinted_priority = jnp.where(unhinted_avail > 0, slot_indices, 1e9)
        oldest_unhinted_slot = jnp.argmin(unhinted_priority)

        oldest_discard_mask = jnp.zeros(self.num_actions)
        chosen_slot = jnp.where(has_unhinted, oldest_unhinted_slot, 0)
        oldest_discard_mask = oldest_discard_mask.at[self.discard_start + chosen_slot].set(1.0)
        oldest_discard_mask = oldest_discard_mask * avail_mask
        has_oldest_discard = oldest_discard_mask.sum() > 0

        endgame_play_mask = jnp.zeros(self.num_actions)
        endgame_play_mask = endgame_play_mask.at[self.play_start:self.play_end].set(
            (prob_playable > 0.0).astype(jnp.float32)
        )
        endgame_play_mask = endgame_play_mask * avail_mask
        can_endgame_play = (endgame_play_mask.sum() > 0) & deck_empty & (life_tokens > 1)

        random_hint_mask = jnp.zeros(self.num_actions)
        random_hint_mask = random_hint_mask.at[self.hint_color_start:self.hint_rank_end].set(1.0)
        random_hint_mask = random_hint_mask * avail_mask
        has_random_hint = (random_hint_mask.sum() > 0) & (info_tokens > 0)

        m2l = self._mask_to_logits
        rules = [
            (has_safe_play,        m2l(safe_play_mask)),
            (can_prob_play,        m2l(prob_play_mask)),
            (can_hint_playable,    m2l(hint_playable_mask)),
            (has_useless_discard,  m2l(useless_discard_mask)),
            (can_hint_dispensable, m2l(hint_dispensable_mask)),
            (has_oldest_discard,   m2l(oldest_discard_mask)),
            (can_endgame_play,     m2l(endgame_play_mask)),
            (has_random_hint,      m2l(random_hint_mask)),
        ]
        action = self._select_priority_action(rules, avail_mask, rng)
        return action, agent_state
