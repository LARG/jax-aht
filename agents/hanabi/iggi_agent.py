
import jax
import jax.numpy as jnp
from agents.hanabi.base_agent import BaseAgent, AgentState
from typing import Tuple


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
        self.discard_start = 0
        self.discard_end = hand_size
        self.play_start = hand_size
        self.play_end = 2 * hand_size
        self.hint_color_start = 2 * hand_size
        self.hint_color_end = 2 * hand_size + num_colors
        self.hint_rank_start = 2 * hand_size + num_colors
        self.hint_rank_end = 2 * hand_size + num_colors + num_ranks

    def _is_certainly_playable(self, my_knowledge_2d, next_playable_rank):
        rank_idx = jnp.arange(self.num_ranks)
        playable_matrix = (rank_idx[None, :] == next_playable_rank[:, None]).astype(jnp.float32)

        non_playable_possibilities = my_knowledge_2d * (1.0 - playable_matrix[None, :, :])
        has_any_possibility = my_knowledge_2d.sum(axis=(1, 2)) > 0
        is_playable = (non_playable_possibilities.sum(axis=(1, 2)) == 0) & has_any_possibility
        return is_playable

    def _is_certainly_useless(self, my_knowledge_2d, next_playable_rank, env_state=None):
        rank_idx = jnp.arange(self.num_ranks)
        completed_matrix = (rank_idx[None, :] < next_playable_rank[:, None]).astype(jnp.float32)

        if env_state is not None:
            card_counts = jnp.array([3, 2, 2, 2, 1])[:self.num_ranks]

            max_discard = env_state.discard_pile.shape[0]
            valid_mask = (jnp.arange(max_discard) < env_state.num_cards_discarded)[:, None, None]
            discarded_count = (env_state.discard_pile * valid_mask).sum(axis=0)

            played_count = env_state.fireworks

            total_gone = discarded_count + played_count
            total_copies = jnp.broadcast_to(card_counts[None, :], (self.num_colors, self.num_ranks))
            all_copies_gone = (total_gone >= total_copies).astype(jnp.float32)

            useless_matrix = jnp.maximum(completed_matrix, all_copies_gone)
        else:
            useless_matrix = completed_matrix

        useful_possibilities = my_knowledge_2d * (1.0 - useless_matrix[None, :, :])
        has_any_possibility = my_knowledge_2d.sum(axis=(1, 2)) > 0
        is_useless = (useful_possibilities.sum(axis=(1, 2)) == 0) & has_any_possibility
        return is_useless

    @staticmethod
    def _mask_to_logits(mask):
        return jnp.where(mask > 0, 0.0, -1e9)

    def _select_priority_action(self, rules, avail_mask, rng):
        fallback = jnp.zeros(self.num_actions)
        fallback = fallback.at[self.discard_start:self.discard_end].set(1.0)
        fallback = fallback * avail_mask
        action_logits = jnp.where(fallback > 0, 0.0, -1e9)
        for condition, rule_logits in reversed(rules):
            action_logits = jnp.where(condition, rule_logits, action_logits)
        return jax.random.categorical(rng, action_logits)

    def _find_playable_hint_mask(self, partner_hand, next_playable_rank, avail_mask):
        partner_colors = jnp.argmax(partner_hand.sum(axis=2), axis=1)
        partner_ranks = jnp.argmax(partner_hand.sum(axis=1), axis=1)

        partner_playable = (
            partner_ranks == next_playable_rank[partner_colors]
        ).astype(jnp.float32)

        playable_cards = partner_hand * partner_playable[:, None, None]
        colors_with_playable = (playable_cards.sum(axis=(0, 2)) > 0).astype(jnp.float32)
        ranks_with_playable = (playable_cards.sum(axis=(0, 1)) > 0).astype(jnp.float32)
        hint_mask = jnp.zeros(self.num_actions)
        hint_mask = hint_mask.at[self.hint_color_start:self.hint_color_end].set(colors_with_playable)
        hint_mask = hint_mask.at[self.hint_rank_start:self.hint_rank_end].set(ranks_with_playable)
        return hint_mask * avail_mask

    def _random_discard_mask(self, avail_mask):
        mask = jnp.zeros(self.num_actions)
        mask = mask.at[self.discard_start:self.discard_end].set(1.0)
        return mask * avail_mask

    def _oldest_unhinted_discard_mask(self, env_state, my_id, avail_mask):
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
        return oldest_discard_mask * avail_mask

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

        oldest_discard_mask = self._oldest_unhinted_discard_mask(env_state, my_id, avail_mask)
        has_oldest_discard = oldest_discard_mask.sum() > 0

        random_hint_mask = jnp.zeros(self.num_actions)
        random_hint_mask = random_hint_mask.at[self.hint_color_start:self.hint_rank_end].set(1.0)
        random_hint_mask = random_hint_mask * avail_mask
        has_random_hint = (random_hint_mask.sum() > 0) & (info_tokens > 0)

        m2l = self._mask_to_logits
        rules = [
            (has_safe_play,        m2l(safe_play_mask)),
            (can_hint_playable,    m2l(hint_playable_mask)),
            (has_useless_discard,  m2l(useless_discard_mask)),
            (has_oldest_discard,   m2l(oldest_discard_mask)),
            (has_random_hint,      m2l(random_hint_mask)),
        ]
        action = self._select_priority_action(rules, avail_mask, rng)
        return action, agent_state
