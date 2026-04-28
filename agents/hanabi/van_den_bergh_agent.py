import jax
import jax.numpy as jnp
from agents.hanabi.iggi_agent import IGGIAgent
from agents.hanabi.base_agent import AgentState
from typing import Tuple


class VanDenBerghAgent(IGGIAgent):
    def __init__(self, play_threshold: float = 0.6, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.play_threshold = play_threshold

    def _probability_playable(self, my_knowledge_2d, next_playable_rank):
        rank_idx = jnp.arange(self.num_ranks)
        playable_matrix = (rank_idx[None, :] == next_playable_rank[:, None]).astype(jnp.float32)
        playable_possibilities = (my_knowledge_2d * playable_matrix[None, :, :]).sum(axis=(1, 2))
        total_possibilities = my_knowledge_2d.sum(axis=(1, 2))
        prob = playable_possibilities / jnp.maximum(total_possibilities, 1e-8)
        return prob

    def _probability_useless(self, my_knowledge_2d, next_playable_rank):
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
        partner_colors = jnp.argmax(partner_hand.sum(axis=2), axis=1)
        partner_ranks = jnp.argmax(partner_hand.sum(axis=1), axis=1)

        color_poss = partner_knowledge_2d.sum(axis=2)
        n_colors_possible = (color_poss > 0).sum(axis=1)
        color_unknown = n_colors_possible > 1

        rank_poss = partner_knowledge_2d.sum(axis=1)
        n_ranks_possible = (rank_poss > 0).sum(axis=1)
        rank_unknown = n_ranks_possible > 1

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

        prob_playable = self._probability_playable(my_knowledge_2d, next_playable_rank)
        is_certainly_playable = self._is_certainly_playable(my_knowledge_2d, next_playable_rank)

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

        is_useless = self._is_certainly_useless(my_knowledge_2d, next_playable_rank, env_state)
        useless_discard_mask = jnp.zeros(self.num_actions)
        useless_discard_mask = useless_discard_mask.at[self.discard_start:self.discard_end].set(
            is_useless.astype(jnp.float32)
        )
        useless_discard_mask = useless_discard_mask * avail_mask
        has_useless_discard = useless_discard_mask.sum() > 0

        hint_playable_mask = self._find_playable_hint_mask(
            partner_hand, next_playable_rank, avail_mask
        )
        can_hint_playable = (hint_playable_mask.sum() > 0) & (info_tokens > 0)

        hint_dispensable_mask = self._find_dispensable_hint_mask(
            partner_hand, next_playable_rank, avail_mask
        )
        can_hint_dispensable = (hint_dispensable_mask.sum() > 0) & (info_tokens > 0)

        most_info_mask = self._find_most_informative_hint_mask(
            partner_hand, partner_knowledge_2d, avail_mask
        )
        most_info_best = (most_info_mask == jnp.max(most_info_mask)).astype(jnp.float32) * (most_info_mask > 0)
        can_tell_most_info = (most_info_mask.sum() > 0) & (info_tokens > 0)

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

        m2l = self._mask_to_logits
        rules = [
            (has_safe_play,        m2l(safe_play_mask)),
            (has_useless_discard,  m2l(useless_discard_mask)),
            (can_hint_playable,    m2l(hint_playable_mask)),
            (can_hint_dispensable, m2l(hint_dispensable_mask)),
            (can_tell_most_info,   m2l(most_info_best)),
            (has_prob_discard,     prob_discard_logits),
        ]
        action = self._select_priority_action(rules, avail_mask, rng)
        return action, agent_state
