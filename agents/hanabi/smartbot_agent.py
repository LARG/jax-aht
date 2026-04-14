"""SmartBot: JAX port of Quuxplusone/Hanabi SmartBot.cc.

Strongest rule-based agent. Belief fixpoint, bilateral conventions,
info-theoretic hints. C++ self-play reference: 23.09/25. This port
scores ~18/25, and the ~5-point gap has two known causes:

  1. Cross-turn belief persistence. C++ SmartBot keeps a persistent
     belief object across turns: once a card's possibilities get
     narrowed by a hint or by seeing another card discarded, that
     narrowing is preserved even after intervening turns. This port
     rebuilds beliefs from JaxMARL's card_knowledge each turn (see
     init_beliefs() at the top of _get_action), so cumulative
     narrowing from earlier turns is lost. Fixing this would require
     threading a per-slot belief array through hstate and shifting it
     correctly on play/discard.

  2. Discard finesse convention. In C++ SmartBot, when a partner
     discards a card from a known slot, the receiving agent uses the
     position shift of the remaining cards to infer what was at that
     slot. See smartbot_conventions.py for the positional-shift
     helpers; the inference step itself is not implemented because
     JaxMARL's slot-indexed state makes card identity tracking across
     shifts awkward.

Both are documented as future work in the writeup.
"""
from typing import Tuple

import jax
import jax.numpy as jnp

from agents.hanabi.base_agent import BaseAgent
from agents.hanabi.smartbot_conventions import (
    SmartBotState, init_smartbot_state,
    apply_all_conventions, update_smartbot_state,
    get_removed_slot, shift_conv_mask, shift_bool_array,
    apply_conv_mask_to_beliefs,
)
from agents.hanabi.smartbot_knowledge import (
    STANDARD_CARD_COUNTS,
    init_beliefs,
    compute_played_count,
    run_located_fixpoint,
    apply_eyesight,
    get_next_playable_ranks,
    compute_playable_matrix,
    compute_unreachable_matrix,
    batch_is_certainly_playable,
    batch_is_certainly_worthless,
    batch_probability_playable,
    batch_probability_worthless,
    is_card_valuable,
)
from agents.hanabi.smartbot_hints import find_best_hint

# mystery-card gamble threshold indexed by lives remaining
MYSTERY_THRESHOLD = jnp.array([-99, 1, 1, 3])


class SmartBotAgent(BaseAgent):

    def __init__(self, hand_size=5, num_colors=5, num_ranks=5,
                 num_actions=21, card_counts=None, **kwargs):
        super().__init__(num_actions=num_actions, **kwargs)
        self.hand_size = hand_size
        self.num_colors = num_colors
        self.num_ranks = num_ranks
        # card_counts[r] = number of copies of rank r+1 in the deck (per color).
        # Defaults to the standard Hanabi [3,2,2,2,1][:num_ranks], but mini-Hanabi
        # uses [2,2,1] which is not just a slice. Passing env_kwargs.num_cards_of_rank
        # keeps belief tracking and endgame deck count correct on custom decks.
        if card_counts is None:
            card_counts = STANDARD_CARD_COUNTS[:num_ranks]
        card_counts = jnp.asarray(card_counts)
        assert card_counts.shape == (num_ranks,), (
            f"card_counts has shape {card_counts.shape}, expected ({num_ranks},)"
        )
        self.card_counts = card_counts

        self.discard_start = 0
        self.discard_end = hand_size
        self.play_start = hand_size
        self.play_end = 2 * hand_size
        self.hint_color_start = 2 * hand_size
        self.hint_color_end = 2 * hand_size + num_colors
        self.hint_rank_start = 2 * hand_size + num_colors
        self.hint_rank_end = 2 * hand_size + num_colors + num_ranks

    def init_agent_state(self, agent_id):
        return init_smartbot_state(
            agent_id, hand_size=self.hand_size,
            num_colors=self.num_colors, num_ranks=self.num_ranks
        )

    def _next_discard_index(self, beliefs, playable_matrix, unreachable,
                            played_count):
        """C++ nextDiscardIndex. Returns -1 if should play/discard worthless instead."""
        is_play = batch_is_certainly_playable(beliefs, playable_matrix)
        is_worth = batch_is_certainly_worthless(beliefs, unreachable)
        should_not_discard = is_play.any() | is_worth.any()

        prob_w = batch_probability_worthless(beliefs, unreachable)

        possibly_valuable = jnp.zeros(self.hand_size)
        for c in range(self.num_colors):
            for r in range(self.num_ranks):
                is_val = is_card_valuable(c, r, played_count, self.card_counts, unreachable)
                possibly_valuable = possibly_valuable + (
                    beliefs[:, c, r] * is_val.astype(jnp.float32)
                )

        fitness = jnp.where(
            is_play | is_worth, -200.0,
            jnp.where(possibly_valuable > 0, -100.0, 100.0 + prob_w)
        )
        best_idx = jnp.argmax(fitness)
        return jnp.where(should_not_discard, jnp.int32(-1), best_idx)

    def _play_lowest_playable(self, my_beliefs, playable_matrix, public_beliefs):
        """C++ maybePlayLowestPlayableCard. Prefers low-value cards, +100 if not publicly known."""
        is_play = batch_is_certainly_playable(my_beliefs, playable_matrix)
        publicly_play = batch_is_certainly_playable(public_beliefs, playable_matrix)

        rank_vals = jnp.arange(self.num_ranks).astype(jnp.float32) + 1.0
        ev = (my_beliefs * rank_vals[None, None, :]).sum(axis=(-2, -1))
        ev = ev / jnp.maximum(my_beliefs.sum(axis=(-2, -1)), 1e-8)

        fitness = jnp.where(is_play, 6.0 - ev, -1e9)
        fitness = fitness + jnp.where(is_play & ~publicly_play, 100.0, 0.0)

        best_slot = jnp.argmax(fitness)
        has_play = is_play.any()
        return best_slot, has_play

    def _maybe_play_mystery(self, my_beliefs, playable_matrix, life_tokens,
                            cards_in_deck):
        """C++ maybePlayMysteryCard. Gamble when deck is nearly empty."""
        prob_play = batch_probability_playable(my_beliefs, playable_matrix)
        best_slot = jnp.argmax(prob_play)
        best_prob = prob_play[best_slot]

        lives_clamped = jnp.clip(life_tokens, 0, 3).astype(jnp.int32)
        threshold = MYSTERY_THRESHOLD[lives_clamped]
        can_gamble = (cards_in_deck <= threshold) & (best_prob > 0.0)

        return best_slot, can_gamble

    def _maybe_give_warning(self, partner_beliefs, partner_hand,
                            playable_matrix, unreachable, played_count,
                            info_tokens, avail_mask, hint_fitness):
        """C++ maybeGiveValuableWarning. Warn if partner's discard candidate is valuable."""
        p_discard_idx = self._next_discard_index(
            partner_beliefs, playable_matrix, unreachable, played_count
        )
        p_card_color = jnp.argmax(partner_hand[jnp.maximum(p_discard_idx, 0)].sum(axis=1))
        p_card_rank = jnp.argmax(partner_hand[jnp.maximum(p_discard_idx, 0)].sum(axis=0))

        card_is_valuable = is_card_valuable(
            p_card_color, p_card_rank, played_count, self.card_counts, unreachable
        )

        warning_action = self.hint_rank_start + p_card_rank
        should_warn = (
            (p_discard_idx >= 0) &
            card_is_valuable &
            (info_tokens > 0) &
            (hint_fitness <= 0) &  # [#22] prefer helpful hint over warning
            (avail_mask[warning_action] > 0)
        )
        return warning_action, should_warn, p_card_rank

    def _get_action(
        self,
        obs: jnp.ndarray,
        env_state,
        avail_mask: jnp.ndarray,
        agent_state: SmartBotState,
        rng: jax.random.PRNGKey,
    ) -> Tuple[int, SmartBotState]:

        my_id = agent_state.agent_id
        partner_id = 1 - my_id

        # --- decode ---
        fireworks = env_state.fireworks
        next_playable = get_next_playable_ranks(fireworks)
        playable_matrix = compute_playable_matrix(next_playable, self.num_colors, self.num_ranks)

        partner_hand = env_state.player_hands[partner_id]
        info_tokens = jnp.sum(env_state.info_tokens)
        life_tokens = jnp.sum(env_state.life_tokens)

        discard_pile = env_state.discard_pile
        num_discarded = env_state.num_cards_discarded

        played_count = compute_played_count(fireworks, discard_pile, num_discarded)
        unreachable = compute_unreachable_matrix(
            played_count, self.card_counts, next_playable,
            self.num_colors, self.num_ranks
        )

        # deck count for endgame
        total_cards = jnp.int32(self.card_counts.sum() * self.num_colors)
        cards_in_play = fireworks.sum().astype(jnp.int32) + num_discarded + 2 * self.hand_size
        cards_in_deck = jnp.maximum(total_cards - cards_in_play, 0)

        # --- beliefs ---
        all_beliefs = init_beliefs(
            env_state.card_knowledge, 2, self.hand_size,
            self.num_colors, self.num_ranks
        )
        all_beliefs, _ = run_located_fixpoint(
            all_beliefs, played_count, self.card_counts
        )
        public_beliefs = all_beliefs[my_id]  # pre-eyesight = public knowledge
        partner_beliefs = all_beliefs[partner_id]

        # shift convention state to match JaxMARL's card-shift after play/discard
        removed = get_removed_slot(agent_state.prev_my_action, self.hand_size)
        conv_mask = shift_conv_mask(
            agent_state.conv_mask, removed, self.hand_size
        )
        shifted_discard_safe = shift_bool_array(
            agent_state.discard_safe, removed, self.hand_size
        )

        # apply persisted convention mask before eyesight (avoids stale eyesight contamination)
        narrowed_public = apply_conv_mask_to_beliefs(public_beliefs, conv_mask)

        my_beliefs = apply_eyesight(
            narrowed_public, partner_hand, played_count, self.card_counts
        )

        # --- conventions ---
        my_beliefs, new_discard_safe, conv_mask = apply_all_conventions(
            my_beliefs, env_state, agent_state,
            playable_matrix, next_playable, played_count,
            self.card_counts, unreachable,
            self.num_colors, self.num_ranks, self.hand_size,
            conv_mask, shifted_discard_safe, public_beliefs
        )

        # --- candidates ---
        play_slot, has_safe_play = self._play_lowest_playable(
            my_beliefs, playable_matrix, public_beliefs
        )
        play_action = self.play_start + play_slot

        # mystery
        mystery_slot, can_mystery = self._maybe_play_mystery(
            my_beliefs, playable_matrix, life_tokens, cards_in_deck
        )
        mystery_action = self.play_start + mystery_slot

        # hint (with valueToAvoid to not confuse partner's discard candidate)
        p_discard_idx = self._next_discard_index(
            partner_beliefs, playable_matrix, unreachable, played_count
        )
        p_card_rank = jnp.argmax(
            partner_hand[jnp.maximum(p_discard_idx, 0)].sum(axis=0)
        )
        p_card_is_valuable = is_card_valuable(
            jnp.argmax(partner_hand[jnp.maximum(p_discard_idx, 0)].sum(axis=1)),
            p_card_rank, played_count, self.card_counts, unreachable
        )
        value_to_avoid = jnp.where(
            p_card_is_valuable & (p_discard_idx >= 0),
            p_card_rank, jnp.int32(-1)
        )

        best_hint_action, hint_fitness = find_best_hint(
            partner_beliefs, partner_hand, playable_matrix, next_playable,
            avail_mask, self.num_colors, self.num_ranks, self.hand_size,
            self.hint_color_start, self.hint_rank_start,
            value_to_avoid=value_to_avoid,  # [#20] passed INTO the search
        )
        can_give_hint = (hint_fitness > 0) & (info_tokens > 0)

        # warning
        warning_action, can_warn, _ = self._maybe_give_warning(
            partner_beliefs, partner_hand, playable_matrix, unreachable,
            played_count, info_tokens, avail_mask, hint_fitness
        )

        # discard
        is_worthless = batch_is_certainly_worthless(my_beliefs, unreachable)
        has_worthless = is_worthless.any()
        worthless_slot = jnp.argmax(is_worthless.astype(jnp.float32))
        worthless_action = self.discard_start + worthless_slot

        my_discard_idx = self._next_discard_index(
            my_beliefs, playable_matrix, unreachable, played_count
        )
        my_prob_w = batch_probability_worthless(my_beliefs, unreachable)
        is_play = batch_is_certainly_playable(my_beliefs, playable_matrix)
        discard_fitness = jnp.where(
            is_play | is_worthless, -200.0, my_prob_w
        )
        discard_fitness = discard_fitness + new_discard_safe.astype(jnp.float32) * 50.0
        best_discard_slot = jnp.argmax(discard_fitness)
        best_discard_action = self.discard_start + best_discard_slot
        can_discard = avail_mask[best_discard_action] > 0
        any_discard_legal = avail_mask[self.discard_start:self.discard_end].sum() > 0

        # throwaway hint when can't discard (hint-stone reclaim)
        oldest_slot = self.hand_size - 1
        partner_oldest_rank = jnp.argmax(partner_hand[oldest_slot].sum(axis=0))
        throwaway_action = self.hint_rank_start + partner_oldest_rank
        throwaway_legal = avail_mask[throwaway_action] > 0
        can_throwaway = ~any_discard_legal & throwaway_legal

        # --- cascade (C++ pleaseMakeMove) ---
        # endgame: deck empty -> play safe or gamble
        endgame = cards_in_deck <= 0
        endgame_action = jnp.where(has_safe_play, play_action, mystery_action)
        endgame_can = endgame & (has_safe_play | can_mystery)

        action = jnp.where(
            endgame_can, endgame_action,
            jnp.where(
                can_warn, warning_action,
                jnp.where(
                    has_safe_play, play_action,
                    jnp.where(
                        can_give_hint, best_hint_action,
                        jnp.where(
                            can_mystery, mystery_action,
                            jnp.where(
                                can_throwaway, throwaway_action,
                                jnp.where(
                                    has_worthless, worthless_action,
                                    jnp.where(
                                        can_discard, best_discard_action,
                                        jnp.where(
                                            any_discard_legal,
                                            self.discard_start,
                                            jnp.int32(0)
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )

        # safety net: fall back to random legal action if cascade pick is illegal
        action = jnp.where(
            avail_mask[action] > 0, action,
            jax.random.categorical(rng, jnp.where(avail_mask > 0, 0.0, -1e9))
        )

        new_state = update_smartbot_state(
            env_state, agent_state, new_discard_safe, action, conv_mask
        )
        return action, new_state
