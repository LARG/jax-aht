"""SmartBot hint evaluation. Simulates all 10 hints, scores by info gain
and convention correctness. Must use same convention target logic as
the receiver (bilateral agreement).
"""
import jax
import jax.numpy as jnp

from agents.hanabi.smartbot_knowledge import batch_is_certainly_playable
from agents.hanabi.smartbot_conventions import find_convention_target


def simulate_color_hint(partner_beliefs, partner_hand, color, num_colors, num_ranks):
    touched = partner_hand[:, color, :].sum(axis=1) > 0  # (hs,)
    new_beliefs = partner_beliefs

    for slot in range(partner_beliefs.shape[0]):
        # Touched: must be this color (zero out all other colors)
        touched_mask = jnp.zeros((num_colors, num_ranks))
        touched_mask = touched_mask.at[color, :].set(1.0)
        # Not touched: cannot be this color (zero out this color)
        not_touched_mask = jnp.ones((num_colors, num_ranks))
        not_touched_mask = not_touched_mask.at[color, :].set(0.0)

        mask = jnp.where(touched[slot], touched_mask, not_touched_mask)
        new_beliefs = new_beliefs.at[slot].set(new_beliefs[slot] * mask)

    return new_beliefs, touched


def simulate_rank_hint(partner_beliefs, partner_hand, rank, num_colors, num_ranks):
    touched = partner_hand[:, :, rank].sum(axis=1) > 0  # (hs,)
    new_beliefs = partner_beliefs

    for slot in range(partner_beliefs.shape[0]):
        touched_mask = jnp.zeros((num_colors, num_ranks))
        touched_mask = touched_mask.at[:, rank].set(1.0)
        not_touched_mask = jnp.ones((num_colors, num_ranks))
        not_touched_mask = not_touched_mask.at[:, rank].set(0.0)

        mask = jnp.where(touched[slot], touched_mask, not_touched_mask)
        new_beliefs = new_beliefs.at[slot].set(new_beliefs[slot] * mask)

    return new_beliefs, touched


def score_hint(old_beliefs, new_beliefs, touched, partner_hand,
               playable_matrix, next_playable_rank, hand_size):
    """Score by info gain + convention correctness. Rejects misleading hints."""
    old_poss = old_beliefs.sum(axis=(-2, -1))
    new_poss = new_beliefs.sum(axis=(-2, -1))
    info_gain = (old_poss - new_poss).sum()

    # did the hint directly reveal a playable card?
    old_certainly_playable = batch_is_certainly_playable(old_beliefs, playable_matrix)
    new_certainly_playable = batch_is_certainly_playable(new_beliefs, playable_matrix)
    newly_revealed = new_certainly_playable & ~old_certainly_playable
    reveals_playable = newly_revealed.any()
    playability_bonus = newly_revealed.sum().astype(jnp.float32) * 100.0

    # convention target check (bilateral: same logic as receiver)
    has_target, target_idx = find_convention_target(
        new_beliefs, touched, playable_matrix, hand_size
    )

    partner_colors = jnp.argmax(partner_hand.sum(axis=2), axis=1)
    partner_ranks = jnp.argmax(partner_hand.sum(axis=1), axis=1)
    actually_playable = partner_ranks == next_playable_rank[partner_colors]

    convention_correct = jnp.where(
        has_target,
        actually_playable[target_idx],
        True  # no target = not misleading
    )

    # reject if convention target exists but isn't actually playable
    is_misleading = has_target & ~convention_correct & ~reveals_playable
    fitness = jnp.where(is_misleading, -1000.0, info_gain + playability_bonus)

    # reject zero-info hints
    is_useless = (info_gain <= 0) & ~reveals_playable & ~(has_target & convention_correct)
    fitness = jnp.where(is_useless, -1.0, fitness)

    return fitness


def find_best_hint(partner_beliefs, partner_hand, playable_matrix,
                   next_playable_rank, avail_mask, num_colors, num_ranks,
                   hand_size, hint_color_start, hint_rank_start,
                   value_to_avoid=None):
    """Best non-misleading hint. value_to_avoid rejects rank hints that
    would be misread as a warning about partner's discard candidate.
    """
    if value_to_avoid is None:
        value_to_avoid = jnp.int32(-1)

    best_action = jnp.int32(-1)
    best_fitness = jnp.float32(-1e9)

    for c in range(num_colors):
        action_idx = hint_color_start + c
        is_legal = avail_mask[action_idx] > 0
        new_beliefs, touched = simulate_color_hint(
            partner_beliefs, partner_hand, c, num_colors, num_ranks
        )
        fitness = score_hint(
            partner_beliefs, new_beliefs, touched, partner_hand,
            playable_matrix, next_playable_rank, hand_size
        )
        fitness = jnp.where(is_legal, fitness, -1e9)
        is_better = fitness > best_fitness
        best_action = jnp.where(is_better, action_idx, best_action)
        best_fitness = jnp.where(is_better, fitness, best_fitness)

    for r in range(num_ranks):
        action_idx = hint_rank_start + r
        is_legal = avail_mask[action_idx] > 0
        new_beliefs, touched = simulate_rank_hint(
            partner_beliefs, partner_hand, r, num_colors, num_ranks
        )
        fitness = score_hint(
            partner_beliefs, new_beliefs, touched, partner_hand,
            playable_matrix, next_playable_rank, hand_size
        )
        # reject rank hints matching valueToAvoid
        is_avoided = (r == value_to_avoid)
        fitness = jnp.where(is_legal & ~is_avoided, fitness, -1e9)
        is_better = fitness > best_fitness
        best_action = jnp.where(is_better, action_idx, best_action)
        best_fitness = jnp.where(is_better, fitness, best_fitness)

    return best_action, best_fitness
