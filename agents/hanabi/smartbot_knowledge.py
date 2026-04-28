"""SmartBot belief system. locatedCount fixpoint, eyesight, card queries.

Extends JaxMARL's card_knowledge with cross-hand elimination and
private eyesight. All JIT-compatible.
"""
import jax
import jax.numpy as jnp
from functools import partial

# copies per rank: [3, 2, 2, 2, 1] for standard Hanabi
STANDARD_CARD_COUNTS = jnp.array([3, 2, 2, 2, 1])


def init_beliefs(card_knowledge, num_players, hand_size, num_colors, num_ranks):
    """Reshape JaxMARL's flat card_knowledge into (np, hs, nc, nr).

    Note: this rebuilds beliefs from the current turn's card_knowledge
    only. C++ SmartBot persists a belief object across turns so earlier
    narrowing (from hints or from seeing specific cards get discarded)
    carries forward. We don't, and this is one of the two attributed
    causes of the ~5pt gap to C++. Fix would require threading a
    per-slot belief array through hstate.
    """
    return card_knowledge.reshape(num_players, hand_size, num_colors, num_ranks)


def compute_played_count(fireworks, discard_pile, num_discarded):
    """Count publicly visible cards (fireworks + discard pile)."""
    max_discard = discard_pile.shape[0]
    valid_mask = (jnp.arange(max_discard) < num_discarded)[:, None, None]
    discarded = (discard_pile * valid_mask).sum(axis=0)
    return fireworks + discarded


def compute_located_count(beliefs):
    """Count uniquely-identified cards across all hands."""
    possibilities_per_card = beliefs.sum(axis=(-2, -1))  # (np, hs)
    is_identified = (possibilities_per_card == 1).astype(jnp.float32)
    identified_cards = beliefs * is_identified[:, :, None, None]
    return identified_cards.sum(axis=(0, 1))


def eliminate_fully_accounted(beliefs, played_count, located_count, card_counts):
    """Zero out possibilities for (c,r) where all copies are accounted for."""
    total_copies = jnp.broadcast_to(card_counts[None, :], played_count.shape)
    fully_accounted = (played_count + located_count >= total_copies).astype(jnp.float32)

    # only eliminate from ambiguous cards, don't touch already-identified ones
    poss_per_card = beliefs.sum(axis=(-2, -1))
    is_ambiguous = (poss_per_card > 1).astype(jnp.float32)

    eliminate_mask = fully_accounted[None, None, :, :] * is_ambiguous[:, :, None, None]
    return beliefs * (1.0 - eliminate_mask)


def run_located_fixpoint(beliefs, played_count, card_counts, max_iters=10):
    """Iterate locate -> eliminate until convergence. Typically 2-3 iters."""
    def body_fn(carry):
        b, prev_loc = carry
        loc = compute_located_count(b)
        b = eliminate_fully_accounted(b, played_count, loc, card_counts)
        return (b, loc)

    def cond_fn(carry):
        b, prev_loc = carry
        loc = compute_located_count(b)
        return jnp.any(loc != prev_loc)

    init_loc = compute_located_count(beliefs)
    beliefs, init_loc = body_fn((beliefs, init_loc))
    beliefs, final_loc = jax.lax.while_loop(cond_fn, body_fn, (beliefs, init_loc))
    return beliefs, final_loc


def apply_eyesight(my_beliefs, partner_hand, played_count, card_counts):
    """Eliminate possibilities using what I can see (partner hand + my identified cards)."""
    partner_count = partner_hand.sum(axis=0)
    my_poss = my_beliefs.sum(axis=(-2, -1))
    my_identified = (my_poss == 1).astype(jnp.float32)
    my_identified_count = (my_beliefs * my_identified[:, None, None]).sum(axis=0)

    eyesight_count = played_count + partner_count + my_identified_count

    total_copies = jnp.broadcast_to(card_counts[None, :], played_count.shape)
    fully_seen = (eyesight_count >= total_copies).astype(jnp.float32)

    is_ambiguous = (my_poss > 1).astype(jnp.float32)
    eliminate = fully_seen[None, :, :] * is_ambiguous[:, None, None]

    return my_beliefs * (1.0 - eliminate)


def get_next_playable_ranks(fireworks):
    return jnp.sum(fireworks, axis=1).astype(jnp.int32)


def compute_playable_matrix(next_playable_rank, num_colors, num_ranks):
    """Matrix where playable_matrix[c, r] = 1 iff r == next_playable_rank[c]."""
    rank_idx = jnp.arange(num_ranks)
    return (rank_idx[None, :] == next_playable_rank[:, None]).astype(jnp.float32)


def compute_completed_matrix(next_playable_rank, num_ranks):
    """Matrix where completed[c, r] = 1 iff r < next_playable_rank[c] (already played)."""
    rank_idx = jnp.arange(num_ranks)
    return (rank_idx[None, :] < next_playable_rank[:, None]).astype(jnp.float32)


def compute_unreachable_matrix(played_count, card_counts, next_playable_rank, num_colors, num_ranks):
    """unreachable[c,r] = 1 if completed or a prerequisite rank is fully gone."""
    completed = compute_completed_matrix(next_playable_rank, num_ranks)
    total_copies = jnp.broadcast_to(card_counts[None, :], (num_colors, num_ranks))

    blocked = jnp.zeros((num_colors, num_ranks))
    for c in range(num_colors):
        # Cumulative "all copies gone" from the next playable rank upward
        all_gone = (played_count[c, :] >= total_copies[c, :]).astype(jnp.float32)
        # A rank r is blocked if any rank below it (but >= next_playable) is all-gone
        # Use cumulative max: if any prerequisite is gone, everything above is blocked
        cumulative_blocked = jnp.zeros(num_ranks)
        for r in range(num_ranks):
            # r is blocked if: (a) it's completed, OR (b) some prerequisite r' < r
            # has all copies gone AND r' >= next_playable_rank[c]
            is_above_playable = (r >= next_playable_rank[c]).astype(jnp.float32)
            prereq_gone = jnp.zeros(())
            for rr in range(num_ranks):
                is_prereq = (rr < r) & (rr >= next_playable_rank[c])
                prereq_gone = prereq_gone + jnp.where(
                    is_prereq, all_gone[rr], 0.0
                )
            cumulative_blocked = cumulative_blocked.at[r].set(
                jnp.where(prereq_gone > 0, 1.0, 0.0) * is_above_playable
            )
        blocked = blocked.at[c, :].set(cumulative_blocked)

    return jnp.maximum(completed, blocked)


def is_certainly_playable(card_beliefs, playable_matrix):
    non_playable = card_beliefs * (1.0 - playable_matrix)
    has_any = card_beliefs.sum() > 0
    return (non_playable.sum() == 0) & has_any


def is_certainly_worthless(card_beliefs, unreachable_matrix):
    useful = card_beliefs * (1.0 - unreachable_matrix)
    has_any = card_beliefs.sum() > 0
    return (useful.sum() == 0) & has_any


def is_card_valuable(color, rank, played_count, card_counts, unreachable_matrix):
    """True if this is the last needed copy of (color, rank)."""
    total = card_counts[rank]
    in_play = played_count[color, rank]
    is_last_copy = (in_play >= total - 1)
    not_unreachable = unreachable_matrix[color, rank] < 0.5
    return is_last_copy & not_unreachable


def probability_playable(card_beliefs, playable_matrix):
    playable_poss = (card_beliefs * playable_matrix).sum()
    total_poss = card_beliefs.sum()
    return playable_poss / jnp.maximum(total_poss, 1e-8)


def probability_worthless(card_beliefs, unreachable_matrix):
    worthless_poss = (card_beliefs * unreachable_matrix).sum()
    total_poss = card_beliefs.sum()
    return worthless_poss / jnp.maximum(total_poss, 1e-8)


def batch_is_certainly_playable(beliefs_2d, playable_matrix):
    non_playable = beliefs_2d * (1.0 - playable_matrix[None, :, :])
    has_any = beliefs_2d.sum(axis=(-2, -1)) > 0
    return (non_playable.sum(axis=(-2, -1)) == 0) & has_any


def batch_is_certainly_worthless(beliefs_2d, unreachable_matrix):
    useful = beliefs_2d * (1.0 - unreachable_matrix[None, :, :])
    has_any = beliefs_2d.sum(axis=(-2, -1)) > 0
    return (useful.sum(axis=(-2, -1)) == 0) & has_any


def batch_probability_playable(beliefs_2d, playable_matrix):
    playable_poss = (beliefs_2d * playable_matrix[None, :, :]).sum(axis=(-2, -1))
    total_poss = beliefs_2d.sum(axis=(-2, -1))
    return playable_poss / jnp.maximum(total_poss, 1e-8)


def batch_probability_worthless(beliefs_2d, unreachable_matrix):
    worthless_poss = (beliefs_2d * unreachable_matrix[None, :, :]).sum(axis=(-2, -1))
    total_poss = beliefs_2d.sum(axis=(-2, -1))
    return worthless_poss / jnp.maximum(total_poss, 1e-8)
