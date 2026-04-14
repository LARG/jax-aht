"""SmartBot convention system. Bilateral: sender and receiver must agree.

Conv 1: newest touched maybe-playable card is inferred playable.
Conv 2: if partner didn't warn, my discard candidate isn't valuable.
Conv 3 (value-warning): rank hint touching discard candidate = it's valuable.

Hint detection uses colors_revealed/ranks_revealed (not card_knowledge diffing).

Not implemented: discard finesse. C++ SmartBot has a convention where
if partner discards from a known slot, the receiving agent uses the
position shift of remaining cards to infer what was at that slot.
shift_conv_mask / shift_bool_array below handle the positional shift
correctly, but the slot-identity-inference step that constitutes the
finesse is missing. This is one of the two attributed causes of the
~5pt gap to C++ SmartBot.

TODO: implement discard finesse. Needs per-slot belief tracking
threaded through hstate.
"""
import jax
import jax.numpy as jnp
from flax import struct

from agents.hanabi.smartbot_knowledge import (
    compute_playable_matrix,
    batch_is_certainly_playable,
    batch_is_certainly_worthless,
    batch_probability_worthless,
    is_card_valuable,
)


@struct.dataclass
class SmartBotState:
    agent_id: int
    prev_info_tokens_sum: jnp.ndarray     # scalar int
    prev_score: jnp.ndarray               # scalar int
    prev_life_tokens_sum: jnp.ndarray     # scalar int
    prev_num_discarded: jnp.ndarray       # scalar int
    prev_turn: jnp.ndarray                # scalar int
    prev_colors_revealed: jnp.ndarray     # (hand_size, num_colors)
    prev_ranks_revealed: jnp.ndarray      # (hand_size, num_ranks)
    discard_safe: jnp.ndarray             # (hand_size,) bool
    # persisted convention constraint mask (all-ones = no constraint)
    conv_mask: jnp.ndarray                # (hand_size, num_colors, num_ranks)
    prev_my_action: jnp.ndarray           # scalar int


def init_smartbot_state(agent_id, hand_size=5, num_colors=5, num_ranks=5):
    return SmartBotState(
        agent_id=agent_id,
        prev_info_tokens_sum=jnp.int32(8),
        prev_score=jnp.int32(0),
        prev_life_tokens_sum=jnp.int32(3),
        prev_num_discarded=jnp.int32(0),
        prev_turn=jnp.int32(0),
        prev_colors_revealed=jnp.zeros((hand_size, num_colors)),
        prev_ranks_revealed=jnp.zeros((hand_size, num_ranks)),
        discard_safe=jnp.zeros(hand_size, dtype=jnp.bool_),
        conv_mask=jnp.ones((hand_size, num_colors, num_ranks)),
        prev_my_action=jnp.int32(-1),
    )



def get_removed_slot(action, hand_size):
    """Which slot was removed by play/discard? -1 if hint/noop."""
    is_discard = (action >= 0) & (action < hand_size)
    is_play = (action >= hand_size) & (action < 2 * hand_size)
    return jnp.where(is_discard, action,
                     jnp.where(is_play, action - hand_size, jnp.int32(-1)))


def shift_bool_array(arr, removed_slot, hand_size):
    """Left-shift after card removal. New card at rightmost slot."""
    needs_shift = removed_slot >= 0
    indices = jnp.arange(hand_size)
    src_indices = jnp.where(indices < removed_slot, indices, indices + 1)
    src_indices = jnp.clip(src_indices, 0, hand_size - 1)
    shifted = arr[src_indices]
    shifted = shifted.at[hand_size - 1].set(False)
    return jnp.where(needs_shift, shifted, arr)


def shift_conv_mask(mask, removed_slot, hand_size):
    """Left-shift 3D convention mask after card removal."""
    needs_shift = removed_slot >= 0
    indices = jnp.arange(hand_size)
    src_indices = jnp.where(indices < removed_slot, indices, indices + 1)
    src_indices = jnp.clip(src_indices, 0, hand_size - 1)
    shifted = mask[src_indices]
    shifted = shifted.at[hand_size - 1].set(jnp.ones_like(mask[0]))
    return jnp.where(needs_shift, shifted, mask)


def apply_conv_mask_to_beliefs(public_beliefs, conv_mask):
    """Intersect beliefs with convention mask. Falls back if intersection is empty."""
    result = public_beliefs * conv_mask
    slot_sums = result.sum(axis=(-2, -1))  # (hand_size,)
    orig_sums = public_beliefs.sum(axis=(-2, -1))
    use_original = (slot_sums < 1e-8) & (orig_sums > 1e-8)
    result = jnp.where(use_original[:, None, None], public_beliefs, result)
    return result


def detect_partner_action(env_state, smartbot_state):
    """Infer partner's action type from state diffs. 0=play_ok, 1=play_fail, 2=discard, 3=hint."""
    cur_info = jnp.sum(env_state.info_tokens)
    cur_score = env_state.score
    cur_lives = jnp.sum(env_state.life_tokens)
    cur_discarded = env_state.num_cards_discarded
    cur_turn = env_state.turn

    prev_info = smartbot_state.prev_info_tokens_sum
    prev_score = smartbot_state.prev_score
    prev_lives = smartbot_state.prev_life_tokens_sum
    prev_discarded = smartbot_state.prev_num_discarded
    prev_turn = smartbot_state.prev_turn

    no_change = (cur_turn <= prev_turn)
    play_success = (cur_score > prev_score) & ~no_change
    play_fail = (cur_lives < prev_lives) & ~no_change & ~play_success
    discarded = (cur_discarded > prev_discarded) & ~play_fail & ~play_success & ~no_change
    hinted = (cur_info < prev_info) & ~no_change

    return jnp.where(no_change, -1,
        jnp.where(play_success, 0,
        jnp.where(play_fail, 1,
        jnp.where(discarded, 2,
        jnp.where(hinted, 3, -1)))))


def detect_hint_details(env_state, smartbot_state, hand_size, num_colors, num_ranks):
    """Detect what hint I received by diffing colors_revealed/ranks_revealed."""
    my_id = smartbot_state.agent_id

    cur_colors = env_state.colors_revealed[my_id]  # (hs, nc)
    prev_colors = smartbot_state.prev_colors_revealed
    new_color_reveals = (cur_colors - prev_colors)  # 1 where newly revealed
    new_color_reveals = jnp.clip(new_color_reveals, 0, 1)
    any_color_reveal = new_color_reveals.sum() > 0.5

    cur_ranks = env_state.ranks_revealed[my_id]  # (hs, nr)
    prev_ranks = smartbot_state.prev_ranks_revealed
    new_rank_reveals = jnp.clip(cur_ranks - prev_ranks, 0, 1)
    any_rank_reveal = new_rank_reveals.sum() > 0.5

    color_touched = new_color_reveals.sum(axis=1) > 0.5  # (hs,)
    rank_touched = new_rank_reveals.sum(axis=1) > 0.5  # (hs,)

    touched = jnp.where(any_color_reveal, color_touched, rank_touched)
    received_hint = any_color_reveal | any_rank_reveal

    hint_type = jnp.where(any_color_reveal, 0, jnp.where(any_rank_reveal, 1, -1))
    hinted_color = jnp.argmax(new_color_reveals.sum(axis=0))
    hinted_rank = jnp.argmax(new_rank_reveals.sum(axis=0))
    hint_value = jnp.where(any_color_reveal, hinted_color, hinted_rank)

    return received_hint, touched, hint_type, hint_value


def find_convention_target(beliefs, touched, playable_matrix, hand_size):
    """Newest touched maybe-playable card. Used by both sender and receiver."""
    certainly_playable = batch_is_certainly_playable(beliefs, playable_matrix)
    playable_poss = (beliefs * playable_matrix[None, :, :]).sum(axis=(-2, -1))
    has_any = beliefs.sum(axis=(-2, -1)) > 0
    maybe_playable = (playable_poss > 0) & ~certainly_playable & has_any

    candidates = touched & maybe_playable
    slot_indices = jnp.arange(hand_size)
    target_idx = jnp.where(candidates, slot_indices, -1).max()
    has_target = target_idx >= 0

    return has_target, target_idx


def apply_newest_card_convention(my_beliefs, touched, playable_matrix,
                                 hand_size):
    """Conv 1: eliminate non-playable possibilities from convention target."""
    directly_revealed = batch_is_certainly_playable(my_beliefs, playable_matrix)
    any_direct_reveal = directly_revealed.any()

    has_target, target_idx = find_convention_target(
        my_beliefs, touched, playable_matrix, hand_size
    )

    should_apply = has_target & ~any_direct_reveal

    updated_beliefs = my_beliefs
    for slot in range(hand_size):
        is_target = (slot == target_idx) & should_apply
        new_slot = my_beliefs[slot] * playable_matrix
        new_slot = jnp.where(new_slot.sum() > 0, new_slot, my_beliefs[slot])
        updated_beliefs = updated_beliefs.at[slot].set(
            jnp.where(is_target, new_slot, updated_beliefs[slot])
        )

    return updated_beliefs


def apply_no_warning_inference(my_beliefs, discard_safe, partner_action_type,
                               next_playable_rank, played_count, card_counts,
                               unreachable_matrix, num_colors, num_ranks, hand_size):
    """Conv 2: if partner didn't warn me, my discard candidate is not valuable."""
    partner_did_not_warn = (partner_action_type >= 0) & (partner_action_type != 3)

    playable_matrix = compute_playable_matrix(next_playable_rank, num_colors, num_ranks)
    prob_worthless = batch_probability_worthless(my_beliefs, unreachable_matrix)
    is_playable = batch_is_certainly_playable(my_beliefs, playable_matrix)
    is_worthless = batch_is_certainly_worthless(my_beliefs, unreachable_matrix)

    candidate_fitness = jnp.where(
        is_playable | is_worthless, -100.0, prob_worthless
    )
    discard_candidate = jnp.argmax(candidate_fitness)

    new_discard_safe = jnp.where(
        partner_did_not_warn,
        discard_safe.at[discard_candidate].set(True),
        discard_safe
    )

    valuable_mask = jnp.zeros((num_colors, num_ranks))
    for c in range(num_colors):
        for r in range(num_ranks):
            total = card_counts[r]
            in_play = played_count[c, r]
            is_last = (in_play >= total - 1)
            not_unreach = unreachable_matrix[c, r] < 0.5
            valuable_mask = valuable_mask.at[c, r].set(
                (is_last & not_unreach).astype(jnp.float32)
            )

    candidate_beliefs = my_beliefs[discard_candidate]
    new_candidate = jnp.where(
        partner_did_not_warn,
        candidate_beliefs * (1.0 - valuable_mask),
        candidate_beliefs
    )
    new_candidate = jnp.where(
        new_candidate.sum() > 0, new_candidate, candidate_beliefs
    )
    updated_beliefs = my_beliefs.at[discard_candidate].set(new_candidate)

    return updated_beliefs, new_discard_safe


def apply_all_conventions(my_beliefs, env_state, smartbot_state,
                          playable_matrix, next_playable_rank, played_count,
                          card_counts, unreachable_matrix,
                          num_colors, num_ranks, hand_size,
                          conv_mask, discard_safe, public_beliefs):
    """Apply all conventions. conv_mask/discard_safe already shifted by caller."""
    partner_action = detect_partner_action(env_state, smartbot_state)

    received_hint, touched, hint_type, hint_value = detect_hint_details(
        env_state, smartbot_state, hand_size, num_colors, num_ranks
    )
    received_hint = received_hint & (partner_action == 3)

    # discard candidate computed on pre-convention beliefs
    prob_worthless_pre = batch_probability_worthless(my_beliefs, unreachable_matrix)
    is_play_pre = batch_is_certainly_playable(my_beliefs, playable_matrix)
    is_worth_pre = batch_is_certainly_worthless(my_beliefs, unreachable_matrix)
    candidate_fitness_pre = jnp.where(
        is_play_pre | is_worth_pre, -100.0, prob_worthless_pre
    )
    my_discard_candidate = jnp.argmax(candidate_fitness_pre)

    # conv 1: find target using public beliefs (bilateral agreement)
    directly_revealed_pub = batch_is_certainly_playable(public_beliefs, playable_matrix)
    any_direct_reveal_pub = directly_revealed_pub.any()
    has_target, target_idx = find_convention_target(
        public_beliefs, touched, playable_matrix, hand_size
    )
    conv1_should_apply = has_target & ~any_direct_reveal_pub

    # apply conv 1 to private beliefs
    for slot in range(hand_size):
        is_target = (slot == target_idx) & conv1_should_apply & received_hint
        new_slot = my_beliefs[slot] * playable_matrix
        new_slot = jnp.where(new_slot.sum() > 0, new_slot, my_beliefs[slot])
        my_beliefs = my_beliefs.at[slot].set(
            jnp.where(is_target, new_slot, my_beliefs[slot])
        )

    conv1_fired = received_hint & conv1_should_apply
    for slot in range(hand_size):
        is_target = (slot == target_idx) & conv1_fired
        new_slot_mask = conv_mask[slot] * playable_matrix
        new_slot_mask = jnp.where(
            new_slot_mask.sum() > 0, new_slot_mask, conv_mask[slot]
        )
        conv_mask = conv_mask.at[slot].set(
            jnp.where(is_target, new_slot_mask, conv_mask[slot])
        )

    # conv 2 positive: value-warning detection
    is_rank_hint = (hint_type == 1)
    candidate_was_touched = touched[my_discard_candidate]
    is_value_warning = received_hint & is_rank_hint & candidate_was_touched

    valuable_mask = jnp.zeros((num_colors, num_ranks))
    for c in range(num_colors):
        for r in range(num_ranks):
            total = card_counts[r]
            in_play_cr = played_count[c, r]
            is_last = (in_play_cr >= total - 1)
            not_unreach = unreachable_matrix[c, r] < 0.5
            valuable_mask = valuable_mask.at[c, r].set(
                (is_last & not_unreach).astype(jnp.float32)
            )

    # keep only valuable possibilities for warned card
    candidate_beliefs = my_beliefs[my_discard_candidate]
    warned_beliefs = candidate_beliefs * valuable_mask
    warned_beliefs = jnp.where(
        warned_beliefs.sum() > 0, warned_beliefs, candidate_beliefs
    )
    my_beliefs = jnp.where(
        is_value_warning,
        my_beliefs.at[my_discard_candidate].set(warned_beliefs),
        my_beliefs
    )

    warned_mask = conv_mask[my_discard_candidate] * valuable_mask
    warned_mask = jnp.where(
        warned_mask.sum() > 0, warned_mask, conv_mask[my_discard_candidate]
    )
    conv_mask = jnp.where(
        is_value_warning,
        conv_mask.at[my_discard_candidate].set(warned_mask),
        conv_mask
    )

    # conv 3: no-warning inference
    my_beliefs, new_discard_safe = apply_no_warning_inference(
        my_beliefs, discard_safe, partner_action,
        next_playable_rank, played_count, card_counts, unreachable_matrix,
        num_colors, num_ranks, hand_size
    )

    partner_did_not_warn = (partner_action >= 0) & (partner_action != 3)
    nw_candidate_fitness = jnp.where(
        is_play_pre | is_worth_pre, -100.0, prob_worthless_pre
    )
    nw_candidate = jnp.argmax(nw_candidate_fitness)
    not_valuable_mask = 1.0 - valuable_mask
    nw_new_mask = conv_mask[nw_candidate] * not_valuable_mask
    nw_new_mask = jnp.where(
        nw_new_mask.sum() > 0, nw_new_mask, conv_mask[nw_candidate]
    )
    conv_mask = jnp.where(
        partner_did_not_warn,
        conv_mask.at[nw_candidate].set(nw_new_mask),
        conv_mask
    )

    new_discard_safe = jnp.where(
        is_value_warning,
        new_discard_safe.at[my_discard_candidate].set(False),
        new_discard_safe
    )

    return my_beliefs, new_discard_safe, conv_mask


def update_smartbot_state(env_state, old_state, new_discard_safe,
                          action, conv_mask):
    my_id = old_state.agent_id
    return SmartBotState(
        agent_id=old_state.agent_id,
        prev_info_tokens_sum=jnp.sum(env_state.info_tokens).astype(jnp.int32),
        prev_score=env_state.score.astype(jnp.int32),
        prev_life_tokens_sum=jnp.sum(env_state.life_tokens).astype(jnp.int32),
        prev_num_discarded=env_state.num_cards_discarded.astype(jnp.int32),
        prev_turn=env_state.turn.astype(jnp.int32),
        prev_colors_revealed=env_state.colors_revealed[my_id],
        prev_ranks_revealed=env_state.ranks_revealed[my_id],
        discard_safe=new_discard_safe,
        conv_mask=conv_mask,
        prev_my_action=action.astype(jnp.int32),
    )
