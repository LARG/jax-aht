"""Other-Play color permutation for IPPO self-play (Hu et al. 2020).

Shuffle color labels in one agent's obs + inverse-shuffle their color
hint actions. If the policy learns to ignore which-color-is-which,
independently trained copies can cooperate without coordinating on
"red means play slot 3" conventions.

Getting the obs permutation right took me a while. The color-dependent
features are scattered across the 658-dim vector, not grouped by card.
Ranges below are for the default 2-player 5c/5r/hand5 config; mini
Hanabi is 3c/3r/hand3 and most widths change but the layout is the
same. The offsets here come from reading JaxMARL's hanabi_obs_utils
line by line. If JaxMARL ever refactors the obs layout these numbers
need to move.

  [0:125]     partner hand: (5 cards, 5 colors, 5 ranks). Permute axis 1.
  [125:127]   missing-card flags: no color
  [127:167]   deck remaining: no color
  [167:192]   fireworks: (5 colors, 5 ranks). Permute axis 0.
  [192:200]   info tokens: no color
  [200:203]   life tokens: no color
  [203:253]   discard pile: 5 groups of 10. Reorder groups.
  [253:308]   last action: has a 5-entry color_revealed block + an
              embedded card identity (5c x 5r). Permute both.
  [308:658]   belief features: 2 agents x 5 cards x 35. Each 35-slot
              card has 25 deductions (5c x 5r) + 5 color hints + 5
              rank hints. Permute the first two, leave ranks alone.

This module is gated behind USE_OTHER_PLAY in the IPPO config; default
is off so non-Hanabi envs (or vanilla IPPO on Hanabi) are unaffected.
"""
import jax
import jax.numpy as jnp


def sample_color_permutation(rng, num_colors=5):
    """Random permutation over color labels. Returns (num_colors,) int array."""
    return jax.random.permutation(rng, num_colors)


def permute_observation(obs, perm, num_colors=5, num_ranks=5, hand_size=5,
                        num_agents=2, max_info_tokens=8, max_life_tokens=3,
                        deck_size=50, discard_entries_per_color=10):
    """Apply a color permutation to every color-dependent obs section.

    `discard_entries_per_color` is the sum of num_cards_of_rank, so 10
    for default Hanabi ([3,2,2,2,1]) and 5 for mini-Hanabi ([2,2,1]).
    The first time I wrote this I forgot to permute the card_identity
    block in last_action and only the deck-discard part was working,
    which looked right until you noticed agents couldn't learn.
    """
    card_block = num_colors * num_ranks

    # offsets
    # hands_feats
    other_hands_size = (num_agents - 1) * hand_size * card_block  # 125
    hands_end = other_hands_size + num_agents  # +2 missing cards = 127

    # board_feats
    deck_n = deck_size - num_agents * hand_size  # 40
    fireworks_off = hands_end + deck_n  # 167
    board_end = fireworks_off + card_block + max_info_tokens + max_life_tokens  # 203

    # discards_feats
    discards_off = board_end  # 203
    discards_size = num_colors * discard_entries_per_color  # 50
    discards_end = discards_off + discards_size  # 253

    # last_action_feats
    la_off = discards_end  # 253
    # acting_player(num_agents) + move_type(4) + target(num_agents)
    color_rev_off = la_off + num_agents + 4 + num_agents  # 261
    # rank_revealed follows color_revealed
    rank_rev_off = color_rev_off + num_colors  # 266
    # reveal_outcome(hand_size) + position(hand_size) then card identity
    played_card_off = rank_rev_off + num_ranks + hand_size + hand_size  # 281
    la_end = played_card_off + card_block + 1 + 1  # 308

    # v0_belief_feats
    belief_off = la_end  # 308
    feats_per_card = card_block + num_colors + num_ranks  # 35

    # partner hand
    partner = obs[:other_hands_size].reshape(-1, num_colors, num_ranks)
    obs = obs.at[:other_hands_size].set(partner[:, perm, :].reshape(-1))

    # fireworks
    fw = obs[fireworks_off:fireworks_off + card_block].reshape(num_colors, num_ranks)
    obs = obs.at[fireworks_off:fireworks_off + card_block].set(fw[perm, :].reshape(-1))

    # discard pile
    disc = obs[discards_off:discards_end].reshape(num_colors, discard_entries_per_color)
    obs = obs.at[discards_off:discards_end].set(disc[perm, :].reshape(-1))

    # last-action color_revealed
    cr = obs[color_rev_off:color_rev_off + num_colors]
    obs = obs.at[color_rev_off:color_rev_off + num_colors].set(cr[perm])

    # last-action card identity (played or discarded)
    pc = obs[played_card_off:played_card_off + card_block].reshape(num_colors, num_ranks)
    obs = obs.at[played_card_off:played_card_off + card_block].set(pc[perm, :].reshape(-1))

    # belief feats: per-card deductions + color hints (leave rank hints alone)
    # 2 agents x 5 cards = 10 blocks of 35 features each
    n_cards_total = num_agents * hand_size
    belief_block = obs[belief_off:belief_off + n_cards_total * feats_per_card]
    belief_block = belief_block.reshape(n_cards_total, feats_per_card)

    # Deductions: first 25 entries per card -> (5, 5) colors x ranks
    deductions = belief_block[:, :card_block].reshape(n_cards_total, num_colors, num_ranks)
    deductions = deductions[:, perm, :].reshape(n_cards_total, card_block)

    # Color hints: next 5 entries per card
    color_hints = belief_block[:, card_block:card_block + num_colors]
    color_hints = color_hints[:, perm]

    # Rank hints: last 5 entries per card (unchanged)
    rank_hints = belief_block[:, card_block + num_colors:]

    new_belief = jnp.concatenate([deductions, color_hints, rank_hints], axis=1)
    obs = obs.at[belief_off:belief_off + n_cards_total * feats_per_card].set(
        new_belief.reshape(-1)
    )

    return obs


def permute_action(action, perm, inv_perm, hand_size=5, num_colors=5):
    """Inverse-permute color-hint actions so the env gets the right color.

    The agent sees permuted colors, picks a hint like "hint color 2",
    but the env still uses original colors, so we unmap before passing
    it through. Play/discard/rank-hint actions are color-blind and pass
    through unchanged.
    """
    hint_color_start = 2 * hand_size
    hint_color_end = hint_color_start + num_colors

    is_color_hint = (action >= hint_color_start) & (action < hint_color_end)
    color_idx = action - hint_color_start
    permuted_color = inv_perm[color_idx]
    corrected_action = jnp.where(
        is_color_hint,
        hint_color_start + permuted_color,
        action
    )
    return corrected_action


def inverse_permutation(perm):
    """Inverse of a permutation. `inv[perm[i]] == i`."""
    n = perm.shape[0]
    inv = jnp.zeros(n, dtype=jnp.int32)
    return inv.at[perm].set(jnp.arange(n))
