"""Hanabi color-permutation Other-Play (Hu et al. 2020).

Two layers:
  - Functions (sample_color_permutation, permute_observation,
    permute_action, inverse_permutation) used directly by BC training
    for offline data augmentation.
  - HanabiColorSymmetry: implements the EnvSymmetry interface so
    Other-Play during RL is just an env wrapper.
"""
import jax
import jax.numpy as jnp

from envs.common.symmetry import EnvSymmetry


def sample_color_permutation(rng, num_colors=5):
    return jax.random.permutation(rng, num_colors)


def permute_observation(obs, perm, num_colors=5, num_ranks=5, hand_size=5,
                        num_agents=2, max_info_tokens=8, max_life_tokens=3,
                        deck_size=50, discard_entries_per_color=10):
    card_block = num_colors * num_ranks

    other_hands_size = (num_agents - 1) * hand_size * card_block
    hands_end = other_hands_size + num_agents

    deck_n = deck_size - num_agents * hand_size
    fireworks_off = hands_end + deck_n
    board_end = fireworks_off + card_block + max_info_tokens + max_life_tokens

    discards_off = board_end
    discards_size = num_colors * discard_entries_per_color
    discards_end = discards_off + discards_size

    la_off = discards_end
    color_rev_off = la_off + num_agents + 4 + num_agents
    rank_rev_off = color_rev_off + num_colors
    played_card_off = rank_rev_off + num_ranks + hand_size + hand_size
    la_end = played_card_off + card_block + 1 + 1

    belief_off = la_end
    feats_per_card = card_block + num_colors + num_ranks

    partner = obs[:other_hands_size].reshape(-1, num_colors, num_ranks)
    obs = obs.at[:other_hands_size].set(partner[:, perm, :].reshape(-1))

    fw = obs[fireworks_off:fireworks_off + card_block].reshape(num_colors, num_ranks)
    obs = obs.at[fireworks_off:fireworks_off + card_block].set(fw[perm, :].reshape(-1))

    disc = obs[discards_off:discards_end].reshape(num_colors, discard_entries_per_color)
    obs = obs.at[discards_off:discards_end].set(disc[perm, :].reshape(-1))

    cr = obs[color_rev_off:color_rev_off + num_colors]
    obs = obs.at[color_rev_off:color_rev_off + num_colors].set(cr[perm])

    pc = obs[played_card_off:played_card_off + card_block].reshape(num_colors, num_ranks)
    obs = obs.at[played_card_off:played_card_off + card_block].set(pc[perm, :].reshape(-1))

    n_cards_total = num_agents * hand_size
    belief_block = obs[belief_off:belief_off + n_cards_total * feats_per_card]
    belief_block = belief_block.reshape(n_cards_total, feats_per_card)

    deductions = belief_block[:, :card_block].reshape(n_cards_total, num_colors, num_ranks)
    deductions = deductions[:, perm, :].reshape(n_cards_total, card_block)

    color_hints = belief_block[:, card_block:card_block + num_colors]
    color_hints = color_hints[:, perm]

    rank_hints = belief_block[:, card_block + num_colors:]

    new_belief = jnp.concatenate([deductions, color_hints, rank_hints], axis=1)
    obs = obs.at[belief_off:belief_off + n_cards_total * feats_per_card].set(
        new_belief.reshape(-1)
    )

    return obs


def permute_action(action, perm, hand_size=5, num_colors=5):
    hint_color_start = 2 * hand_size
    hint_color_end = hint_color_start + num_colors

    is_color_hint = (action >= hint_color_start) & (action < hint_color_end)
    color_idx = action - hint_color_start
    permuted_color = perm[color_idx]
    corrected_action = jnp.where(
        is_color_hint,
        hint_color_start + permuted_color,
        action
    )
    return corrected_action


def inverse_permutation(perm):
    n = perm.shape[0]
    inv = jnp.zeros(n, dtype=jnp.int32)
    return inv.at[perm].set(jnp.arange(n))


class HanabiColorSymmetry(EnvSymmetry):
    """Color-permutation symmetry for Hanabi (Hu et al. 2020).

    Hanabi's task semantics are invariant under any color permutation
    (red->blue, blue->green, etc.). Training under this augmentation
    discourages convention-specific behavior and improves zero-shot
    coordination with unseen partners.
    """

    def __init__(self, num_colors=5, num_ranks=5, hand_size=5,
                 num_agents=2, max_info_tokens=8, max_life_tokens=3,
                 num_cards_of_rank=None, **unused_env_kwargs):
        self.num_colors = num_colors
        self.num_ranks = num_ranks
        self.hand_size = hand_size
        self.num_agents = num_agents
        self.max_info_tokens = max_info_tokens
        self.max_life_tokens = max_life_tokens
        if num_cards_of_rank is None:
            num_cards_of_rank = [3, 2, 2, 2, 1][:num_ranks]
        cards_per_color = int(sum(num_cards_of_rank))
        self.deck_size = num_colors * cards_per_color
        self.discard_entries_per_color = cards_per_color
        self.hint_color_start = 2 * hand_size
        self.hint_color_end = self.hint_color_start + num_colors

    def sample(self, rng):
        return jax.random.permutation(rng, self.num_colors)

    def apply_to_obs(self, obs, perm):
        return permute_observation(
            obs, perm,
            num_colors=self.num_colors, num_ranks=self.num_ranks,
            hand_size=self.hand_size, num_agents=self.num_agents,
            max_info_tokens=self.max_info_tokens,
            max_life_tokens=self.max_life_tokens,
            deck_size=self.deck_size,
            discard_entries_per_color=self.discard_entries_per_color,
        )

    def apply_to_action(self, action, perm):
        return permute_action(
            action, perm,
            hand_size=self.hand_size, num_colors=self.num_colors,
        )

    def apply_to_avail_actions(self, avail, perm):
        # only color hints follow the permutation. discard, play, and
        # rank-hint slots stay as-is
        color_avail = avail[..., self.hint_color_start:self.hint_color_end]
        permuted = color_avail[..., perm]
        return avail.at[..., self.hint_color_start:self.hint_color_end].set(permuted)
