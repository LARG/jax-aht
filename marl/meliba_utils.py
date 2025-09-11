import functools
import numpy as np

from typing import NamedTuple

import jax
import jax.numpy as jnp
import flax.linen as nn

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray
    joint_act_onehot: jnp.ndarray
    prev_action_onehot: jnp.ndarray
    partner_action: jnp.ndarray
    partner_action_onehot: jnp.ndarray

class DecoderScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, hiddens, dones = x
        rnn_state = jnp.where(
            dones[:, np.newaxis],
            hiddens,
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

def transform_timestep_to_batch_vmap(array, pad_value=0.0, return_mask=False):
    """
    Transform array from (timestep, feat) to (batch, k_timesteps, feat) using vmap

    Args:
        array: JAX array of shape (H+1, feat) where timesteps go from 0 to H
        pad_value: Value to use for padding shorter sequences (default: 0.0)
        return_mask: If True, also return a mask indicating valid positions

    Returns:
        result: JAX array of shape (H, H+1, feat)
        mask (optional): JAX array of shape (H, H+1) with True for valid positions, False for padding
    """
    H_plus_1, feat = array.shape
    H = H_plus_1 - 1

    def get_subsequence_for_start_idx(start_idx):
        """Get subsequence starting at start_idx, padded to full length"""
        # Create indices for this subsequence
        indices = jnp.arange(H_plus_1) + start_idx  # [start_idx, start_idx+1, ..., start_idx+H]

        # Create mask for valid positions (within original array bounds)
        valid_mask = indices < H_plus_1

        # Clamp indices to valid range
        safe_indices = jnp.clip(indices, 0, H_plus_1 - 1)

        # Gather values
        gathered = array[safe_indices]  # Shape: (H+1, feat)

        # Apply padding where mask is False
        result = jnp.where(valid_mask[:, None], gathered, pad_value)

        if return_mask:
            return result, valid_mask
        else:
            return result

    # Create starting indices [0, 1, 2, ..., H-1]
    start_indices = jnp.arange(H)

    # Use vmap to apply the function to each starting index
    if return_mask:
        results, masks = jax.vmap(get_subsequence_for_start_idx)(start_indices)
        return results, masks
    else:
        results = jax.vmap(get_subsequence_for_start_idx)(start_indices)
        return results

def shift_padding_to_front_vectorized(data, mask):
    """
    More efficient vectorized version that avoids loops
    """
    batch_size, seq_len, feat_dim = data.shape

    # Count valid elements per batch
    valid_counts = jnp.sum(mask, axis=1)  # (batch_size,)
    pad_counts = seq_len - valid_counts   # Number of padding positions per batch

    # Create indices for the new positions
    batch_indices = jnp.arange(batch_size)[:, None]  # (batch, 1)
    pos_indices = jnp.arange(seq_len)[None, :]        # (1, seq_len)

    # Determine which positions should be padding vs valid data
    is_padding_position = pos_indices < pad_counts[:, None]  # (batch, seq_len)
    new_mask = ~is_padding_position

    # For valid data positions, determine which original valid position to use
    valid_data_index = pos_indices - pad_counts[:, None]  # (batch, seq_len)

    # Get the mapping from valid_data_index to actual array positions
    # Use jnp.where to get positions of valid elements for each batch
    def get_valid_positions(single_mask):
        return jnp.where(single_mask, size=seq_len, fill_value=0)[0]

    valid_position_maps = jax.vmap(get_valid_positions)(mask)  # (batch, seq_len)

    # Create the source indices for gathering
    # Clamp valid_data_index to valid range [0, seq_len-1]
    safe_valid_indices = jnp.clip(valid_data_index, 0, seq_len - 1)
    source_positions = jnp.take_along_axis(
        valid_position_maps, safe_valid_indices, axis=1
    )  # (batch, seq_len)

    # Gather the data
    gathered_data = jnp.take_along_axis(
        data, source_positions[..., None], axis=1
    )  # (batch, seq_len, feat_dim)

    # Zero out padding positions
    shifted_data = jnp.where(
        new_mask[..., None], gathered_data, 0.0
    )

    return shifted_data, new_mask

def fill_to_first_true(array):

    if array.size == 0:
        return array

    def fill_to_first_true(array):
        """
        Fill everything before and including the first True to be True,
        everything after to be False.
        """
        # Find the first True position
        first_true_idx = jnp.argmax(array)

        # Create a mask: True up to and including first_true_idx, False after
        # But only if there actually is a True in the array
        has_true = jnp.any(array)
        indices = jnp.arange(len(array))
        mask = indices <= first_true_idx

        # Only apply the mask if there's actually a True in the array
        return jnp.where(has_true, mask, array)

    # Apply to each column (batch element) using vmap
    fill_batch = jax.vmap(fill_to_first_true, in_axes=0, out_axes=0)

    return fill_batch(array)
