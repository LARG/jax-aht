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
    """
    A RNN module that can be scanned over time.

    It resets its state based on the `dones` signal and
    resets to the provided `hiddens` state when `done` is True.
    """
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

def transform_timestep_to_k_batch(array, pad_value=0.0, return_mask=False):
    """
    Transform array from (timestep, feat) to (k_batch, timesteps, feat)

    Each k_batch element contains a subsequence starting at each timestep,
    padded with pad_value where necessary.

    Args:
        array: jnp.array, array of shape (timesteps, feat) where timesteps go from 0 to H
        pad_value: float, Value to use for padding shorter sequences (default: 0.0)
        return_mask: bool, If True, also return a mask indicating valid positions (default: False)

    Returns:
        result: jnp.array, array of shape (H-1, H, feat)
        mask (optional): jnp.array, array of shape (H-1, H) with True for valid positions, False for padding
    """
    H_timestep, _ = array.shape
    H_timestep_minus_1 = H_timestep - 1

    def get_subsequence_for_start_idx(start_idx):
        """
        Get subsequence starting at start_idx, padded to full length

        Args:
            start_idx: int, starting index for the subsequence

        Returns:
            result: jnp.array, array of shape (H, feat)
            mask (optional): jnp.array, array of shape (H,) with True for valid positions, False for padding
        """
        # Create indices for this subsequence
        # [start_idx, start_idx+1, ..., start_idx+H]
        indices = jnp.arange(H_timestep) + start_idx

        # Create mask for valid positions (within original array bounds)
        valid_mask = indices < H_timestep

        # Clamp indices to valid range
        safe_indices = jnp.clip(indices, 0, H_timestep - 1)

        # Gather values
        gathered = array[safe_indices]  # Shape: (H+1, feat)

        # Apply padding where mask is False
        result = jnp.where(valid_mask[:, None], gathered, pad_value)

        if return_mask:
            return result, valid_mask
        else:
            return result

    # Create starting indices [0, 1, 2, ..., H-1]
    start_indices = jnp.arange(H_timestep_minus_1)

    # Use vmap to apply the function to each starting index
    if return_mask:
        results, masks = jax.vmap(get_subsequence_for_start_idx)(start_indices)
        return results, masks
    else:
        results = jax.vmap(get_subsequence_for_start_idx)(start_indices)
        return results

def shift_padding_to_front_vectorized(data, mask):
    """
    Shift padding in data to the front based on the mask.

    Args:
        data: jnp.array of shape (batch_size, seq_len, feat_dim)
        mask: jnp.array of shape (batch_size, seq_len) with boolean values

    Returns:
        shifted_data: jnp.array of shape (batch_size, seq_len, feat_dim)
        new_mask: jnp.array of shape (batch_size, seq_len) with boolean values
    """
    _, seq_len, _ = data.shape

    # Count valid elements per batch
    # (batch_size,)
    valid_counts = jnp.sum(mask, axis=1)

    # Number of padding positions per batch
    pad_counts = seq_len - valid_counts

    # Create indices for the new positions
    # (batch, 1)
    # batch_indices = jnp.arange(batch_size)[:, None]
    # (1, seq_len)
    pos_indices = jnp.arange(seq_len)[None, :]

    # Determine which positions should be padding vs valid data
    is_padding_position = pos_indices < pad_counts[:, None]  # (batch, seq_len)
    new_mask = ~is_padding_position

    # For valid data positions, determine which original valid position to use
    valid_data_index = pos_indices - pad_counts[:, None]  # (batch, seq_len)

    # Get the mapping from valid_data_index to actual array positions
    # Use jnp.where to get positions of valid elements for each batch
    def get_valid_positions(single_mask):
        return jnp.where(single_mask, size=seq_len, fill_value=0)[0]

    # (batch, seq_len)
    valid_position_maps = jax.vmap(get_valid_positions)(mask)

    # Create the source indices for gathering
    # Clamp valid_data_index to valid range [0, seq_len-1]
    safe_valid_indices = jnp.clip(valid_data_index, 0, seq_len - 1)
    # (batch, seq_len)
    source_positions = jnp.take_along_axis(
        valid_position_maps, safe_valid_indices, axis=1
    )

    # Gather the data
    # (batch, seq_len, feat_dim)
    gathered_data = jnp.take_along_axis(
        data, source_positions[..., None], axis=1
    )

    # Zero out padding positions
    shifted_data = jnp.where(
        new_mask[..., None], gathered_data, 0.0
    )

    return shifted_data, new_mask

def fill_to_first_true(array):
    """
    Fill everything before and including the first True to be True,
    everything after to be False.

    Args:
        array: JAX boolean array of shape (timesteps, batch_size)

    Returns:
        modified_array: JAX boolean array of same shape as input
    """

    # Handle empty array case
    if array.size == 0:
        return array

    def fill_to_first_true(array):
        """
        VMAP function

        Fill everything before and including the first True to be True,
        everything after to be False.

        Args:
            array: JAX boolean array of shape (timesteps,)

        Returns:
            modified_array: JAX boolean array of same shape as input
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

    # Apply to each batch element using vmap
    fill_batch = jax.vmap(fill_to_first_true, in_axes=0, out_axes=0)

    return fill_batch(array)
