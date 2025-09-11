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

def run_single_episode(rng, env, agent_0_param, agent_0_policy,
                       agent_1_param, agent_1_policy,
                       max_episode_steps, agent_0_test_mode=False, agent_1_test_mode=False):
    # Reset the env.
    rng, reset_rng = jax.random.split(rng)
    init_obs, init_env_state = env.reset(reset_rng)
    init_done = {k: jnp.zeros((1), dtype=bool) for k in env.agents + ["__all__"]}
    init_act_onehot = {k: jnp.zeros((env.action_space(env.agents[i]).n)) for i, k in enumerate(env.agents)}
    init_reward = {k: jnp.zeros((1)) for i, k in enumerate(env.agents)}

    # Initialize hidden states. Agent id is passed as part of the hstate initialization to support heuristic agents.
    init_hstate_0 = agent_0_policy.init_hstate(1, aux_info={"agent_id": 0})
    init_hstate_1 = agent_1_policy.init_hstate(1, aux_info={"agent_id": 1})

    # Get available actions for agent 0 from environment state
    avail_actions = env.get_avail_actions(init_env_state.env_state)
    avail_actions = jax.lax.stop_gradient(avail_actions)
    avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
    avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)

    # Do one step to get a dummy info structure
    rng, act0_rng, act1_rng, step_rng = jax.random.split(rng, 4)

    init_joint_act_onehot = jnp.concatenate((init_act_onehot["agent_0"].reshape(1, 1, -1),
                                             init_act_onehot["agent_1"].reshape(1, 1, -1)), axis=-1)

    # Get ego action
    act_0, hstate_0 = agent_0_policy.get_action(
        params=agent_0_param,
        obs=init_obs["agent_0"].reshape(1, 1, -1),
        done=init_done["agent_0"].reshape(1, 1),
        avail_actions=avail_actions_0,
        hstate=init_hstate_0,
        rng=act0_rng,
        aux_obs=(init_joint_act_onehot, init_reward["agent_0"].reshape(1, 1, -1)),
        env_state=init_env_state,
        test_mode=agent_0_test_mode
    )
    act_0 = act_0.squeeze()

    # Get partner action using the underlying policy class's get_action method directly
    act_1, hstate_1 = agent_1_policy.get_action(
        params=agent_1_param,
        obs=init_obs["agent_1"].reshape(1, 1, -1),
        done=init_done["agent_1"].reshape(1, 1),
        avail_actions=avail_actions_1,
        hstate=init_hstate_1,  # shape of entry 0 is (1, 1, 8)
        rng=act1_rng,
        aux_obs=None,
        env_state=init_env_state,
        test_mode=agent_1_test_mode
    )
    act_1 = act_1.squeeze()

    both_actions = [act_0, act_1]
    env_act = {k: both_actions[i] for i, k in enumerate(env.agents)}
    env_act_onehot = {k: jax.nn.one_hot(both_actions[i], env.action_space(env.agents[i]).n) for i, k in enumerate(env.agents)}
    obs, env_state, reward, done, dummy_info = env.step(step_rng, init_env_state, env_act)

    # We'll use a scan to iterate steps until the episode is done.
    ep_ts = 1
    init_carry = (ep_ts, env_state, obs, rng, done, reward, env_act_onehot, hstate_0, hstate_1, dummy_info)
    def scan_step(carry, _):
        def take_step(carry_step):
            ep_ts, env_state, obs, rng, done, reward, act_onehot, hstate_0, hstate_1, last_info = carry_step
            # Get available actions for agent 0 from environment state
            avail_actions = env.get_avail_actions(env_state.env_state)
            avail_actions = jax.lax.stop_gradient(avail_actions)
            avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
            avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)

            joint_act_onehot = jnp.concatenate((act_onehot["agent_0"].reshape(1, 1, -1),
                                                act_onehot["agent_1"].reshape(1, 1, -1)), axis=-1)

            # Get ego action
            rng, act0_rng, act1_rng, step_rng = jax.random.split(rng, 4)
            act_0, hstate_0_next = agent_0_policy.get_action(
                params=agent_0_param,
                obs=obs["agent_0"].reshape(1, 1, -1),
                done=done["agent_0"].reshape(1, 1),
                avail_actions=avail_actions_0,
                hstate=hstate_0,
                rng=act0_rng,
                aux_obs=(joint_act_onehot, reward["agent_0"].reshape(1, 1, -1)),
                env_state=env_state,
                test_mode=agent_0_test_mode
            )
            act_0 = act_0.squeeze()

            # Get partner action with proper hidden state tracking
            act_1, hstate_1_next = agent_1_policy.get_action(
                params=agent_1_param,
                obs=obs["agent_1"].reshape(1, 1, -1),
                done=done["agent_1"].reshape(1, 1),
                avail_actions=avail_actions_1,
                hstate=hstate_1,
                rng=act1_rng,
                env_state=env_state,
                test_mode=agent_1_test_mode
            )
            act_1 = act_1.squeeze()

            both_actions = [act_0, act_1]
            env_act = {k: both_actions[i] for i, k in enumerate(env.agents)}
            env_act_onehot = {k: jax.nn.one_hot(both_actions[i], env.action_space(env.agents[i]).n) for i, k in enumerate(env.agents)}
            obs_next, env_state_next, reward, done_next, info_next = env.step(step_rng, env_state, env_act)

            return (ep_ts + 1, env_state_next, obs_next, rng, done_next, reward, env_act_onehot, hstate_0_next, hstate_1_next, info_next)

        ep_ts, env_state, obs, rng, done, reward, act_onehot, hstate_0, hstate_1, last_info = carry
        new_carry = jax.lax.cond(
            done["__all__"],
            lambda curr_carry: curr_carry, # True fn
            take_step, # False fn
            operand=carry
        )
        return new_carry, None

    final_carry, _ = jax.lax.scan(
        scan_step, init_carry, None, length=max_episode_steps)
    # Return the final info (which includes the episode return via LogWrapper).
    return final_carry[-1]

def run_episodes(rng, env, agent_0_param, agent_0_policy,
                 agent_1_param, agent_1_policy,
                 max_episode_steps, num_eps, agent_0_test_mode=False, agent_1_test_mode=False):
    '''Given a single ego agent and a single partner agent, run num_eps episodes in parallel using vmap.'''
    # Create episode-specific RNGs
    rngs = jax.random.split(rng, num_eps + 1)
    ep_rngs = rngs[1:]

    # Vectorize run_single_episode over the first argument (rng)
    vmap_run_single_episode = jax.jit(jax.vmap(
        lambda ep_rng: run_single_episode(
            ep_rng, env, agent_0_param, agent_0_policy,
            agent_1_param, agent_1_policy, max_episode_steps,
            agent_0_test_mode, agent_1_test_mode
        )
    ))
    # Run episodes in parallel
    all_outs = vmap_run_single_episode(ep_rngs)
    return all_outs  # each leaf has shape (num_eps, ...)

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
