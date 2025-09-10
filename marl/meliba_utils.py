from typing import NamedTuple

import jax
import jax.numpy as jnp

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

def transform_timestep_to_batch_vmap(array, pad_value=0.0):
    """
    Transform array from (timestep, feat) to (batch, k_timesteps, feat) using vmap

    Args:
        array: JAX array of shape (H+1, feat) where timesteps go from 0 to H
        pad_value: Value to use for padding shorter sequences (default: 0.0)

    Returns:
        JAX array of shape (H, H+1, feat) where:
        - batch dimension has size H
        - batch[k] contains timesteps k to H, padded to length H+1
        - padding is applied at the end of shorter sequences
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

        return result

    # Create starting indices [0, 1, 2, ..., H-1]
    start_indices = jnp.arange(H)

    # Use vmap to apply the function to each starting index
    result = jax.vmap(get_subsequence_for_start_idx)(start_indices)

    return result

def transform_timestep_to_batch_indexing(array, pad_value=0.0):
    """
    Transform using indexing (for comparison) - this is the previous working version
    """
    H_plus_1, feat = array.shape
    H = H_plus_1 - 1
    batch_size = H
    max_seq_length = H_plus_1

    # Create indices for gathering
    batch_indices = jnp.arange(batch_size)[:, None]  # (H, 1)
    seq_indices = jnp.arange(max_seq_length)[None, :]  # (1, H+1)

    # Actual timestep indices for each batch element
    timestep_indices = seq_indices + batch_indices  # (H, H+1)

    # Create mask for valid (non-padded) positions
    valid_mask = timestep_indices < H_plus_1  # (H, H+1)

    # Clamp indices to valid range for gathering
    safe_indices = jnp.clip(timestep_indices, 0, H_plus_1 - 1)

    # Gather values and apply padding
    gathered = array[safe_indices]  # (H, H+1, feat)
    result = jnp.where(valid_mask[..., None], gathered, pad_value)

    return result

# Example usage
if __name__ == "__main__":
    # Example: H=3, so timesteps 0,1,2,3 and we want batch size 3
    H = 3
    feat_dim = 2

    # Create sample data: (4, 2) array representing timesteps 0-3
    array = jnp.array([
        [1.0, 10.0],  # timestep 0
        [2.0, 20.0],  # timestep 1
        [3.0, 30.0],  # timestep 2
        [4.0, 40.0],  # timestep 3
    ])

    print("Original array shape:", array.shape)
    print("Original array:\n", array)

    # Transform using vmap
    print("\n=== Using vmap approach ===")
    result_vmap = transform_timestep_to_batch_vmap(array, pad_value=0.0)

    print(f"Transformed array shape: {result_vmap.shape}")
    print("Transformed array (with padding):")
    for k in range(result_vmap.shape[0]):
        valid_length = H + 1 - k
        print(f"Batch {k} (timesteps {k} to {H}, padded): valid_length={valid_length}")
        print(f"  Data: {result_vmap[k]}")
        print()

    # Transform using indexing for comparison
    print("=== Using indexing approach ===")
    result_indexing = transform_timestep_to_batch_indexing(array, pad_value=0.0)

    # Verify both methods give the same result
    print(f"Both methods give identical results: {jnp.allclose(result_vmap, result_indexing)}")

    # Performance comparison (rough)
    print("\n=== Performance comparison ===")
    import time

    # Warm up
    for _ in range(10):
        transform_timestep_to_batch_vmap(array)
        transform_timestep_to_batch_indexing(array)

    # Time vmap version
    start = time.time()
    for _ in range(1000):
        result_vmap = transform_timestep_to_batch_vmap(array)
    vmap_time = time.time() - start

    # Time indexing version
    start = time.time()
    for _ in range(1000):
        result_indexing = transform_timestep_to_batch_indexing(array)
    indexing_time = time.time() - start

    print(f"vmap version: {vmap_time:.4f}s")
    print(f"indexing version: {indexing_time:.4f}s")
    print(f"Speedup: {vmap_time/indexing_time:.2f}x ({'indexing faster' if indexing_time < vmap_time else 'vmap faster'})")


# def create_full_sliding_window(input_jax_array, episode_ids):
#     batch_size, sliding_window_size = input_jax_array.shape[0], input_jax_array.shape[1]

#     # Assume that input_jax_array shape is (batch_size, sliding_window_size, feature_dim)
#     # and episode_ids shape is (batch_size,sliding_window_size)

#     # Pad extra rows of -1s to the end of the input array on the time dimension
#     # to handle sliding windows that extend beyond the current episode
#     # Similarly, pad episode_ids with -1s at the end of the time dimension

#     padded_input = jnp.pad(input_jax_array, ((0, 0), (0, sliding_window_size - 1), (0, 0)), mode='constant', constant_values=-1)
#     padded_episode_ids = jnp.pad(episode_ids, ((0, 0), (0, sliding_window_size - 1)), mode='constant', constant_values=-1)
#     sliding_window_idxs = jnp.arange(sliding_window_size)[:, None] + jnp.arange(sliding_window_size)[None, :]

#     def get_per_sliding_window_input(input_array):
#         return input_array[sliding_window_idxs, :]

#     def get_per_example_window_eps_id(episode_id_array):
#         return episode_id_array[sliding_window_idxs]

#     return jnp.stack(jax.vmap(get_per_sliding_window_input)(padded_input)), jnp.stack(jax.vmap(get_per_example_window_eps_id)(padded_episode_ids))

# if __name__ == "__main__":
#     a = jnp.arange(150).reshape([5, 10, 3])
#     episode_ids = jnp.array([
#         [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
#         [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
#         [0, 0, 1, 1, 1, 2, 2, 2, 2, 2],
#         [0, 0, 1, 1, 1, 2, 2, 2, 3, 3],
#         [0, 0, 1, 1, 2, 2, 2, 2, 2, 2],
#     ])
#     input_in_windows, eps_id_in_windows = create_full_sliding_window(a, episode_ids)

#     # Within each sliding window, create a mask that indicates which
#     # elements belong to the same episode as the first element in the window
#     # This later can be used to mask out losses that come from invalid timesteps

#     per_batch_and_timestep_valid_mask = (
#         eps_id_in_windows == eps_id_in_windows[:, :, 0][:, :, None]
#     )
#     jax.debug.breakpoint()
