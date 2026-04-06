'''Behavior cloning human proxy agent for Overcooked-v1. Implementation 
based on https://github.com/overcookedv2/experiments/tree/main

Results:
'''
from functools import partial
from pathlib import Path

import chex
from flax import linen as nn
from flax.linen.initializers import constant, orthogonal
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from jaxmarl.environments.overcooked.layouts import overcooked_layouts
from jaxmarl.environments.overcooked.overcooked import Actions

from envs import make_env
from envs.overcooked_v1_v1.overcooked_v1 import OvercookedV1
from envs.overcooked_v1_v1.overcooked_wrapper import OvercookedWrapper
from envs.overcooked_v1_v1.augmented_layouts import augmented_layouts
from agents.agent_interface import AgentPolicy
from agents.overcooked_v1_v1.bc_featurizer import BCFeaturizer


ACTION_DIM = len(Actions)
STATE_HISTORY_LEN = 3


# Layout name to base directory mapping
# Get path to repo root
REPO_ROOT = Path(__file__).parent.parent.parent
LAYOUT_TO_BASE_DIR = {
    "cramped_room": REPO_ROOT / "eval_teammates/overcooked-v1/cramped_room/bc_proxy/models",
    "asymm_advantages": REPO_ROOT / "eval_teammates/overcooked-v1/asymm_advantages/bc_proxy/models",
    "counter_circuit": REPO_ROOT / "eval_teammates/overcooked-v1/counter_circuit/bc_proxy/models",
    "coord_ring": REPO_ROOT / "eval_teammates/overcooked-v1/coord_ring/bc_proxy/models",
    "forced_coord": REPO_ROOT / "eval_teammates/overcooked-v1/forced_coord/bc_proxy/models",
}


class BCPolicy(AgentPolicy):
    """Behavior Cloning policy that directly implements the AgentPolicy interface."""
    def __init__(self, layout_name, using_log_wrapper=True):
        """
        Initialize BC policy with layout name.
        The agent_id must be provided through init_hstate via aux_info.
        
        Args:
            layout_name: Name of the layout (e.g., "cramped_room")
            using_log_wrapper: If True, assume that the agent is interacting with the wrapped Overcooked-v1
                environment. If False, assume that the agent is interacting with the original Overcooked-v1
                environment. 
        """
        super().__init__(action_dim=ACTION_DIM, obs_dim=None)
        self.network = BCModel(action_dim=ACTION_DIM, hidden_dims=(64, 64))
        self.layout_name = layout_name
        self.run_id = 0 # checkpoints are available from 0 to 4
        self.unblock_if_stuck = True
        self.using_log_wrapper = using_log_wrapper
        
        # Load parameters from checkpoint
        self.params = self._load_params()
        
        # we don't need the augmented layout here because we only need the wall map info
        layout = overcooked_layouts[layout_name] 
        self.featurizer = BCFeaturizer(layout, num_pots=2, max_steps=400)
    
    def _load_params(self):
        """Load parameters from checkpoint based on layout name and run ID."""
        if self.layout_name not in LAYOUT_TO_BASE_DIR:
            raise ValueError(f"Layout '{self.layout_name}' not found in LAYOUT_TO_BASE_DIR mapping")
        
        base_dir = Path(LAYOUT_TO_BASE_DIR[self.layout_name])
        model_dir = base_dir / str(self.run_id)
        
        orbax_checkpointer = ocp.PyTreeCheckpointer()
        ckpt = orbax_checkpointer.restore(model_dir, item=None)
        _, params = ckpt["config"], ckpt["params"]
        return params

    @partial(jax.jit, static_argnums=(0,))
    def get_action(self, params, obs, done, avail_actions, hstate, rng, 
                   aux_obs=None, env_state=None, test_mode=True):
        """
        Get action from BC policy. The BC policy does much better in deterministic mode
        (test_mode=True) than in stochastic mode (test_mode=False).
        
        Args:
            params: Ignored (BC policy manages its own parameters)
            obs: Ignored because the featurized obs will be computed from the state and used instead.
            done: Done flag from previous env step
            avail_actions: Ignored (BC policy doesn't use available actions)
            hstate: Hidden state
            rng: Random key
            aux_obs: Ignored
            env_state: wrapped Overcooked environment state
            test_mode: If True, use deterministic action selection
        
        Returns:
            action, new_hstate
        """
        # Extract agent_id from hidden state
        policy_hstate = BCPolicyHState.from_numpy(hstate[0])
        agent_id = policy_hstate.agent_id
        
        if self.using_log_wrapper:
            featurized_obs_array = self.featurizer.featurize_state_as_array(env_state.env_state.env_state)
        else:
            featurized_obs_array = self.featurizer.featurize_state_as_array(env_state.env_state)

        feat_obs = featurized_obs_array[agent_id]

        logits = self.network.apply({"params": self.params}, feat_obs)
        action_probs = jax.nn.softmax(logits)

        # Reset hstate if done - need to preserve agent_id
        def _reset_hstate():
            return self.init_hstate(1, aux_info={"agent_id": agent_id})
        
        hstate = jnp.where(
            done, _reset_hstate(), hstate
        )

        if self.unblock_if_stuck:
            def _handle_stuck(bc_hstate, feat_obs, action_probs):
                new_action_probs = self._remove_indices_and_renormalize(
                    action_probs, bc_hstate.actions
                )

                is_stuck = bc_hstate.is_stuck(feat_obs)
                return jnp.where(is_stuck, new_action_probs, action_probs)

            # Extract BC state from policy hidden state after potential reset
            policy_hstate_updated = BCPolicyHState.from_numpy(hstate[0])
            bc_hstate = BCHState.from_numpy(policy_hstate_updated.bc_state)
            action_probs = _handle_stuck(bc_hstate, feat_obs, action_probs)


        # Action computation from logits
        argmax_action = jnp.argmax(action_probs, axis=-1)
        def _sample_action(key, action_probs):
            return jax.random.choice(key, self.action_dim, axis=-1, p=action_probs)

        sampled_action = _sample_action(rng, action_probs)
        action = jax.lax.cond(test_mode, lambda: argmax_action, lambda: sampled_action)

        if self.unblock_if_stuck:
            def _append_and_to_numpy(bc_hstate, action, feat_obs):
                return bc_hstate.append(action, feat_obs).to_numpy()

            # Update the BC state and create new policy hidden state
            updated_bc_state = _append_and_to_numpy(bc_hstate, action, feat_obs)
            new_policy_hstate = BCPolicyHState(agent_id=agent_id, bc_state=updated_bc_state)
            hstate = new_policy_hstate.to_numpy()[jnp.newaxis, ...]

        return action, hstate

    @staticmethod
    def _remove_indices_and_renormalize(probs, indices):
        """
        Remove specified action indices from probability distribution and renormalize.
        
        Used for anti-stuck mechanism: when agent is stuck, removes probabilities of 
        recently taken actions to force exploration of different actions.
        
        Args:
            probs: Action probability distribution (1D array)
            indices: Action indices to remove/zero out (1D array)
            
        Returns:
            Renormalized probability distribution with specified indices set to 0
        """
        assert probs.ndim == 1
        assert indices.ndim == 1

        # Zero out probabilities for the specified action indices
        probs = probs.at[indices].set(0)
        
        # Create backup uniform distribution (excluding the removed indices)
        alt_probs = jnp.ones_like(probs).at[indices].set(0)
        
        # Use modified probs if they sum to something, otherwise use backup
        sum_probs = probs.sum()
        probs = jax.lax.select(sum_probs > 0, probs, alt_probs)
        
        # Renormalize to ensure probabilities sum to 1
        return probs / probs.sum()

    def init_hstate(self, batch_size: int, aux_info: dict) -> chex.Array:
        """Initialize hidden state for the BC policy."""
        
        agent_id = aux_info["agent_id"]
        
        if self.unblock_if_stuck:
            # Create policy hidden state with agent_id and empty BC state
            policy_hstate = BCPolicyHState.init_empty(agent_id)
            hstate = policy_hstate.to_numpy()
            hstate = jnp.repeat(hstate[jnp.newaxis, ...], batch_size, axis=0)
        else:
            # For non-unblock case, just store agent_id
            policy_hstate = BCPolicyHState(agent_id=agent_id, bc_state=jnp.array([]))
            hstate = policy_hstate.to_numpy()
            hstate = jnp.repeat(hstate[jnp.newaxis, ...], batch_size, axis=0)

        return hstate
    
# Define the model architecture
class BCModel(nn.Module):
    action_dim: int
    hidden_dims: tuple

    @nn.compact
    def __call__(self, x):
        for dim in self.hidden_dims:
            x = nn.Dense(
                dim,
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
            )(x)
            x = nn.relu(x)
        x = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(x)
        return x


@chex.dataclass 
class BCPolicyHState:
    agent_id: int
    bc_state: chex.Array  # Serialized BCHState
    
    @staticmethod
    def from_numpy(arr: jnp.ndarray):
        """Deserialize from numpy array: [agent_id, bc_state_data...]"""
        agent_id = arr[0].astype(jnp.int32)
        bc_state_data = arr[1:]
        return BCPolicyHState(agent_id=agent_id, bc_state=bc_state_data)
    
    def to_numpy(self):
        """Serialize to numpy array: [agent_id, bc_state_data...]"""
        agent_id_array = jnp.array([self.agent_id], dtype=jnp.float32)
        return jnp.concatenate([agent_id_array, self.bc_state])
    
    @staticmethod
    def init_empty(agent_id: int):
        """Initialize empty hidden state with agent_id"""
        bc_state = BCHState.init_empty().to_numpy()
        return BCPolicyHState(agent_id=agent_id, bc_state=bc_state)
    
    
@chex.dataclass
class BCHState:
    '''Tracks the history of actions, positions, and directions.
    TODO: simplify class to only track action history because that's 
    the only thing we need. 
    '''
    actions: chex.Array  # Shape: (STATE_HISTORY_LEN,)
    pos: chex.Array  # Shape: (STATE_HISTORY_LEN, 2)
    dir: chex.Array  # Shape: (STATE_HISTORY_LEN, 4)

    @staticmethod
    def from_numpy(arr: jnp.ndarray):
        history_len = STATE_HISTORY_LEN
        idx = 0

        actions_len = history_len
        pos_len = history_len * 2
        dir_len = history_len * 4

        actions = arr[idx : idx + actions_len].reshape((history_len,)).astype(jnp.int32)
        idx += actions_len

        pos = arr[idx : idx + pos_len].reshape((history_len, 2)).astype(jnp.int32)
        idx += pos_len

        dir = arr[idx : idx + dir_len].reshape((history_len, 4)).astype(jnp.int32)
        idx += dir_len

        return BCHState(
            actions=actions,
            pos=pos,
            dir=dir,
        )

    def to_numpy(self):
        actions = self.actions.reshape(-1)
        pos = self.pos.reshape(-1)
        dir = self.dir.reshape(-1)
        data = jnp.concatenate([actions, pos, dir])
        return data

    @staticmethod
    def init_empty():
        return BCHState(
            actions=jnp.zeros((STATE_HISTORY_LEN,), dtype=jnp.int32),
            pos=jnp.zeros((STATE_HISTORY_LEN, 2), dtype=jnp.int32),
            dir=jnp.zeros((STATE_HISTORY_LEN, 4), dtype=jnp.int32),
        )

    @staticmethod
    def _extract_pos_dir_from_obs(obs):
        pos = obs[-2:]
        dir = obs[:4]
        return pos, dir

    def append(self, action, obs):
        pos, dir = self._extract_pos_dir_from_obs(obs)

        # Update actions
        actions = jnp.concatenate(
            [self.actions[1:], jnp.array([action], dtype=self.actions.dtype)]
        )
        # Update self state
        pos = jnp.concatenate([self.pos[1:], jnp.array([pos], dtype=self.pos.dtype)])
        dir = jnp.concatenate([self.dir[1:], jnp.array([dir], dtype=self.dir.dtype)])

        # Return updated state
        return BCHState(
            actions=actions,
            pos=pos,
            dir=dir,
        )

    def is_stuck(self, obs):
        pos, dir = self._extract_pos_dir_from_obs(obs)

        # Get the recent positions and directions
        pos_history = self.pos  # Shape: (stuck_time, 2)
        dir_history = self.dir  # Shape: (stuck_time, 4)

        # Check if all positions and directions are equal
        pos_equal = jnp.all(pos_history == pos)
        dir_equal = jnp.all(dir_history == dir)

        return pos_equal & dir_equal

def test_bc_policy(layout_name: str, num_episodes=64, test_mode=True, do_reward_shaping=True):
    """Load and evaluate a BC policy on Overcooked-v1"""
    assert layout_name in LAYOUT_TO_BASE_DIR, f"Layout {layout_name} not found in LAYOUT_TO_BASE_DIR"

    num_timesteps = 400
    seed = 0
    
    print(f"Loading BC policy for layout: {layout_name}")
    
    # Load environment
    env = make_env(env_name="overcooked-v1", 
                   env_kwargs={
                       "layout": layout_name, 
                       "max_steps": num_timesteps,
                       "random_obj_state": True,
                       "do_reward_shaping": do_reward_shaping,
                       "reward_shaping_params": {
                            "PLACEMENT_IN_POT_REW": .5,
                            "PLATE_PICKUP_REWARD": .1,
                            "SOUP_PICKUP_REWARD": 1.0,
                            "ONION_PICKUP_REWARD": .1,
                            "COUNTER_PICKUP_REWARD": 0,
                            "COUNTER_DROP_REWARD": 0,
                        }
                    })
    # env = OvercookedWrapper(layout=augmented_layouts[layout_name], max_steps=num_timesteps)
    
    policy0 = BCPolicy(layout_name, using_log_wrapper=False)
    policy1 = BCPolicy(layout_name, using_log_wrapper=False)
    
    # Initialize random key
    key = jax.random.PRNGKey(seed)
    
    # Evaluation function
    @partial(jax.jit, static_argnums=(0, 1, 2, 3))
    def rollout_episode(env, policy0, policy1, num_timesteps, key):
        """Run a single episode and return the total reward."""
        
        policy_dict = {0: policy0, 1: policy1}
        # Reset environment
        key, reset_key = jax.random.split(key)
        obs, state = env.reset(reset_key)
        
        # Initialize hidden states for both agents
        hstates = {f"agent_{i}": policy_dict[i].init_hstate(1, aux_info={"agent_id": i}) for i in range(2)}
        
        # Initialize done flags
        done = {"agent_0": False, "agent_1": False, "__all__": False}
        total_reward = 0.0
        
        # Single timestep function
        def timestep(carry, timestep_key):
            obs, state, done, hstates, total_reward = carry
            
            # Skip if episode is already done
            def step_fn():
                # Split key for this timestep
                _, step_key = jax.random.split(timestep_key)
                
                # Compute actions for both agents
                actions = {}
                new_hstates = {}
                
                # Split key for each agent
                agent_keys = jax.random.split(step_key, 2)
                
                for i in range(2):
                    agent_id = f"agent_{i}"
                    # agent_obs = jnp.array([obs[agent_id]])  # shape (1, ...)
                    agent_done = done[agent_id] # jnp.array([done[agent_id]])  # shape (1,)
                    agent_hstate = hstates[agent_id]  # shape (1, ...)
                    
                    # Compute action using AgentPolicy interface
                    action, new_hstate = policy_dict[i].get_action(
                        None,  # params (ignored)
                        None,  # obs (ignored)
                        agent_done, 
                        None,  # avail_actions (ignored)
                        agent_hstate, 
                        agent_keys[i],
                        env_state=state,  # always pass state for featurizer
                        test_mode=test_mode
                    )
                    # if hasattr(action, 'shape') and action.shape[0] == 1:
                    #     action = action[0]
                    
                    actions[agent_id] = action
                    new_hstates[agent_id] = new_hstate
                
                # Step environment
                new_obs, new_state, reward, new_done, info = env.step(step_key, state, actions)
                
                # Update total reward
                episode_reward = sum(reward.values())
                new_total_reward = total_reward + episode_reward
                
                return new_obs, new_state, new_done, new_hstates, new_total_reward
            
            def no_step_fn():
                return obs, state, done, hstates, total_reward
            
            # Only step if not done  
            new_carry = jax.lax.cond(done["__all__"], no_step_fn, step_fn)
            return new_carry, None
        
        # Run episode using scan
        keys = jax.random.split(key, num_timesteps)
        
        init_carry = (obs, state, done, hstates, total_reward)
        final_carry, _ = jax.lax.scan(timestep, init_carry, keys)
        
        final_total_reward = final_carry[4]
        
        return final_total_reward

    # Run evaluation over multiple episodes
    @partial(jax.jit, static_argnums=(0, 1, 2, 3, 4))
    def evaluate_episodes(env, policy0, policy1, num_timesteps, num_episodes, key):
        """Run multiple episodes and return all rewards."""
        
        # Generate keys for all episodes
        episode_keys = jax.random.split(key, num_episodes)
        
        def run_single_episode(_, episode_key):            
            episode_reward = rollout_episode(env, policy0, policy1, num_timesteps, episode_key)
            return None, episode_reward
        
        _, all_ep_rewards = jax.lax.scan(run_single_episode, None, episode_keys)
        return all_ep_rewards

    print(f"Running evaluation over {num_episodes} episodes...")
    
    # Run all episodes
    all_ep_rewards = evaluate_episodes(env, policy0, policy1, num_timesteps, num_episodes, key)
    
    # Print results
    mean_reward = jnp.mean(all_ep_rewards)
    std_reward = jnp.std(all_ep_rewards)
    print(f"Results for {layout_name}, reward shaping={do_reward_shaping}:")
    print(f"Mean reward: {mean_reward:.3f} ± {std_reward:.3f}")    
    return mean_reward, std_reward, all_ep_rewards

if __name__ == "__main__":
    for layout_name in LAYOUT_TO_BASE_DIR.keys():
        test_bc_policy(layout_name, 
                       num_episodes=64, 
                       test_mode=True,
                       do_reward_shaping=True
                       )