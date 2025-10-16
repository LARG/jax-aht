import abc
from typing import Tuple, Dict
import chex
from functools import partial
import jax
import jax.numpy as jnp


class AgentPolicy(abc.ABC):
    '''Abstract base class for a policy.'''

    def __init__(self, action_dim, obs_dim):
        '''
        Args:
            action_dim: int, dimension of the action space
            obs_dim: int, dimension of the observation space
        '''
        self.action_dim = action_dim
        self.obs_dim = obs_dim

    @abc.abstractmethod
    @partial(jax.jit, static_argnums=(0,))
    def get_action(self, params, obs, done, avail_actions, hstate, rng,
                   aux_obs=None, env_state=None, test_mode=False) -> Tuple[int, chex.Array]:
        """
        Only computes an action given an observation, done flag, available actions, hidden state, and random key.

        Args:
            params (dict): The parameters of the policy.
            obs (chex.Array): The observation.
            done (chex.Array): The done flag.
            avail_actions (chex.Array): The available actions.
            hstate (chex.Array): The hidden state.
            key (jax.random.PRNGKey): The random key.
            env_state (chex.Array): The environment state.
            aux_obs (chex.Array): an optional auxiliary vector to append to the observation
        Returns:
            Tuple[int, chex.Array]: A tuple containing the action and the new hidden state.
        """
        pass

    @partial(jax.jit, static_argnums=(0,))
    def get_action_value_policy(self, params, obs, done, avail_actions, hstate, rng,
                                aux_obs=None, env_state=None) -> Tuple[int, chex.Array, chex.Array, chex.Array]:
        """
        Computes the action, value, and policy given an observation,
        done flag, available actions, hidden state, and random key.

        Args:
            params (dict): The parameters of the policy.
            obs (chex.Array): The observation.
            done (chex.Array): The done flag.
            avail_actions (chex.Array): The available actions.
            hstate (chex.Array): The hidden state.
            key (jax.random.PRNGKey): The random key.
            aux_obs (chex.Array): an optional auxiliary vector to append to the observation
        Returns:
            Tuple[int, chex.Array, chex.Array, chex.Array]:
                A tuple containing the action, value, policy, and new hidden state.
        """
        pass

    def init_hstate(self, batch_size, aux_info: dict=None) -> chex.Array:
        """Initialize the hidden state for the policy.
        Args:
            batch_size: int, the batch size of the hidden state
            aux_info: any auxiliary information needed to initialize the hidden state at the
            start of an episode (e.g. the agent id).
        Returns:
            chex.Array: the initialized hidden state
        """
        return None

    def init_params(self, rng) -> Dict:
        """Initialize the parameters for the policy."""
        return None
