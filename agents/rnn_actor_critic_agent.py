from functools import partial

import jax
import jax.numpy as jnp

from agents.agent_interface import AgentPolicy
from agents.rnn_actor_critic import RNNActorCritic, ScannedRNN


class RNNActorCriticPolicy(AgentPolicy):
    """Policy wrapper for RNN Actor-Critic"""

    def __init__(self, action_dim, obs_dim,
                 activation="tanh", fc_hidden_dim=64, gru_hidden_dim=64):
        """
        Args:
            action_dim: int, dimension of the action space
            obs_dim: int, dimension of the observation space
            activation: str, activation function to use
            fc_hidden_dim: int, dimension of the feed-forward hidden layers
            gru_hidden_dim: int, dimension of the GRU hidden state
        """
        super().__init__(action_dim, obs_dim)
        self.network = RNNActorCritic(
            action_dim,
            fc_hidden_dim=fc_hidden_dim,
            gru_hidden_dim=gru_hidden_dim,
            activation=activation
        )
        self.gru_hidden_dim = gru_hidden_dim

    @partial(jax.jit, static_argnums=(0,))
    def get_action(self, params, obs, done, avail_actions, hstate, rng,
                   aux_obs=None, env_state=None, test_mode=False):
        """Get actions for the RNN policy.
        Shape of obs, done, avail_actions should correspond to (seq_len, batch_size, ...)
        Shape of hstate should correspond to (1, batch_size, -1). We maintain the extra first dimension for
        compatibility with the learning codes.
        """
        batch_size = obs.shape[1]
        new_hstate, pi, _ = self.network.apply(params, hstate.squeeze(0), (obs, done, avail_actions))
        action = jax.lax.cond(test_mode,
                              lambda: pi.mode(),
                              lambda: pi.sample(seed=rng))
        return action, new_hstate.reshape(1, batch_size, -1)

    @partial(jax.jit, static_argnums=(0,))
    def get_action_value_policy(self, params, obs, done, avail_actions, hstate, rng,
                                aux_obs=None, env_state=None):
        """Get actions, values, and policy for the RNN policy.
        Shape of obs, done, avail_actions should correspond to (seq_len, batch_size, ...)
        Shape of hstate should correspond to (1, batch_size, -1). We maintain the extra first dimension for
        compatibility with the learning codes.
        """
        batch_size = obs.shape[1]
        new_hstate, pi, val = self.network.apply(params, hstate.squeeze(0), (obs, done, avail_actions))
        action = pi.sample(seed=rng)
        return action, val, pi, new_hstate.reshape(1, batch_size, -1)

    def init_hstate(self, batch_size, aux_info=None):
        """Initialize hidden state for the RNN policy."""
        hstate =  ScannedRNN.initialize_carry(batch_size, self.gru_hidden_dim)
        hstate = hstate.reshape(1, batch_size, self.gru_hidden_dim)
        return hstate

    def init_params(self, rng):
        """Initialize parameters for the RNN policy."""
        batch_size = 1
        # Initialize hidden state
        init_hstate = self.init_hstate(batch_size)

        # Create dummy inputs - add time dimension
        dummy_obs = jnp.zeros((1, batch_size, self.obs_dim))
        dummy_done = jnp.zeros((1, batch_size))
        dummy_avail = jnp.ones((1, batch_size, self.action_dim))
        dummy_x = (dummy_obs, dummy_done, dummy_avail)

        # Initialize model
        return self.network.init(rng, init_hstate.reshape(batch_size, -1), dummy_x)
