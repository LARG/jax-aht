from functools import partial

import jax
import jax.numpy as jnp

from agents.agent_interface import AgentPolicy
from agents.rnn_ma_actor_critic import RNNActor, RNNCritic, ScannedRNN


class RNNMAActorCriticPolicy(AgentPolicy):
    """Policy wrapper for RNN Actor-Critic"""

    def __init__(self, action_dim, obs_dim, state_dim,
                 activation="relu", fc_hidden_dim=64, gru_hidden_dim=64):
        """
        Args:
            action_dim: int, dimension of the action space
            obs_dim: int, dimension of the observation space
            state_dim: int, dimension of the state space
            activation: str, activation function to use
            fc_hidden_dim: int, dimension of the feed-forward hidden layers
            gru_hidden_dim: int, dimension of the GRU hidden state
        """
        super().__init__(action_dim, obs_dim)
        self.gru_hidden_dim = gru_hidden_dim
        self.state_dim = state_dim
        self.actor = RNNActor(
            action_dim,
            fc_hidden_dim=fc_hidden_dim,
            gru_hidden_dim=gru_hidden_dim,
            activation=activation
        )
        self.critic = RNNCritic(
            fc_hidden_dim=fc_hidden_dim,
            gru_hidden_dim=gru_hidden_dim,
            activation=activation
        )

    @partial(jax.jit, static_argnums=(0,))
    def get_action(self, params, obs, done, avail_actions, hstate, rng,
                   aux_obs=None, env_state=None, test_mode=False):
        """Get actions for the RNN policy.
        Shape of obs, done, avail_actions should correspond to (seq_len, batch_size, ...)
        Shape of hstate should correspond to (1, batch_size, -1). We maintain the extra first dimension for
        compatibility with the learning codes.
        """
        batch_size = obs.shape[1]
        new_hstate, pi = self.actor.apply(params[0], hstate.squeeze(0), (obs, done, avail_actions))
        action = jax.lax.cond(test_mode,
                              lambda: pi.mode(),
                              lambda: pi.sample(seed=rng))
        return action, new_hstate.reshape(1, batch_size, -1)

    @partial(jax.jit, static_argnums=(0,))
    def get_action_policy(self, params, obs, done, avail_actions, hstate, rng,
                   aux_obs=None, env_state=None, test_mode=False):
        """Get actions for the RNN policy.
        Shape of obs, done, avail_actions should correspond to (seq_len, batch_size, ...)
        Shape of hstate should correspond to (1, batch_size, -1). We maintain the extra first dimension for
        compatibility with the learning codes.
        """
        batch_size = obs.shape[1]
        new_hstate, pi = self.actor.apply(params[0], hstate.squeeze(0), (obs, done, avail_actions))
        action = jax.lax.cond(test_mode,
                              lambda: pi.mode(),
                              lambda: pi.sample(seed=rng))
        return action, pi, new_hstate.reshape(1, batch_size, -1)

    @partial(jax.jit, static_argnums=(0,))
    def get_value(self, params, state, done, hstate, rng,
                  aux_obs=None, env_state=None):
        """Get actions, values, and policy for the MLP policy."""
        batch_size = state.shape[1]
        new_critic_hstate, val = self.critic.apply(params, hstate.squeeze(0), (state, done))

        return val, new_critic_hstate.reshape(1, batch_size, -1)

    @partial(jax.jit, static_argnums=(0,))
    def get_action_value_policy(self, params, state, obs, done, avail_actions, hstate, rng,
                                aux_obs=None, env_state=None):
        """Get actions, values, and policy for the RNN policy.
        Shape of obs, done, avail_actions should correspond to (seq_len, batch_size, ...)
        Shape of hstate should correspond to (1, batch_size, -1). We maintain the extra first dimension for
        compatibility with the learning codes.
        """
        batch_size = obs.shape[1]
        new_actor_hstate, pi = self.actor.apply(params[0], hstate[0].squeeze(0), (obs, done, avail_actions))
        action = pi.sample(seed=rng)

        new_critic_hstate, val = self.critic.apply(params[1], hstate[1].squeeze(0), (state, done))

        return action, val, pi, new_actor_hstate.reshape(1, batch_size, -1), new_critic_hstate.reshape(1, batch_size, -1)

    def init_hstate(self, batch_size, aux_info=None):
        """Initialize hidden state for the RNN policy."""
        actor_hstate =  ScannedRNN.initialize_carry(batch_size, self.gru_hidden_dim)
        actor_hstate = actor_hstate.reshape(1, batch_size, self.gru_hidden_dim)

        return actor_hstate

    def init_critic_hstate(self, batch_size, aux_info=None):
        """Initialize hidden state for the RNN policy."""
        critic_hstate =  ScannedRNN.initialize_carry(batch_size, self.gru_hidden_dim)
        critic_hstate = critic_hstate.reshape(1, batch_size, self.gru_hidden_dim)

        return critic_hstate

    def init_params(self, rng):
        """Initialize parameters for the RNN policy."""
        batch_size = 1
        # Initialize hidden state
        init_actor_hstate = self.init_hstate(batch_size)
        init_critic_hstate = self.init_critic_hstate(batch_size)

        # Create dummy inputs - add time dimension
        dummy_obs = jnp.zeros((1, batch_size, self.obs_dim))
        dummy_done = jnp.zeros((1, batch_size))
        dummy_avail = jnp.ones((1, batch_size, self.action_dim))
        dummy_x = (dummy_obs, dummy_done, dummy_avail)
        actor_params = self.actor.init(rng, init_actor_hstate.reshape(batch_size, -1), dummy_x)

        dummy_state = jnp.zeros((1, batch_size, self.state_dim))
        dummy_x = (dummy_state, dummy_done)
        critic_params = self.critic.init(rng, init_critic_hstate.reshape(batch_size, -1), dummy_x)

        return {"actor": actor_params, "critic": critic_params}
