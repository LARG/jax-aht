from functools import partial

import jax
import jax.numpy as jnp

from agents.agent_interface import AgentPolicy
from agents.mlp_ma_actor_critic import Actor, Critic


class MLPMAActorCriticPolicy(AgentPolicy):
    """Policy wrapper for MLP Actor-Critic"""

    def __init__(self, action_dim, obs_dim, state_dim, activation="relu"):
        """
        Args:
            action_dim: int, dimension of the action space
            obs_dim: int, dimension of the observation space
            state_dim: int, dimension of the state space
            activation: str, activation function to use
        """
        super().__init__(action_dim, obs_dim)
        self.state_dim = state_dim
        self.actor = Actor(action_dim, activation=activation)
        self.critic = Critic(activation=activation)

    @partial(jax.jit, static_argnums=(0,))
    def get_action(self, params, obs, done, avail_actions, hstate, rng,
                   aux_obs=None, env_state=None, test_mode=False):
        """Get actions for the MLP policy."""
        pi = self.actor.apply(params, (obs, avail_actions))
        action = jax.lax.cond(test_mode,
                              lambda: pi.mode(),
                              lambda: pi.sample(seed=rng))
        return action, None  # no hidden state

    @partial(jax.jit, static_argnums=(0,))
    def get_action_policy(self, params, obs, done, avail_actions, hstate, rng,
                   aux_obs=None, env_state=None, test_mode=False):
        """Get actions for the MLP policy."""
        pi = self.actor.apply(params, (obs, avail_actions))
        action = jax.lax.cond(test_mode,
                              lambda: pi.mode(),
                              lambda: pi.sample(seed=rng))
        return action, pi, None  # no hidden state

    @partial(jax.jit, static_argnums=(0,))
    def get_value(self, params, state, done, hstate, rng,
                  aux_obs=None, env_state=None):
        """Get actions, values, and policy for the MLP policy."""
        val = self.critic.apply(params, state)

        return val, None  # no hidden state

    @partial(jax.jit, static_argnums=(0,))
    def get_action_value_policy(self, params, state, obs, done, avail_actions, hstate, rng,
                                aux_obs=None, env_state=None):
        """Get actions, values, and policy for the MLP policy."""
        pi = self.actor.apply(params[0], (obs, avail_actions))
        action = pi.sample(seed=rng)

        val = self.critic.apply(params[1], state)

        return action, val, pi, (None, None)  # no hidden state

    def init_critic_hstate(self, batch_size, aux_info: dict=None):
        """Initialize the hidden state for the policy.
        Args:
            batch_size: int, the batch size of the hidden state
            aux_info: any auxiliary information needed to initialize the hidden state at the
            start of an episode (e.g. the agent id).
        Returns:
            chex.Array: the initialized hidden state
        """
        return None

    def init_params(self, rng):
        """Initialize parameters for the MLP policy."""
        dummy_obs = jnp.zeros((self.obs_dim,))
        dummy_avail = jnp.ones((self.action_dim,))
        init_x = (dummy_obs, dummy_avail)
        actor_params = self.actor.init(rng, init_x)

        dummy_state = jnp.zeros((self.state_dim,))
        critic_params = self.critic.init(rng, dummy_state)

        return {"actor": actor_params, "critic": critic_params}
