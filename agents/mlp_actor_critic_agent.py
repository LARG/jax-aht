from functools import partial

import jax
import jax.numpy as jnp

from agents.agent_interface import AgentPolicy
from agents.mlp_actor_critic import ActorCritic
from agents.mlp_actor_critic import ActorWithDoubleCritic
from agents.mlp_actor_critic import ActorWithConditionalCritic


class MLPActorCriticPolicy(AgentPolicy):
    """Policy wrapper for MLP Actor-Critic"""

    def __init__(self, action_dim, obs_dim, activation="tanh"):
        """
        Args:
            action_dim: int, dimension of the action space
            obs_dim: int, dimension of the observation space
            activation: str, activation function to use
        """
        super().__init__(action_dim, obs_dim)
        self.network = ActorCritic(action_dim, activation=activation)

    @partial(jax.jit, static_argnums=(0,))
    def get_action(self, params, obs, done, avail_actions, hstate, rng,
                   aux_obs=None, env_state=None, test_mode=False):
        """Get actions for the MLP policy."""
        pi, _ = self.network.apply(params, (obs, avail_actions))
        action = jax.lax.cond(test_mode,
                              lambda: pi.mode(),
                              lambda: pi.sample(seed=rng))
        return action, None  # no hidden state

    @partial(jax.jit, static_argnums=(0,))
    def get_action_value_policy(self, params, obs, done, avail_actions, hstate, rng,
                                aux_obs=None, env_state=None):
        """Get actions, values, and policy for the MLP policy."""
        pi, val = self.network.apply(params, (obs, avail_actions))
        action = pi.sample(seed=rng)
        return action, val, pi, None  # no hidden state

    def init_params(self, rng):
        """Initialize parameters for the MLP policy."""
        dummy_obs = jnp.zeros((self.obs_dim,))
        dummy_avail = jnp.ones((self.action_dim,))
        init_x = (dummy_obs, dummy_avail)
        return self.network.init(rng, init_x)

class ActorWithDoubleCriticPolicy(AgentPolicy):
    """Policy wrapper for Actor with Double Critics"""

    def __init__(self, action_dim, obs_dim, activation="tanh"):
        """
        Args:
            action_dim: int, dimension of the action space
            obs_dim: int, dimension of the observation space
            activation: str, activation function to use
        """
        super().__init__(action_dim, obs_dim)
        self.network = ActorWithDoubleCritic(action_dim, activation=activation)

    @partial(jax.jit, static_argnums=(0,))
    def get_action(self, params, obs, done, avail_actions, hstate, rng,
                   aux_obs=None, env_state=None, test_mode=False):
        """Get actions for the policy with double critics.
        """
        pi, _, _ = self.network.apply(params, (obs, avail_actions))
        action = jax.lax.cond(test_mode,
                              lambda: pi.mode(),
                              lambda: pi.sample(seed=rng))
        return action, None  # no hidden state

    @partial(jax.jit, static_argnums=(0,))
    def get_action_value_policy(self, params, obs, done, avail_actions, hstate, rng,
                                aux_obs=None, env_state=None):
        """Get actions, values, and policy for the policy with double critics."""
        # convention: val1 is value of of ego agent, val2 is value of best response agent
        pi, val1, val2 = self.network.apply(params, (obs, avail_actions))
        action = pi.sample(seed=rng)
        return action, (val1, val2), pi, None # no hidden state

    def init_params(self, rng):
        """Initialize parameters for the policy with double critics."""
        dummy_obs = jnp.zeros((self.obs_dim,))
        dummy_avail = jnp.ones((self.action_dim,))
        init_x = (dummy_obs, dummy_avail)
        return self.network.init(rng, init_x)

class PseudoActorWithDoubleCriticPolicy(ActorWithDoubleCriticPolicy):
    """Enables ActorWithDoubleCritic to masquerade as an actor with a single critic."""
    def __init__(self, action_dim, obs_dim, activation="tanh"):
        super().__init__(action_dim, obs_dim, activation)

    def get_action_value_policy(self, params, obs, done, avail_actions, hstate, rng,
                                aux_obs=None, env_state=None):
        action, (val1, _), pi, hidden_state = super().get_action_value_policy(
            params, obs, done, avail_actions, hstate, rng,
            aux_obs, env_state)
        return action, val1, pi, hidden_state

class ActorWithConditionalCriticPolicy(AgentPolicy):
    """Policy wrapper for ActorWithConditionalCritic
    """
    def __init__(self, action_dim, obs_dim, pop_size, activation="tanh"):
        """
        Args:
            action_dim: int, dimension of the action space
            obs_dim: int, dimension of the observation space
            pop_size: int, number of agents in the population that the critic was trained with
            activation: str, activation function to use
        """
        super().__init__(action_dim, obs_dim)
        self.pop_size = pop_size
        self.network = ActorWithConditionalCritic(action_dim, activation=activation)

    @partial(jax.jit, static_argnums=(0,))
    def get_action(self, params, obs, done, avail_actions, hstate, rng,
                   aux_obs=None, env_state=None, test_mode=False):
        """Get actions."""
        # The agent id is only used by the critic, so we pass in a
        # dummy vector to represent the one-hot agent id
        dummy_agent_id = jnp.zeros(obs.shape[:-1] + (self.pop_size,))
        pi, _ = self.network.apply(params, (obs, dummy_agent_id, avail_actions))
        action = jax.lax.cond(test_mode,
                              lambda: pi.mode(),
                              lambda: pi.sample(seed=rng))
        return action, None  # no hidden state

    @partial(jax.jit, static_argnums=(0,))
    def get_action_value_policy(self, params, obs, done, avail_actions, hstate, rng,
                                aux_obs=None, env_state=None):
        """Get actions, values, and policy for the policy with conditional critics.
        The auxiliary observation should be used to pass in the agent ids that we wish to predict
        values for.
        """
        pi, value = self.network.apply(params, (obs, aux_obs, avail_actions))
        action = pi.sample(seed=rng)
        return action, value, pi, None # no hidden state

    def init_params(self, rng):
        """Initialize parameters for the policy with conditional critics."""
        dummy_obs = jnp.zeros((self.obs_dim,))
        dummy_ids = jnp.zeros((self.pop_size,))
        dummy_avail = jnp.ones((self.action_dim,))
        init_x = (dummy_obs, dummy_ids, dummy_avail)
        return self.network.init(rng, init_x)

class PseudoActorWithConditionalCriticPolicy(ActorWithConditionalCriticPolicy):
    """Enables PseudoActorWithConditionalCriticPolicy to act as an MLPActorCriticPolicy.
    by passing in a dummy agent id.
    """
    def __init__(self, action_dim, obs_dim, pop_size, activation="tanh"):
        super().__init__(action_dim, obs_dim, pop_size, activation)

    def get_action_value_policy(self, params, obs, done, avail_actions, hstate, rng,
                                aux_obs=None, env_state=None):
        dummy_agent_id = jnp.zeros(obs.shape[:-1] + (self.pop_size,))
        action, val, pi, hidden_state = super().get_action_value_policy(
            params, obs, done, avail_actions, hstate, rng,
            dummy_agent_id, env_state)
        return action, val, pi, hidden_state
