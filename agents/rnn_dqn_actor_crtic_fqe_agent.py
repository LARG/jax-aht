from functools import partial

import jax
import jax.numpy as jnp

from agents.agent_interface import AgentPolicy
from agents.rnn_q_network import RNNQNetwork
from agents.rnn_actor_critic import ScannedRNN


class RNNDQNActorCriticFQEPolicy(AgentPolicy):
    """Policy wrapper for DRQN Actor-Critic Fitted Q Estimation"""

    def __init__(self, action_dim, obs_dim, actor_critic_policy, epsilon_start=1.0, epsilon_finish=0.1, epsilon_anneal_time=10000,
                 gru_hidden_dim=64, init_scale=1.0):
        """
        Args:
            action_dim: int, dimension of the action space
            obs_dim: int, dimension of the observation space
            actor_critic_policy: nn.Module, the actor-critic network to use for value estimation
            epsilon_start: float, initial epsilon for epsilon-greedy exploration
            epsilon_finish: float, final epsilon for epsilon-greedy exploration
            epsilon_anneal_time: int, number of timesteps over which to anneal epsilon
            gru_hidden_dim: int, hidden dimension for the GRU in the DRQN
            init_scale: float, scale for orthogonal initialization of network weights
        """
        super().__init__(action_dim, obs_dim)
        self.network = RNNQNetwork(action_dim, gru_hidden_dim, init_scale)
        self.actor_critic = actor_critic_policy
        self.epsilon_start = epsilon_start
        self.epsilon_finish = epsilon_finish
        self.epsilon_anneal_time = epsilon_anneal_time

    def eps_greedy_exploration(self, rng, q_vals, t):
        """Epsilon-greedy exploration strategy."""
        # Keys for sampling random actions and picking actions
        rng_a, rng_e = jax.random.split(rng, 2)

        # Compute epsilon for the current timestep
        eps = jnp.clip(
            ((self.epsilon_finish - self.epsilon_start) / self.epsilon_anneal_time)
            * t + self.epsilon_start, self.epsilon_finish,
        )

        # Greedy argmax actions from the Q-values
        greedy_actions = jnp.argmax(q_vals, axis=-1)
        chosen_actions = jnp.where(
            # Pick the actions that should be random
            jax.random.uniform(rng_e, greedy_actions.shape) < eps,
            # Sample random actions uniformly from the action space
            jax.random.randint(rng_a, shape=greedy_actions.shape, minval=0, maxval=q_vals.shape[-1]),
            greedy_actions,
        )
        return chosen_actions


    def eps_greedy_actor_critic_exploration(self, rng, actions, avail_actions, t):
        """Epsilon-greedy exploration strategy."""
        # Keys for sampling random actions and picking actions
        rng_a, rng_e = jax.random.split(rng, 2)

        # Compute epsilon for the current timestep
        eps = jnp.clip(
            ((self.epsilon_finish - self.epsilon_start) / self.epsilon_anneal_time)
            * t + self.epsilon_start, self.epsilon_finish,
        )

        # Sample random actions only from available actions
        # Generate random scores for all actions, mask unavailable ones
        random_scores = jax.random.uniform(rng_a, shape=avail_actions.shape)
        masked_scores = jnp.where(avail_actions, random_scores, -jnp.inf)
        random_actions = jnp.argmax(masked_scores, axis=-1)

        chosen_actions = jnp.where(
            # Pick the actions that should be random
            jax.random.uniform(rng_e, actions.shape) < eps,
            # Use random available actions
            random_actions,
            actions,
        )
        return chosen_actions

    @partial(jax.jit, static_argnums=(0,))
    def get_action(self, params, obs, done, avail_actions, hstate, rng,
                   aux_obs=None, env_state=None, test_mode=False):
        """Get actions for the DRQN policy."""
        batch_size = obs.shape[1]
        new_hstate, qvals = self.network.apply(params, hstate.squeeze(0), (obs, done))

        # Mask out unavailable actions by setting their Q-values to -inf
        masked_qvals = jnp.where(avail_actions, qvals, -jnp.inf)

        actions = jax.lax.cond(test_mode,
                               lambda: jnp.argmax(masked_qvals, axis=-1),
                               lambda: self.eps_greedy_exploration(rng, masked_qvals, env_state.env_state.env_state.timestep))
        return actions, new_hstate.reshape(1, batch_size, -1)

    @partial(jax.jit, static_argnums=(0,))
    def get_actor_critic_action(self, params, obs, done, avail_actions, hstate, rng,
                   aux_obs=None, env_state=None, test_mode=False):
        """Get actions for the DRQN policy."""
        actions, new_ac_hstate = self.actor_critic.get_action(params, obs, done, avail_actions, hstate, rng,
                                                              aux_obs, env_state, test_mode=False)
        actions = jax.lax.stop_gradient(actions)
        new_ac_hstate = jax.lax.stop_gradient(new_ac_hstate)

        actions = jax.lax.cond(test_mode,
                               lambda: actions,
                               lambda: self.eps_greedy_actor_critic_exploration(rng, actions,
                                                                                avail_actions,
                                                                                env_state.env_state.env_state.timestep))

        return actions, new_ac_hstate

    @partial(jax.jit, static_argnums=(0,))
    def get_action_value_policy(self, params, obs, done, avail_actions, hstate, rng,
                                aux_obs=None, env_state=None):
        """Get actions, values, and policy for the DRQN policy."""
        batch_size = obs.shape[1]
        new_hstate, qvals = self.network.apply(params, hstate.squeeze(0), (obs, done))

        # Mask out unavailable actions by setting their Q-values to -inf
        masked_qvals = jnp.where(avail_actions, qvals, -jnp.inf)

        # Greedy argmax actions from the Q-values
        actions = jnp.argmax(masked_qvals, axis=-1)

        return actions, qvals, None, new_hstate.reshape(1, batch_size, -1) # no policy

    def init_hstate(self, batch_size, aux_info=None):
        """Initialize hidden state for the DRQN policy."""
        hstate =  ScannedRNN.initialize_carry(batch_size, self.gru_hidden_dim)
        hstate = hstate.reshape(1, batch_size, self.gru_hidden_dim)
        # actor_critic_hstate = self.actor_critic.init_hstate(batch_size, aux_info)
        return hstate #, actor_critic_hstate

    def init_params(self, rng):
        """Initialize parameters for the DRQN policy."""
        batch_size = 1
        # Initialize hidden state
        init_hstate = self.init_hstate(batch_size)

        # Create dummy inputs - add time dimension
        dummy_obs = jnp.zeros((1, batch_size, self.obs_dim))
        dummy_done = jnp.zeros((1, batch_size))
        dummy_x = (dummy_obs, dummy_done)

        # Initialize model
        return self.network.init(rng, init_hstate.reshape(batch_size, -1), dummy_x)
