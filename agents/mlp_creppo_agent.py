import flax
import jax
import jax.numpy as jnp
from typing import Any

import chex
import distrax
from flax.training.train_state import TrainState

from functools import partial

from agents.agent_interface import AgentPolicy
from agents.mlp_creppo import hl_gauss, QNetwork

@chex.dataclass(frozen=True)
class Transition:
    obs: chex.Array
    action: chex.Array
    action_logp: chex.Array
    reward: chex.Array
    done: chex.Array
    avail_actions: chex.Array
    next_obs: chex.Array
    next_avail_actions: chex.Array
    next_val: chex.Array = None
    soft_reward: chex.Array = None
    info: dict = None

class CustomTrainState(TrainState):
    batch_stats: Any

class CReppoTrainState(flax.struct.PyTreeNode):
    q_network_train_state: TrainState
    target_train_state: TrainState
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


class MLPCREPPOPolicy(AgentPolicy):
    """Policy wrapper for MLP Actor-Critic"""

    def __init__(self, action_dim, obs_dim,
                 norm_type, norm_input, num_bins, v_min, v_max,
                 init_alpha=0.01, hidden_size=128, num_layers=2):
        """
        Args:
            action_dim: int, dimension of the action space
            obs_dim: int, dimension of the observation space
        """
        super().__init__(action_dim, obs_dim)

        self.q_network = QNetwork(action_dim=action_dim,
                                  num_bins=num_bins,
                                  v_min=v_min,
                                  v_max=v_max,
                                  norm_type=norm_type,
                                  norm_input=norm_input,
                                  init_alpha=init_alpha,
                                  hidden_size=hidden_size,
                                  num_layers=num_layers)


    @partial(jax.jit, static_argnums=(0,))
    def get_action(self, params, obs, done, avail_actions, hstate, rng,
                   aux_obs=None, env_state=None, test_mode=False):
        """Get actions for the MLP policy."""
        params={
            "params": params[0],
            "batch_stats": params[1],
        }

        pi_logits = self.q_network.apply(params, (obs, avail_actions), train=False)["q_values"]

        # Can be used to control exploration, if wanted
        sample_pi = distrax.Categorical(logits=pi_logits / jnp.exp(params["params"]["log_alpha"]))

        action = jax.lax.cond(test_mode,
                              lambda: jnp.argmax(pi_logits, axis=-1), # Greedy action, test mode
                              lambda: sample_pi.sample(seed=rng))     # Sample action, training

        return action, None  # no hidden state

    @partial(jax.jit, static_argnums=(0,))
    def get_action_value_policy(self, params, obs, done, avail_actions, hstate, rng,
                   aux_obs=None, env_state=None, test_mode=False):
        """Get actions for the MLP policy."""
        params={
            "params": params[0],
            "batch_stats": params[1],
        }

        pi_logits = self.q_network.apply(params, (obs, avail_actions), train=False)["q_values"]

        # Can be used to control exploration, if wanted
        sample_pi = distrax.Categorical(logits=pi_logits / jnp.exp(params["params"]["log_alpha"]))

        action = jax.lax.cond(test_mode,
                              lambda: jnp.argmax(pi_logits, axis=-1), # Greedy action, test mode
                              lambda: sample_pi.sample(seed=rng))     # Sample action, training

        return action, pi_logits, sample_pi, None  # no hidden state

    @partial(jax.jit, static_argnums=(0,))
    def get_critic_out(self, params, obs, done, avail_actions, hstate, rng,
                                aux_obs=None, env_state=None):
        """Get actions, values, and policy for the MLP policy."""
        params={
            "params": params[0],
            "batch_stats": params[1],
        }

        out = self.q_network.apply(params, (obs, avail_actions), train=False)

        return out, None  # no hidden state

    def init_params(self, rng):
        """Initialize parameters for the MLP policy."""
        dummy_obs = jnp.zeros((self.obs_dim,))
        dummy_avail = jnp.ones((self.action_dim,))
        init_x = (dummy_obs, dummy_avail)

        q_network_params = self.q_network.init(rng, init_x, train=False)

        return q_network_params
