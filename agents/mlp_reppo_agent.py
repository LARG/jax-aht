import flax
import jax
import jax.numpy as jnp
from typing import Any

import chex
import distrax
from flax.training.train_state import TrainState

from functools import partial

from agents.agent_interface import AgentPolicy
from agents.mlp_reppo import hl_gauss, QNetwork, Actor

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
    importance_weight: chex.Array = None
    info: dict = None

class CustomTrainState(TrainState):
    batch_stats: Any

class ReppoTrainState(flax.struct.PyTreeNode):
    actor_train_state: TrainState
    q_network_train_state: TrainState
    target_actor_train_state: TrainState
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


class MLPREPPOPolicy(AgentPolicy):
    """Policy wrapper for MLP Actor-Critic"""

    def __init__(self, action_dim, obs_dim,
                 norm_type, norm_input, num_bins, v_min, v_max,
                 init_alpha, init_lagrangian,
                 min_is_weight, max_is_weight):
        """
        Args:
            action_dim: int, dimension of the action space
            obs_dim: int, dimension of the observation space
            activation: str, activation function to use
        """
        super().__init__(action_dim, obs_dim)
        self.min_is_weight = min_is_weight
        self.max_is_weight = max_is_weight

        self.q_network = QNetwork(action_dim=action_dim,
                                  norm_type=norm_type,
                                  norm_input=norm_input,
                                  num_bins=num_bins,
                                  v_min=v_min,
                                  v_max=v_max)
        self.actor = Actor(action_dim,
                           norm_type=norm_type,
                           norm_input=norm_input,
                           init_alpha=init_alpha,
                           init_lagrangian=init_lagrangian)


    @partial(jax.jit, static_argnums=(0,))
    def get_action(self, params, obs, done, avail_actions, hstate, rng,
                   aux_obs=None, env_state=None, test_mode=False,
                   temp_schedule=1.0):
        """Get actions for the MLP policy."""
        params={
            "params": params[0],
            "batch_stats": params[1],
        }

        pi = self.actor.apply(params, (obs, avail_actions), train=False)

        # Can be used to control exploration, if wanted
        sample_pi = distrax.Categorical(logits=pi.logits / temp_schedule)

        action = jax.lax.cond(test_mode,
                              lambda: jnp.argmax(pi.logits, axis=-1), # Greedy action, test mode
                              lambda: sample_pi.sample(seed=rng))     # Sample action, training

        return action, None  # no hidden state

    @partial(jax.jit, static_argnums=(0,))
    def get_action_importance_policy(self, params, obs, done, avail_actions, hstate, rng,
                   aux_obs=None, env_state=None, test_mode=False,
                   temp_schedule=1.0):
        """Get actions for the MLP policy."""
        params={
            "params": params[0],
            "batch_stats": params[1],
        }

        pi = self.actor.apply(params, (obs, avail_actions), train=False)

        # Can be used to control exploration, if wanted
        sample_pi = distrax.Categorical(logits=pi.logits / temp_schedule)

        actions = jax.lax.cond(test_mode,
                               lambda: jnp.argmax(pi.logits, axis=-1), # Greedy action, test mode
                               lambda: sample_pi.sample(seed=rng))     # Sample action, training

        importance_weight = jnp.clip(
            jnp.nan_to_num(
                pi.log_prob(actions) - sample_pi.log_prob(actions),
                nan=jnp.log(self.min_is_weight),  # If log_prob is NaN, set importance weight to min_is_weight
            ),
            a_min=jnp.log(self.min_is_weight),
            a_max=jnp.log(self.max_is_weight),
        )

        return actions, importance_weight, pi, None  # no hidden state

    @partial(jax.jit, static_argnums=(0,))
    def get_critic_logits_probs_values(self, params, obs, done, avail_actions, hstate, rng,
                                       aux_obs=None, env_state=None):
        """Get logits, probs, and valies for the MLP critic policy."""
        params={
            "params": params[0],
            "batch_stats": params[1],
        }

        critic_outs = self.q_network.apply(params, obs, train=False)

        return critic_outs["logits"], critic_outs["probs"], critic_outs["q_values"], None  # no hidden state

    # @partial(jax.jit, static_argnums=(0,))
    # def get_action_value_policy(self, params, obs, done, avail_actions, hstate, rng,
    #                             aux_obs=None, env_state=None):
    #     """Get actions, values, and policy for the MLP policy."""
    #     pi, val = self.network.apply(params, (obs, avail_actions))
    #     action = pi.sample(seed=rng)
    #     return action, val, pi, None  # no hidden state

    def init_hstate(self, batch_size, aux_info: dict=None) -> chex.Array:
        """Initialize the hidden state for the policy.
        Args:
            batch_size: int, the batch size of the hidden state
            aux_info: any auxiliary information needed to initialize the hidden state at the
            start of an episode (e.g. the agent id).
        Returns:
            chex.Array: the initialized hidden state
        """
        return None, None, None

    def init_params(self, rng):
        """Initialize parameters for the MLP policy."""
        rng_q_network, rng_actor  = jax.random.split(rng, 2)
        dummy_obs = jnp.zeros((self.obs_dim,))
        dummy_avail = jnp.ones((self.action_dim,))
        init_x = (dummy_obs, dummy_avail)

        q_network_params = self.q_network.init(rng_q_network, dummy_obs, train=False)
        actor_params = self.actor.init(rng_actor, init_x, train=False)

        return q_network_params, actor_params
