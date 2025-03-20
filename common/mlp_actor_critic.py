import functools
import numpy as np
from typing import Sequence

import distrax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import jax
import jax.numpy as jnp

class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"
    fc_hidden_dim: int = 64

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        obs, avail_actions = x
        actor_mean = nn.Dense(
            self.fc_hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.fc_hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        
        # Mask unavailable actions if avail_actions is provided
        unavail_actions = 1 - avail_actions
        actor_mean = actor_mean - (unavail_actions * 1e10)
            
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.fc_hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        critic = activation(critic)
        critic = nn.Dense(
            self.fc_hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)
