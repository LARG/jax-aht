from typing import Dict, Any, Sequence

import flax.linen as nn
import jax.numpy as jnp

from flax.linen.initializers import constant, orthogonal

from agents.s5_actor_critic import StackedEncoderModel


class S5QNetwork(nn.Module):
    action_dim: Sequence[int]
    ssm_init_fn: Any
    ssm_hidden_dim: int = 512 # ssm_size
    s5_d_model: int = 512
    s5_n_layers: int = 2
    s5_activation: str = "full_glu"
    s5_do_norm: bool = True
    s5_prenorm: bool = True
    s5_do_gtrxl_norm: bool = True
    s5_no_reset: bool = False

    def setup(self):
        self.layer_0 = nn.Dense(self.ssm_hidden_dim, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))

        self.s5 = StackedEncoderModel(
            ssm=self.ssm_init_fn,
            d_model=self.s5_d_model,
            n_layers=self.s5_n_layers,
            activation=self.s5_activation,
            do_norm=self.s5_do_norm,
            prenorm=self.s5_prenorm,
            do_gtrxl_norm=self.s5_do_gtrxl_norm,
        )

        self.out_layer = nn.Dense(self.action_dim, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x

        if self.s5_no_reset:
            dones = jnp.zeros_like(dones)
        embedding = self.layer_0(obs)

        embedding = nn.leaky_relu(embedding)

        # hidden: (1, num_actors, ssm_hidden_dim / 2)
        # embedding: (1, num_actors, ssm_hidden_dim)
        # dones: (1, num_actors)
        hidden, embedding = self.s5(hidden, embedding, dones)

        q_vals = self.out_layer(embedding)

        return hidden, q_vals
