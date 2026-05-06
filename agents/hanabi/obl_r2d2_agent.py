"""OBL R2D2 agent (Hu et al. 2021). Flax reimplementation that loads
pretrained weights from mttga/obl-r2d2-flax. Private MLP * public LSTM,
greedy over advantage values.
"""

import jax
from jax import numpy as jnp
import flax.linen as nn
from functools import partial
from typing import Tuple
from chex import Array, PRNGKey
from flax.linen.module import compact, nowrap


class MultiLayerLSTM(nn.RNNCellBase):
    """Stacked LSTM matching the OBL PyTorch implementation."""

    num_layers: int
    features: int

    @compact
    def __call__(self, carry, inputs):
        new_hs = []
        new_cs = []
        for layer in range(self.num_layers):
            new_carry, y = nn.LSTMCell(self.features, name=f"l{layer}")(
                jax.tree.map(lambda x: x[layer], carry), inputs
            )
            new_cs.append(new_carry[0])
            new_hs.append(new_carry[1])
            inputs = y

        new_final_carry = (jnp.stack(new_cs), jnp.stack(new_hs))
        return new_final_carry, y

    @nowrap
    def initialize_carry(
        self, rng: PRNGKey, batch_dims: Tuple[int, ...]
    ) -> Tuple[Array, Array]:
        mem_shape = (self.num_layers,) + batch_dims + (self.features,)
        c = jnp.zeros(mem_shape)
        h = jnp.zeros(mem_shape)
        return (c, h)

    @property
    def num_feature_axes(self) -> int:
        return 1


class OBLAgentR2D2(nn.Module):
    """Private MLP * public LSTM, linear head to 21 advantage values."""

    hid_dim: int = 512
    out_dim: int = 21
    num_lstm_layer: int = 2
    # public obs starts at 125 for standard 2p Hanabi
    pub_obs_start: int = 125

    @compact
    def __call__(self, carry, inputs):
        priv_s, publ_s = inputs

        priv_o = nn.Sequential([
            nn.Dense(self.hid_dim, name="priv_net_dense_0"), nn.relu,
            nn.Dense(self.hid_dim, name="priv_net_dense_1"), nn.relu,
            nn.Dense(self.hid_dim, name="priv_net_dense_2"), nn.relu,
        ])(priv_s)

        x = nn.Sequential([
            nn.Dense(self.hid_dim, name="publ_net_dense_0"), nn.relu,
        ])(publ_s)
        carry, publ_o = MultiLayerLSTM(
            num_layers=self.num_lstm_layer, features=self.hid_dim, name="lstm"
        )(carry, x)

        o = priv_o * publ_o
        a = nn.Dense(self.out_dim, name="fc_a")(o)

        return carry, a

    @partial(jax.jit, static_argnums=[0])
    def greedy_act(self, params, carry, inputs):
        """Greedy action from masked advantages. Full 2p Hanabi only (658-dim obs)."""
        obs, legal_move = inputs
        priv_s = obs
        publ_s = obs[..., self.pub_obs_start:]

        carry, adv = self.apply(params, carry, (priv_s, publ_s))

        legal_adv = (1 + adv - adv.min()) * legal_move
        greedy_action = jnp.argmax(legal_adv, axis=-1)

        return carry, greedy_action

    @nowrap
    def initialize_carry(
        self, rng: PRNGKey, batch_dims: Tuple[int, ...]
    ) -> Tuple[Array, Array]:
        return MultiLayerLSTM(
            num_layers=self.num_lstm_layer, features=self.hid_dim
        ).initialize_carry(rng, batch_dims)
