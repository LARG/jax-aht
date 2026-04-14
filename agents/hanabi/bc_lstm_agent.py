"""BC-LSTM agent that imitates human Hanabi play. Trained on AH2AC2 data.

Manual forward pass because AH2AC2 splits LSTM gates into separate
input/hidden projections (not Flax's combined kernel).
"""
import os
from functools import partial

import jax
import jax.numpy as jnp
from safetensors.numpy import load_file as load_safetensors


def load_bc_params(weight_path):
    """Load BC-LSTM params from safetensors. Returns flat dict of JAX arrays."""
    from common.save_load_utils import REPO_PATH

    if not os.path.isabs(weight_path):
        weight_path = os.path.join(REPO_PATH, weight_path)

    if not os.path.exists(weight_path):
        raise FileNotFoundError(
            f"BC-LSTM weights not found at {weight_path}. "
            "Train with AH2AC2's bc.py to regenerate."
        )

    params = load_safetensors(weight_path)
    return {k: jnp.array(v) for k, v in params.items()}


def _layer_norm(x, scale, bias, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return scale * (x - mean) / jnp.sqrt(var + eps) + bias


def _lstm_cell(carry, x, params):
    # AH2AC2 uses separate input/hidden projections per gate, no bias on input side
    c, h = carry
    p = 'ScanLstmCellWithHiddenStateReset_0'

    i = jax.nn.sigmoid(x @ params[f'{p},ii,kernel'] + h @ params[f'{p},hi,kernel'] + params[f'{p},hi,bias'])
    f = jax.nn.sigmoid(x @ params[f'{p},if,kernel'] + h @ params[f'{p},hf,kernel'] + params[f'{p},hf,bias'])
    g = jnp.tanh(x @ params[f'{p},ig,kernel'] + h @ params[f'{p},hg,kernel'] + params[f'{p},hg,bias'])
    o = jax.nn.sigmoid(x @ params[f'{p},io,kernel'] + h @ params[f'{p},ho,kernel'] + params[f'{p},ho,bias'])

    c_new = f * c + i * g
    h_new = o * jnp.tanh(c_new)
    return (c_new, h_new), h_new


def bc_lstm_forward(params, carry, obs, legal_actions=None):
    """Forward pass: Dense(1024) -> GELU -> LN -> LSTM(512) -> Dense(256) -> GELU -> Dense(21)."""
    # preprocess
    x = obs @ params['Dense_0,kernel'] + params['Dense_0,bias']
    x = jax.nn.gelu(x)
    x = _layer_norm(x, params['LayerNorm_0,scale'], params['LayerNorm_0,bias'])

    carry, x = _lstm_cell(carry, x, params)

    # postprocess
    x = x @ params['Dense_1,kernel'] + params['Dense_1,bias']
    x = jax.nn.gelu(x)
    logits = x @ params['Dense_2,kernel'] + params['Dense_2,bias']

    # mask illegal actions
    if legal_actions is not None:
        logits = jnp.where(legal_actions > 0, logits, -1e9)

    return carry, logits


class BCLSTMAgent:
    """BC-LSTM inference wrapper. Params baked in at init."""

    def __init__(self, weight_path='agents/hanabi/bc_lstm_weights/bc_2p.safetensors',
                 lstm_dim=512):
        self.params = load_bc_params(weight_path)
        self.lstm_dim = lstm_dim

    def initialize_carry(self, batch_dims=()):
        mem_shape = batch_dims + (self.lstm_dim,)
        return (jnp.zeros(mem_shape), jnp.zeros(mem_shape))

    @partial(jax.jit, static_argnums=(0,))
    def greedy_act(self, carry, obs, legal_actions):
        carry, logits = bc_lstm_forward(self.params, carry, obs, legal_actions)
        action = jnp.argmax(logits, axis=-1)
        return carry, action

    @partial(jax.jit, static_argnums=(0,))
    def sample_act(self, carry, obs, legal_actions, rng):
        carry, logits = bc_lstm_forward(self.params, carry, obs, legal_actions)
        action = jax.random.categorical(rng, logits)
        return carry, action
