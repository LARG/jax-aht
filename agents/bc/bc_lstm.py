from __future__ import annotations
import os
from functools import partial
from typing import Any, NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.training.train_state import TrainState
import optax
import distrax
from safetensors.numpy import load_file

class BCLSTMConfig(NamedTuple):
    obs_dim: int
    action_dim: int
    preprocess_dim: int = 1024
    lstm_dim: int = 512
    postprocess_dim: int = 256
    dropout_rate: float = 0.0


class BCLSTMNetwork(nn.Module):
    action_dim: int
    preprocess_dim: int = 1024
    lstm_dim: int = 512
    postprocess_dim: int = 256
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, carry, obs, training=False):
        x = nn.Dense(self.preprocess_dim)(obs)
        x = nn.gelu(x)
        x = nn.LayerNorm()(x)

        lstm_cell = nn.OptimizedLSTMCell(self.lstm_dim)
        carry, x = lstm_cell(carry, x)

        x = nn.Dense(self.postprocess_dim)(x)
        x = nn.gelu(x)
        if self.dropout_rate > 0 and training:
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
        logits = nn.Dense(self.action_dim)(x)
        return carry, logits


class BCLSTMAgent:
    def __init__(self, config: BCLSTMConfig, params=None, weight_path=None):
        self.config = config
        self.network = BCLSTMNetwork(
            action_dim=config.action_dim,
            preprocess_dim=config.preprocess_dim,
            lstm_dim=config.lstm_dim,
            postprocess_dim=config.postprocess_dim,
            dropout_rate=config.dropout_rate,
        )
        if params is not None:
            self.params = params
        elif weight_path is not None:
            self.params = self._load_weights(weight_path)
        else:
            self.params = None

    def init_params(self, rng):
        dummy_carry = self.init_carry()
        dummy_obs = jnp.zeros((self.config.obs_dim,))
        variables = self.network.init(rng, dummy_carry, dummy_obs)
        self.params = variables['params']
        return self.params

    def init_carry(self, batch_dims=()):
        # Initialize LSTM carry (c, h) w/ zeros
        shape = batch_dims + (self.config.lstm_dim,)
        return (jnp.zeros(shape), jnp.zeros(shape))

    @partial(jax.jit, static_argnums=(0,))
    def forward(self, params, carry, obs, avail_actions=None):
        carry, logits = self.network.apply({'params': params}, carry, obs)
        if avail_actions is not None:
            logits = jnp.where(avail_actions > 0, logits, -1e9)
        return carry, logits

    @partial(jax.jit, static_argnums=(0,))
    def greedy_act(self, carry, obs, avail_actions):
        carry, logits = self.forward(self.params, carry, obs, avail_actions)
        return carry, jnp.argmax(logits, axis=-1)

    @partial(jax.jit, static_argnums=(0,))
    def sample_act(self, carry, obs, avail_actions, rng):
        carry, logits = self.forward(self.params, carry, obs, avail_actions)
        return carry, jax.random.categorical(rng, logits)

    def _load_weights(self, path):
        if not os.path.isabs(path):
            from common.save_load_utils import REPO_PATH
            path = os.path.join(REPO_PATH, path)
        raw = load_file(path)

        params = {}
        for key, val in raw.items():
            parts = key.split('/')
            d = params
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = jnp.array(val)
        return params

    def save_weights(self, path):
        from safetensors.numpy import save_file
        flat = {k: np.array(v) for k, v in jax.tree.leaves_with_path(self.params)}

        flat_dict = {}
        def _flatten(prefix, d):
            if isinstance(d, dict):
                for k, v in d.items():
                    _flatten(f"{prefix}/{k}" if prefix else k, v)
            else:
                flat_dict[prefix] = np.array(d)
        _flatten("", self.params)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_file(flat_dict, path)


def compute_bc_loss(params, network, carry, obs_seq, action_seq, avail_seq, mask):
    # Cross-entropy BC loss over a padded sequence batch.
    _, seq_len, _ = obs_seq.shape

    def step_fn(carry, t):
        action_t = action_seq[:, t]
        mask_t = mask[:, t]

        carry, logits = network.apply({'params': params}, carry, obs_seq[:, t, :])
        pi = distrax.Categorical(logits=logits)
        step_loss = -pi.log_prob(action_t) * mask_t
        return carry, step_loss

    _, all_losses = jax.lax.scan(step_fn, carry, jnp.arange(seq_len))
    total_loss = all_losses.sum()
    num_valid = mask.sum()
    return total_loss / jnp.maximum(num_valid, 1.0)


def create_train_state(config: BCLSTMConfig, rng, learning_rate=3e-4):
    agent = BCLSTMAgent(config)
    params = agent.init_params(rng)
    tx = optax.adam(learning_rate)
    return TrainState.create(
        apply_fn=agent.network.apply,
        params=params,
        tx=tx,
    ), agent
