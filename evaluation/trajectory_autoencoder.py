from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import numpy as np


class MaskedLSTMCell(nn.Module):
    """LSTM cell that conditionally updates carry based on a mask."""
    features: int

    def setup(self):
        self.lstm = nn.OptimizedLSTMCell(features=self.features)

    def __call__(self, carry, inputs):
        x_t, m_t = inputs
        new_carry, y = self.lstm(carry, x_t)
        carry_out = jax.tree.map(
            lambda nc, oc: jnp.where(m_t, nc, oc), new_carry, carry
        )
        return carry_out, y

    def initialize_carry(self, rng, input_shape):
        return self.lstm.initialize_carry(rng, input_shape)


class AutoregressiveLSTMCell(nn.Module):
    """LSTM cell that feeds its hidden state back as input (autoregressive)."""
    features: int
    output_dim: int

    def setup(self):
        self.lstm = nn.OptimizedLSTMCell(features=self.features)
        self.output_proj = nn.Dense(self.output_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))

    def __call__(self, carry, _):
        _, h_prev = carry
        new_carry, y = self.lstm(carry, h_prev)
        pred = self.output_proj(y)
        return new_carry, pred


class LSTMTrajectoryEncoder(nn.Module):
    hidden_dim: int
    latent_dim: int

    def setup(self):
        self.input_proj = nn.Dense(self.hidden_dim, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))
        ScanMaskedLSTM = nn.scan(
            MaskedLSTMCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0,
        )
        self.scan_lstm = ScanMaskedLSTM(features=self.hidden_dim)
        self.latent_proj = nn.Dense(self.latent_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))

    def __call__(self, x, mask):
        # x: (seq_len, obs_dim), mask: (seq_len,)
        x = self.input_proj(x)
        x = nn.leaky_relu(x)

        init_carry = self.scan_lstm.initialize_carry(jax.random.PRNGKey(0), (self.hidden_dim,))
        final_carry, _ = self.scan_lstm(init_carry, (x, mask[:, None]))

        # final_carry is (c, h); use h as context
        _, h = final_carry
        latent = self.latent_proj(h)
        return latent


class LSTMTrajectoryDecoder(nn.Module):
    output_dim: int
    max_seq_len: int
    hidden_dim: int

    def setup(self):
        self.latent_expand = nn.Dense(self.hidden_dim, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))
        ScanAutoregLSTM = nn.scan(
            AutoregressiveLSTMCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0,
        )
        self.scan_lstm = ScanAutoregLSTM(features=self.hidden_dim, output_dim=self.output_dim)

    def __call__(self, latent):
        # latent: (latent_dim,)
        context = self.latent_expand(latent)
        context = nn.leaky_relu(context)

        carry = (context, context)
        # Dummy input; autoregressive cell ignores it
        dummy = jnp.zeros((self.max_seq_len,))
        _, reconstructed = self.scan_lstm(carry, dummy)
        return reconstructed


class LSTMTrajectoryAutoencoder(nn.Module):
    obs_dim: int
    max_seq_len: int
    hidden_dim: int
    latent_dim: int

    def setup(self):
        self.encoder = LSTMTrajectoryEncoder(
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
        )
        self.decoder = LSTMTrajectoryDecoder(
            output_dim=self.obs_dim,
            max_seq_len=self.max_seq_len,
            hidden_dim=self.hidden_dim,
        )

    def __call__(self, x, mask):
        latent = self.encoder(x, mask)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

    def encode(self, x, mask):
        return self.encoder(x, mask)


def pad_episodes(episodes):
    max_len = max(len(ep) for ep in episodes)
    obs_dim = episodes[0].shape[-1]
    N = len(episodes)

    padded = np.zeros((N, max_len, obs_dim), dtype=np.float32)
    masks = np.zeros((N, max_len), dtype=np.float32)

    for i, ep in enumerate(episodes):
        L = len(ep)
        padded[i, :L] = ep
        masks[i, :L] = 1.0

    return jnp.array(padded), jnp.array(masks), max_len


def create_autoencoder(obs_dim, max_seq_len, hidden_dim, latent_dim):
    return LSTMTrajectoryAutoencoder(
        obs_dim=obs_dim,
        max_seq_len=max_seq_len,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
    )


def init_autoencoder(rng, obs_dim, max_seq_len, hidden_dim, latent_dim, learning_rate):
    model = create_autoencoder(obs_dim, max_seq_len, hidden_dim, latent_dim)
    rng, rng_init = jax.random.split(rng)
    dummy_x = jnp.zeros((max_seq_len, obs_dim))
    dummy_mask = jnp.ones((max_seq_len,))
    params = model.init(rng_init, dummy_x, dummy_mask)
    tx = optax.adam(learning_rate)
    train_state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return rng, train_state, model


def make_train_step(model, obs_dim):
    def loss_fn(params, x, mask):
        reconstructed, _ = model.apply(params, x, mask)
        mask_expanded = mask[:, None]
        sq_error = ((reconstructed - x) ** 2) * mask_expanded
        mse = sq_error.sum() / (mask_expanded.sum() * obs_dim + 1e-8)
        return mse

    @jax.jit
    def train_step(train_state, batch_x, batch_mask):
        grad_fn = jax.grad(lambda p: jax.vmap(partial(loss_fn, p))(batch_x, batch_mask).mean())
        grads = grad_fn(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)
        loss = jax.vmap(partial(loss_fn, train_state.params))(batch_x, batch_mask).mean()
        return train_state, loss

    return train_step


def train_autoencoder(rng, train_state, train_step_fn, padded_episodes, masks, num_epochs, batch_size):
    N = padded_episodes.shape[0]
    num_batches = max(1, N // batch_size)

    losses = []
    for epoch in range(num_epochs):
        rng, rng_perm = jax.random.split(rng)
        perm = jax.random.permutation(rng_perm, N)
        padded_shuffled = padded_episodes[perm]
        masks_shuffled = masks[perm]
        epoch_losses = []
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, N)
            batch_x = padded_shuffled[start:end]
            batch_mask = masks_shuffled[start:end]
            train_state, loss = train_step_fn(train_state, batch_x, batch_mask)
            epoch_losses.append(loss)
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
    return rng, train_state, losses


def encode_episodes(model, train_state, episodes, max_seq_len):
    obs_dim = episodes[0].shape[-1]

    @jax.jit
    def encode_one(params, x, mask):
        return model.apply(params, x, mask, method=model.encode)

    latents = []
    for ep in episodes:
        L = len(ep)
        padded = np.zeros((max_seq_len, obs_dim), dtype=np.float32)
        mask = np.zeros((max_seq_len,), dtype=np.float32)
        padded[:L] = ep
        mask[:L] = 1.0
        latent = encode_one(train_state.params, jnp.array(padded), jnp.array(mask))
        latents.append(np.array(latent))

    return np.stack(latents)
