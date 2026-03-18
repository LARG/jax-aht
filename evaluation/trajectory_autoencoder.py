from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import numpy as np

from agents.s5_actor_critic import StackedEncoderModel, init_S5SSM, make_DPLR_HiPPO


def make_ssm_init_fn(d_model, ssm_size, blocks=1):
    block_size = int(ssm_size / blocks)
    Lambda, _, _, V, _ = make_DPLR_HiPPO(ssm_size)
    block_size = block_size // 2
    ssm_size_half = ssm_size // 2
    Lambda = Lambda[:block_size]
    V = V[:, :block_size]
    Vinv = V.conj().T

    return init_S5SSM(
        H=d_model,
        P=ssm_size_half,
        Lambda_re_init=Lambda.real,
        Lambda_im_init=Lambda.imag,
        V=V,
        Vinv=Vinv,
    )


class S5TrajectoryEncoder(nn.Module):
    ssm_init_fn: Any
    d_model: int
    ssm_size: int
    ssm_n_layers: int
    latent_dim: int

    def setup(self):
        self.input_proj = nn.Dense(self.d_model, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))
        self.s5 = StackedEncoderModel(
            ssm=self.ssm_init_fn,
            d_model=self.d_model,
            n_layers=self.ssm_n_layers,
            activation="full_glu",
            do_norm=True,
            prenorm=True,
            do_gtrxl_norm=True,
        )
        self.latent_proj = nn.Dense(self.latent_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))

    def __call__(self, x, mask):
        x = self.input_proj(x)
        x = nn.leaky_relu(x)
        x = x[:, None, :]

        dones = jnp.zeros((x.shape[0], 1))
        hidden = StackedEncoderModel.initialize_carry(1, self.ssm_size // 2, self.ssm_n_layers)
        _, x = self.s5(hidden, x, dones)
        x = x[:, 0, :]

        mask_expanded = mask[:, None]
        x = (x * mask_expanded).sum(axis=0) / (mask_expanded.sum(axis=0) + 1e-8)
        latent = self.latent_proj(x)
        return latent


class S5TrajectoryDecoder(nn.Module):
    ssm_init_fn: Any
    output_dim: int
    max_seq_len: int
    d_model: int
    ssm_size: int
    ssm_n_layers: int

    def setup(self):
        self.latent_expand = nn.Dense(self.d_model, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))
        self.s5 = StackedEncoderModel(
            ssm=self.ssm_init_fn,
            d_model=self.d_model,
            n_layers=self.ssm_n_layers,
            activation="full_glu",
            do_norm=True,
            prenorm=True,
            do_gtrxl_norm=True,
        )
        self.output_proj = nn.Dense(self.output_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))

    def __call__(self, latent, seq_len):
        x = self.latent_expand(latent)
        x = nn.leaky_relu(x)
        x = jnp.broadcast_to(x[None, :], (self.max_seq_len, self.d_model))

        x = x[:, None, :]
        dones = jnp.zeros((self.max_seq_len, 1))
        hidden = StackedEncoderModel.initialize_carry(1, self.ssm_size // 2, self.ssm_n_layers)
        _, x = self.s5(hidden, x, dones)
        x = x[:, 0, :]

        reconstructed = self.output_proj(x)
        return reconstructed


class S5TrajectoryAutoencoder(nn.Module):
    ssm_init_fn: Any
    obs_dim: int
    max_seq_len: int
    d_model: int
    ssm_size: int
    ssm_n_layers: int
    latent_dim: int

    def setup(self):
        self.encoder = S5TrajectoryEncoder(
            ssm_init_fn=self.ssm_init_fn,
            d_model=self.d_model,
            ssm_size=self.ssm_size,
            ssm_n_layers=self.ssm_n_layers,
            latent_dim=self.latent_dim,
        )
        self.decoder = S5TrajectoryDecoder(
            ssm_init_fn=self.ssm_init_fn,
            output_dim=self.obs_dim,
            max_seq_len=self.max_seq_len,
            d_model=self.d_model,
            ssm_size=self.ssm_size,
            ssm_n_layers=self.ssm_n_layers,
        )

    def __call__(self, x, mask):
        latent = self.encoder(x, mask)
        reconstructed = self.decoder(latent, self.max_seq_len)
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


def create_autoencoder(obs_dim, max_seq_len, d_model, ssm_size, ssm_n_layers, latent_dim):
    ssm_init_fn = make_ssm_init_fn(d_model, ssm_size, blocks=1)
    return S5TrajectoryAutoencoder(
        ssm_init_fn=ssm_init_fn,
        obs_dim=obs_dim,
        max_seq_len=max_seq_len,
        d_model=d_model,
        ssm_size=ssm_size,
        ssm_n_layers=ssm_n_layers,
        latent_dim=latent_dim,
    )


def init_autoencoder(rng, obs_dim, max_seq_len, d_model, ssm_size, ssm_n_layers, latent_dim, learning_rate):
    model = create_autoencoder(obs_dim, max_seq_len, d_model, ssm_size, ssm_n_layers, latent_dim)
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

    for _ in range(num_epochs):
        rng, rng_perm = jax.random.split(rng)
        perm = jax.random.permutation(rng_perm, N)
        padded_shuffled = padded_episodes[perm]
        masks_shuffled = masks[perm]
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, N)
            batch_x = padded_shuffled[start:end]
            batch_mask = masks_shuffled[start:end]
            train_state, _ = train_step_fn(train_state, batch_x, batch_mask)
    return rng, train_state


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
