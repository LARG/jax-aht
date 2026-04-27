import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import numpy as np

# Embedding dimension constant - modify this to change the latent space size
LATENT_DIM = 16


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
    latent_dim: int = LATENT_DIM

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
    latent_dim: int = LATENT_DIM

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


class LSTMTrajectoryClassifier(nn.Module):
    obs_dim: int
    max_seq_len: int
    hidden_dim: int
    num_classes: int
    latent_dim: int = LATENT_DIM

    def setup(self):
        self.encoder = LSTMTrajectoryEncoder(
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
        )
        self.classifier = nn.Dense(self.num_classes, kernel_init=orthogonal(1.0), bias_init=constant(0.0))

    def __call__(self, x, mask):
        latent = self.encoder(x, mask)
        logits = self.classifier(latent)
        return logits

    def encode(self, x, mask):
        return self.encoder(x, mask)


def create_classifier(obs_dim, max_seq_len, hidden_dim, num_classes, latent_dim=LATENT_DIM):
    return LSTMTrajectoryClassifier(
        obs_dim=obs_dim,
        max_seq_len=max_seq_len,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_classes=num_classes,
    )


def init_classifier(rng, obs_dim, max_seq_len, hidden_dim, num_classes, learning_rate, latent_dim=LATENT_DIM):
    model = create_classifier(obs_dim, max_seq_len, hidden_dim, num_classes, latent_dim)
    rng, rng_init = jax.random.split(rng)
    dummy_x = jnp.zeros((max_seq_len, obs_dim))
    dummy_mask = jnp.ones((max_seq_len,))
    params = model.init(rng_init, dummy_x, dummy_mask)
    tx = optax.adam(learning_rate)
    train_state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return rng, train_state, model


def make_classifier_train_step(model):
    @jax.jit
    def train_step(train_state, batch_x, batch_mask, batch_y):
        def loss_fn(params):
            logits = jax.vmap(lambda x, m: model.apply(params, x, m))(batch_x, batch_mask)
            return optax.softmax_cross_entropy_with_integer_labels(logits, batch_y).mean()
        loss, grads = jax.value_and_grad(loss_fn)(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, loss

    return train_step


def make_classifier_eval_step(model):
    @jax.jit
    def eval_step(params, x, mask):
        logits = jax.vmap(lambda x_i, m_i: model.apply(params, x_i, m_i))(x, mask)
        return logits

    return eval_step


def train_classifier(rng, train_state, train_step_fn, padded_episodes, masks, labels, num_epochs, batch_size):
    N = padded_episodes.shape[0]
    num_batches = max(1, N // batch_size)

    losses = []
    for epoch in range(num_epochs):
        rng, rng_perm = jax.random.split(rng)
        perm = np.array(jax.random.permutation(rng_perm, N))
        epoch_losses = []
        for batch_idx in range(num_batches):
            idx = perm[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            batch_x = padded_episodes[idx]
            batch_mask = masks[idx]
            batch_y = labels[idx]
            train_state, loss = train_step_fn(train_state, batch_x, batch_mask, batch_y)
            epoch_losses.append(float(loss))
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
    return rng, train_state, losses


def pad_labeled_episodes(episodes_with_labels, max_samples_per_class=None):
    """Pad episodes with labels to the same length.

    Args:
        episodes_with_labels: List of (trajectory_array, label) tuples
        max_samples_per_class: If set, randomly subsample each class to this many episodes.

    Returns:
        padded: (N, max_len, obs_dim) array of padded observations
        masks: (N, max_len) array indicating valid timesteps
        labels: (N,) array of integer labels
        max_len: maximum sequence length
        label_to_idx: dict mapping string labels to integer indices
    """
    episodes, string_labels = zip(*episodes_with_labels)
    episodes = list(episodes)
    string_labels = list(string_labels)
    print(f"[pad_labeled_episodes] Processing {len(episodes)} labeled trajectory episodes")

    # Create label mapping
    unique_labels = sorted(set(string_labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    print(f"[pad_labeled_episodes] Found {len(unique_labels)} unique labels")

    if max_samples_per_class is not None:
        rng_np = np.random.default_rng(42)
        by_class = {label: [] for label in unique_labels}
        for ep, label in zip(episodes, string_labels):
            by_class[label].append(ep)
        episodes, string_labels = [], []
        for label in unique_labels:
            class_eps = by_class[label]
            if len(class_eps) > max_samples_per_class:
                chosen = rng_np.choice(len(class_eps), size=max_samples_per_class, replace=False)
                class_eps = [class_eps[i] for i in chosen]
            episodes.extend(class_eps)
            string_labels.extend([label] * len(class_eps))
        print(f"[pad_labeled_episodes] Subsampled to {len(episodes)} episodes (max {max_samples_per_class}/class)")

    max_len = max(len(ep) for ep in episodes)
    obs_dim = episodes[0].shape[-1]
    N = len(episodes)

    padded = np.zeros((N, max_len, obs_dim), dtype=np.float32)
    masks = np.zeros((N, max_len), dtype=np.float32)
    labels_arr = np.zeros((N,), dtype=np.int32)

    for i, (ep, label) in enumerate(zip(episodes, string_labels)):
        L = len(ep)
        padded[i, :L] = ep
        masks[i, :L] = 1.0
        labels_arr[i] = label_to_idx[label]

    print(f"[pad_labeled_episodes] Padded shape: {padded.shape}, masks shape: {masks.shape}, labels shape: {labels_arr.shape}, max_len: {max_len}")
    return padded, masks, labels_arr, max_len, label_to_idx


def pad_episodes(episodes):
    """Pad episodes to the same length.

    Handles raw trajectory data (arrays only).

    Returns:
        padded: (N, max_len, obs_dim) numpy array of padded observations
        masks: (N, max_len) numpy array indicating valid timesteps
        max_len: maximum sequence length
        agent_indices: None (no artificial indices)
    """
    obs_list = episodes
    print(f"[pad_episodes] Processing {len(obs_list)} raw trajectory episodes")

    max_len = max(len(ep) for ep in obs_list)
    obs_dim = obs_list[0].shape[-1]
    N = len(obs_list)

    padded = np.zeros((N, max_len, obs_dim), dtype=np.float32)
    masks = np.zeros((N, max_len), dtype=np.float32)

    for i, ep in enumerate(obs_list):
        L = len(ep)
        padded[i, :L] = ep
        masks[i, :L] = 1.0

    print(f"[pad_episodes] Padded shape: {padded.shape}, masks shape: {masks.shape}, max_len: {max_len}")
    return padded, masks, max_len, None


def create_autoencoder(obs_dim, max_seq_len, hidden_dim, latent_dim=LATENT_DIM):
    return LSTMTrajectoryAutoencoder(
        obs_dim=obs_dim,
        max_seq_len=max_seq_len,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
    )


def init_autoencoder(rng, obs_dim, max_seq_len, hidden_dim, learning_rate, latent_dim=LATENT_DIM):
    model = create_autoencoder(obs_dim, max_seq_len, hidden_dim, latent_dim)
    rng, rng_init = jax.random.split(rng)
    dummy_x = jnp.zeros((max_seq_len, obs_dim))
    dummy_mask = jnp.ones((max_seq_len,))
    params = model.init(rng_init, dummy_x, dummy_mask)
    tx = optax.adam(learning_rate)
    train_state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return rng, train_state, model


def make_train_step(model, obs_dim):
    @jax.jit
    def train_step(train_state, batch_x, batch_mask):
        def loss_fn(params):
            def single_loss(x, mask):
                reconstructed, _ = model.apply(params, x, mask)
                mask_expanded = mask[:, None]
                sq_error = ((reconstructed - x) ** 2) * mask_expanded
                return sq_error.sum() / (mask_expanded.sum() * obs_dim + 1e-8)
            return jax.vmap(single_loss)(batch_x, batch_mask).mean()
        loss, grads = jax.value_and_grad(loss_fn)(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, loss

    return train_step


def train_autoencoder(rng, train_state, train_step_fn, padded_episodes, masks, num_epochs, batch_size):
    N = padded_episodes.shape[0]
    num_batches = max(1, N // batch_size)

    losses = []
    for epoch in range(num_epochs):
        rng, rng_perm = jax.random.split(rng)
        perm = np.array(jax.random.permutation(rng_perm, N))
        epoch_losses = []
        for batch_idx in range(num_batches):
            idx = perm[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            batch_x = padded_episodes[idx]
            batch_mask = masks[idx]
            train_state, loss = train_step_fn(train_state, batch_x, batch_mask)
            epoch_losses.append(float(loss))
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
    return rng, train_state, losses


def encode_episodes(model, train_state, episodes, max_seq_len):
    """Encode episodes using the trained autoencoder model.
    
    Handles raw trajectory data (arrays only).
    
    Args:
        model: The autoencoder model
        train_state: Training state with parameters
        episodes: List of raw trajectory arrays
        max_seq_len: Maximum sequence length for padding
        
    Returns:
        Array of shape (N, latent_dim) containing latent encodings
    """
    # All episodes are raw trajectory arrays
    obs_list = episodes
    print(f"[encode_episodes] Processing {len(obs_list)} raw trajectory episodes")
    
    obs_dim = obs_list[0].shape[-1]
    print(f"[encode_episodes] obs_dim={obs_dim}, max_seq_len={max_seq_len}")
    print(f"[encode_episodes] Episode lengths: min={min(len(ep) for ep in obs_list)}, max={max(len(ep) for ep in obs_list)}")

    @jax.jit
    def encode_one(params, x, mask):
        return model.apply(params, x, mask, method=model.encode)

    latents = []
    for i, ep in enumerate(obs_list):
        L = len(ep)
        padded = np.zeros((max_seq_len, obs_dim), dtype=np.float32)
        mask = np.zeros((max_seq_len,), dtype=np.float32)
        padded[:L] = ep
        mask[:L] = 1.0
        latent = encode_one(train_state.params, jnp.array(padded), jnp.array(mask))
        latents.append(np.array(latent))

    result = np.stack(latents)
    print(f"[encode_episodes] Successfully encoded {len(result)} episodes. Result shape: {result.shape}")
    return result
