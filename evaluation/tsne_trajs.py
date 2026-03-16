"""

to visualize the qualitative diversities of some teammate set
a low hanging fruit is to use t-sne

inputs:
- some environment (coming from the envs/ --> make_env())
- some teammate set (in the format compatible with heldout_eval in this folder)

to quantify the trajectories, we will train an autoencoder to encode trajectories

1. collect trajectories
- random sampled trajectories from the environment
randomly sampled because we don't want to bias the autoencoder by pre-seeing the teammate trajectories
NOTE TO SELF: in practice, this seems a bit sus. have to think about this

2. train an autoencoder to encode trajectories
since trajectories are sequences, we will do this
input → S5 encoder → latent sequence Z(t) → S5 decoder → reconstructed sequence

3. encode the teammate trajectories using the trained autoencoder, and visualize them using t-sne

"""

from functools import partial
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

from envs import make_env
from agents.s5_actor_critic import (
    StackedEncoderModel,
    init_S5SSM,
    make_DPLR_HiPPO,
)
from common.agent_loader_from_config import initialize_rl_agent_from_config

# ── Hardcoded config for LBF ─────────────────────────────────────────────────
ENV_NAME = "lbf"
ENV_KWARGS = {}  # default: grid_size=7, num_food=3, 2 agents

# Rollout config
NUM_ENVS = 64            # parallel environments
ROLLOUT_STEPS = 128      # steps per rollout
ROLLOUTS_PER_ITER = 5    # rollouts to collect per iteration
NUM_ITERS = 10            # number of collect→train iterations

# Autoencoder S5 config
D_MODEL = 64
SSM_SIZE = 64
SSM_N_LAYERS = 3
BLOCKS = 1
LATENT_DIM = 128
FC_HIDDEN_DIM = 128
S5_ACTIVATION = "full_glu"
S5_DO_NORM = True
S5_PRENORM = True
S5_DO_GTRXL_NORM = True

# Training config
LEARNING_RATE = 3e-4
NUM_EPOCHS = 200
BATCH_SIZE = 64

# Buffer config
MAX_BUFFER_SIZE = 1024


# ── Autoencoder network ──────────────────────────────────────────────────────

def make_ssm_init_fn(d_model, ssm_size, blocks=1):
    """Create an S5 SSM init function (same logic as S5ActorCriticPolicy)."""
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
    """S5-based encoder: sequence → single latent vector."""
    ssm_init_fn: Any
    d_model: int = D_MODEL
    ssm_size: int = SSM_SIZE
    ssm_n_layers: int = SSM_N_LAYERS
    latent_dim: int = LATENT_DIM
    fc_hidden_dim: int = FC_HIDDEN_DIM

    def setup(self):
        self.input_proj = nn.Dense(self.d_model, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))
        self.s5 = StackedEncoderModel(
            ssm=self.ssm_init_fn,
            d_model=self.d_model,
            n_layers=self.ssm_n_layers,
            activation=S5_ACTIVATION,
            do_norm=S5_DO_NORM,
            prenorm=S5_PRENORM,
            do_gtrxl_norm=S5_DO_GTRXL_NORM,
        )
        self.latent_proj = nn.Dense(self.latent_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))

    def __call__(self, x, mask):
        """
        Args:
            x: (L, obs_dim) single trajectory
            mask: (L,) binary mask, 1 for valid timesteps
        Returns:
            latent: (latent_dim,) single vector for the trajectory
        """
        # Project input to d_model
        x = self.input_proj(x)
        x = nn.leaky_relu(x)

        # Run through S5 (no resets within a single episode)
        batch_size = 1
        # Reshape for S5: (L, 1, d_model)
        x = x[:, None, :]
        dones = jnp.zeros((x.shape[0], 1))
        hidden = StackedEncoderModel.initialize_carry(batch_size, self.ssm_size // 2, self.ssm_n_layers)
        _, x = self.s5(hidden, x, dones)
        x = x[:, 0, :]  # (L, d_model)

        # Masked mean pool over time to get a single vector
        mask_expanded = mask[:, None]  # (L, 1)
        x = (x * mask_expanded).sum(axis=0) / (mask_expanded.sum(axis=0) + 1e-8)

        # Project to latent
        latent = self.latent_proj(x)
        return latent


class S5TrajectoryDecoder(nn.Module):
    """S5-based decoder: latent vector → reconstructed sequence."""
    ssm_init_fn: Any
    output_dim: int
    max_seq_len: int
    d_model: int = D_MODEL
    ssm_size: int = SSM_SIZE
    ssm_n_layers: int = SSM_N_LAYERS
    latent_dim: int = LATENT_DIM
    fc_hidden_dim: int = FC_HIDDEN_DIM

    def setup(self):
        self.latent_expand = nn.Dense(self.d_model, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))
        self.s5 = StackedEncoderModel(
            ssm=self.ssm_init_fn,
            d_model=self.d_model,
            n_layers=self.ssm_n_layers,
            activation=S5_ACTIVATION,
            do_norm=S5_DO_NORM,
            prenorm=S5_PRENORM,
            do_gtrxl_norm=S5_DO_GTRXL_NORM,
        )
        self.output_proj = nn.Dense(self.output_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))

    def __call__(self, latent, seq_len):
        """
        Args:
            latent: (latent_dim,) single vector
            seq_len: int, length of sequence to reconstruct
        Returns:
            reconstructed: (max_seq_len, output_dim)
        """
        # Broadcast latent across time: (max_seq_len, d_model)
        x = self.latent_expand(latent)
        x = nn.leaky_relu(x)
        x = jnp.broadcast_to(x[None, :], (self.max_seq_len, self.d_model))

        # Run through S5 decoder
        batch_size = 1
        x = x[:, None, :]  # (max_seq_len, 1, d_model)
        dones = jnp.zeros((self.max_seq_len, 1))
        hidden = StackedEncoderModel.initialize_carry(batch_size, self.ssm_size // 2, self.ssm_n_layers)
        _, x = self.s5(hidden, x, dones)
        x = x[:, 0, :]  # (max_seq_len, d_model)

        # Project to output
        reconstructed = self.output_proj(x)
        return reconstructed


class S5TrajectoryAutoencoder(nn.Module):
    """Full autoencoder: encoder + decoder."""
    ssm_init_fn: Any
    obs_dim: int
    max_seq_len: int
    d_model: int = D_MODEL
    ssm_size: int = SSM_SIZE
    ssm_n_layers: int = SSM_N_LAYERS
    latent_dim: int = LATENT_DIM
    fc_hidden_dim: int = FC_HIDDEN_DIM

    def setup(self):
        self.encoder = S5TrajectoryEncoder(
            ssm_init_fn=self.ssm_init_fn,
            d_model=self.d_model,
            ssm_size=self.ssm_size,
            ssm_n_layers=self.ssm_n_layers,
            latent_dim=self.latent_dim,
            fc_hidden_dim=self.fc_hidden_dim,
        )
        self.decoder = S5TrajectoryDecoder(
            ssm_init_fn=self.ssm_init_fn,
            output_dim=self.obs_dim,
            max_seq_len=self.max_seq_len,
            d_model=self.d_model,
            ssm_size=self.ssm_size,
            ssm_n_layers=self.ssm_n_layers,
            latent_dim=self.latent_dim,
            fc_hidden_dim=self.fc_hidden_dim,
        )

    def __call__(self, x, mask):
        """
        Args:
            x: (L, obs_dim) padded trajectory
            mask: (L,) binary mask
        Returns:
            reconstructed: (L, obs_dim)
            latent: (latent_dim,)
        """
        latent = self.encoder(x, mask)
        reconstructed = self.decoder(latent, self.max_seq_len)
        return reconstructed, latent

    def encode(self, x, mask):
        """Encode only - for inference."""
        return self.encoder(x, mask)


# ── Part 1: Collect random trajectories ──────────────────────────────────────

def collect_random_trajectories(rng, env, num_rollouts=ROLLOUTS_PER_ITER):
    """Collect trajectories by running random actions in LBF.

    Args:
        rng: jax random key
        env: the environment instance
        num_rollouts: number of rollouts to perform

    Returns:
        rng: updated rng key
        episodes: list of arrays, each (ep_len, obs_dim) for agent_0
    """
    all_episodes = []

    for rollout_idx in range(num_rollouts):
        rng, rng_reset = jax.random.split(rng)
        # Vectorized reset across NUM_ENVS
        rng_resets = jax.random.split(rng_reset, NUM_ENVS)
        obs, state = jax.vmap(env.reset)(rng_resets)

        # Storage: (ROLLOUT_STEPS, NUM_ENVS, obs_dim) and dones
        obs_buffer = []
        done_buffer = []

        for step in range(ROLLOUT_STEPS):
            rng, rng_act, rng_step = jax.random.split(rng, 3)

            # Store current obs for agent_0
            obs_buffer.append(obs["agent_0"])

            # Sample random actions for all agents
            actions = {}
            for agent in env.agents:
                rng, rng_a = jax.random.split(rng)
                avail = state.avail_actions[agent]  # (NUM_ENVS, num_actions)
                rng_keys = jax.random.split(rng_a, NUM_ENVS)
                logits = jnp.where(avail, 0.0, -1e10)
                actions[agent] = jax.vmap(
                    lambda key, lg: jax.random.categorical(key, lg)
                )(rng_keys, logits)

            # Step all envs
            rng_steps = jax.random.split(rng_step, NUM_ENVS)
            obs, state, rewards, dones, infos = jax.vmap(env.step)(rng_steps, state, actions)
            done_buffer.append(dones["__all__"])

        # Stack: (ROLLOUT_STEPS, NUM_ENVS, obs_dim) and (ROLLOUT_STEPS, NUM_ENVS)
        obs_buffer = jnp.stack(obs_buffer)
        done_buffer = jnp.stack(done_buffer)

        # Extract individual episodes per env
        obs_np = np.array(obs_buffer)
        done_np = np.array(done_buffer)

        for env_idx in range(NUM_ENVS):
            env_obs = obs_np[:, env_idx, :]      # (ROLLOUT_STEPS, obs_dim)
            env_done = done_np[:, env_idx]        # (ROLLOUT_STEPS,)

            # Split at done boundaries to get individual episodes
            done_indices = np.where(env_done)[0]
            ep_start = 0
            for done_idx in done_indices:
                ep_end = done_idx + 1  # include the done step
                episode = env_obs[ep_start:ep_end]
                if len(episode) > 1:  # skip trivially short episodes
                    all_episodes.append(episode)
                ep_start = ep_end

    return rng, all_episodes


# ── Part 1b: Pad episodes into a batch ───────────────────────────────────────

def pad_episodes(episodes):
    """Pad episodes to the same length and create masks.

    Returns:
        padded: (N, max_len, obs_dim) array
        masks: (N, max_len) binary array
        max_len: int
    """
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


# ── Part 2: Train the autoencoder ────────────────────────────────────────────

def create_autoencoder(obs_dim, max_seq_len):
    """Create the S5 trajectory autoencoder."""
    ssm_init_fn = make_ssm_init_fn(D_MODEL, SSM_SIZE, BLOCKS)
    model = S5TrajectoryAutoencoder(
        ssm_init_fn=ssm_init_fn,
        obs_dim=obs_dim,
        max_seq_len=max_seq_len,
        d_model=D_MODEL,
        ssm_size=SSM_SIZE,
        ssm_n_layers=SSM_N_LAYERS,
        latent_dim=LATENT_DIM,
        fc_hidden_dim=FC_HIDDEN_DIM,
    )
    return model


def init_autoencoder(rng, obs_dim, max_seq_len):
    """Create and initialize the autoencoder, optimizer, and train state."""
    model = create_autoencoder(obs_dim, max_seq_len)

    rng, rng_init = jax.random.split(rng)
    dummy_x = jnp.zeros((max_seq_len, obs_dim))
    dummy_mask = jnp.ones((max_seq_len,))
    params = model.init(rng_init, dummy_x, dummy_mask)

    tx = optax.adam(LEARNING_RATE)
    train_state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return rng, train_state, model


def make_train_step(model, obs_dim):
    """Create a jitted train step function (closure over model and obs_dim)."""

    def loss_fn(params, x, mask):
        reconstructed, latent = model.apply(params, x, mask)
        mask_expanded = mask[:, None]  # (L, 1)
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


def train_autoencoder(rng, train_state, train_step_fn, padded_episodes, masks, num_epochs=NUM_EPOCHS):
    """Train the autoencoder for some epochs on the given data.

    Args:
        rng: jax random key
        train_state: existing TrainState (params + optimizer state)
        train_step_fn: jitted train step function
        padded_episodes: (N, max_len, obs_dim)
        masks: (N, max_len)
        num_epochs: number of training epochs

    Returns:
        rng, updated train_state
    """
    N = padded_episodes.shape[0]
    num_batches = max(1, N // BATCH_SIZE)

    pbar = tqdm(range(num_epochs), desc="Training", leave=True)
    for epoch in pbar:
        rng, rng_perm = jax.random.split(rng)
        perm = jax.random.permutation(rng_perm, N)
        padded_shuffled = padded_episodes[perm]
        masks_shuffled = masks[perm]

        epoch_loss = 0.0
        for batch_idx in range(num_batches):
            start = batch_idx * BATCH_SIZE
            end = min(start + BATCH_SIZE, N)
            batch_x = padded_shuffled[start:end]
            batch_mask = masks_shuffled[start:end]
            train_state, loss = train_step_fn(train_state, batch_x, batch_mask)
            epoch_loss += float(loss)

        avg_loss = epoch_loss / num_batches
        pbar.set_postfix(loss=f"{avg_loss:.6f}")

    return rng, train_state


# ── Collect trajectories from a trained agent (IPPO selfplay) ─────────────────

IPPO_CONFIG = {
    "path": "eval_teammates/lbf/ippo/2025-04-21_23-41-17/saved_train_run",
    "actor_type": "mlp",
    "ckpt_key": "final_params",
    "idx_list": [0],
    "test_mode": False,
}


def collect_ippo_selfplay_trajectories(rng, env, num_rollouts=ROLLOUTS_PER_ITER):
    """Collect trajectories from IPPO selfplay (both agents use the same IPPO policy).

    Returns:
        rng: updated rng key
        episodes: list of arrays, each (ep_len, obs_dim) for agent_0
    """
    rng, load_rng = jax.random.split(rng)
    policy, agent_params, _, _ = initialize_rl_agent_from_config(
        IPPO_CONFIG, "ippo_mlp", env, load_rng
    )
    # agent_params has a leading dim for idx_list; squeeze to get single-agent params
    agent_params = jax.tree.map(lambda x: x[0], agent_params)

    all_episodes = []

    for rollout_idx in range(num_rollouts):
        rng, rng_reset = jax.random.split(rng)
        rng_resets = jax.random.split(rng_reset, NUM_ENVS)
        obs, state = jax.vmap(env.reset)(rng_resets)

        obs_buffer = []
        done_buffer = []

        for step in range(ROLLOUT_STEPS):
            obs_buffer.append(obs["agent_0"])

            # Both agents use the same IPPO policy
            actions = {}
            for agent in env.agents:
                rng, rng_a = jax.random.split(rng)
                avail = state.avail_actions[agent]  # (NUM_ENVS, num_actions)
                rng_keys = jax.random.split(rng_a, NUM_ENVS)
                action, _ = jax.vmap(
                    partial(policy.get_action, agent_params)
                )(
                    obs[agent],              # (NUM_ENVS, obs_dim)
                    jnp.zeros(NUM_ENVS, dtype=bool),  # done
                    avail,                   # (NUM_ENVS, num_actions)
                    None,                    # hstate (MLP has none)
                    rng_keys,                # (NUM_ENVS, 2)
                )
                actions[agent] = action

            rng, rng_step = jax.random.split(rng)
            rng_steps = jax.random.split(rng_step, NUM_ENVS)
            obs, state, rewards, dones, infos = jax.vmap(env.step)(rng_steps, state, actions)
            done_buffer.append(dones["__all__"])

        obs_buffer = jnp.stack(obs_buffer)
        done_buffer = jnp.stack(done_buffer)

        obs_np = np.array(obs_buffer)
        done_np = np.array(done_buffer)

        for env_idx in range(NUM_ENVS):
            env_obs = obs_np[:, env_idx, :]
            env_done = done_np[:, env_idx]

            done_indices = np.where(env_done)[0]
            ep_start = 0
            for done_idx in done_indices:
                ep_end = done_idx + 1
                episode = env_obs[ep_start:ep_end]
                if len(episode) > 1:
                    all_episodes.append(episode)
                ep_start = ep_end

    return rng, all_episodes


# ── Part 3: Encode trajectories and plot t-SNE ───────────────────────────────

def encode_episodes(model, train_state, episodes, max_seq_len):
    """Encode a list of episodes into latent vectors using the trained encoder.

    Args:
        model: the autoencoder model
        train_state: trained TrainState with params
        episodes: list of np arrays, each (ep_len, obs_dim)
        max_seq_len: int, pad length

    Returns:
        latents: (N, latent_dim) np array
    """
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


def plot_tsne(latents_dict, save_path="tsne_trajectories.png", perplexity=30):
    """Run t-SNE on latent vectors and plot, with different colors per group.

    Args:
        latents_dict: dict of {label: (N_i, latent_dim) np array}
        save_path: where to save the plot
        perplexity: t-SNE perplexity
    """
    all_latents = np.concatenate(list(latents_dict.values()), axis=0)
    labels = []
    for label, lat in latents_dict.items():
        labels.extend([label] * len(lat))

    # Adjust perplexity if needed
    n_samples = len(all_latents)
    perplexity = min(perplexity, max(5, n_samples // 4))

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embeddings = tsne.fit_transform(all_latents)

    fig, ax = plt.subplots(figsize=(10, 8))
    unique_labels = list(latents_dict.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_labels), 1)))

    offset = 0
    for i, label in enumerate(unique_labels):
        n = len(latents_dict[label])
        ax.scatter(
            embeddings[offset:offset+n, 0],
            embeddings[offset:offset+n, 1],
            c=[colors[i]],
            label=label,
            alpha=0.6,
            s=20,
        )
        offset += n

    ax.legend()
    ax.set_title("t-SNE of Trajectory Latents")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"t-SNE plot saved to {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = jax.random.PRNGKey(42)

    env = make_env(ENV_NAME, ENV_KWARGS)
    obs_dim = env.observation_space("agent_0").shape[0]

    all_episodes = []
    train_state = None
    model = None
    train_step_fn = None

    for iteration in range(NUM_ITERS):
        # ── Collect ──
        print(f"\n{'=' * 60}")
        print(f"Iteration {iteration+1}/{NUM_ITERS}: Collecting trajectories")
        print(f"{'=' * 60}")
        rng, new_episodes = collect_random_trajectories(rng, env)
        all_episodes.extend(new_episodes)

        # Cap buffer: drop oldest episodes to stay at MAX_BUFFER_SIZE
        if len(all_episodes) > MAX_BUFFER_SIZE:
            all_episodes = all_episodes[-MAX_BUFFER_SIZE:]

        ep_lengths = [len(ep) for ep in all_episodes]
        print(f"  +{len(new_episodes)} episodes (total: {len(all_episodes)}, "
              f"len range: {min(ep_lengths)}-{max(ep_lengths)}, mean: {np.mean(ep_lengths):.1f})")

        # ── Pad all collected episodes ──
        padded_episodes, masks, max_seq_len = pad_episodes(all_episodes)

        # ── Init model on first iteration (need max_seq_len) ──
        if train_state is None:
            rng, train_state, model = init_autoencoder(rng, obs_dim, max_seq_len)
            train_step_fn = make_train_step(model, obs_dim)

        # ── Train ──
        print(f"Iteration {iteration+1}/{NUM_ITERS}: Training on {len(all_episodes)} episodes "
              f"(padded to {max_seq_len})")
        rng, train_state = train_autoencoder(
            rng, train_state, train_step_fn, padded_episodes, masks
        )

    print("\nDone! Autoencoder trained.")

    # ── Collect IPPO selfplay trajectories ──
    print("\nCollecting IPPO selfplay trajectories...")
    rng, ippo_episodes = collect_ippo_selfplay_trajectories(rng, env, num_rollouts=ROLLOUTS_PER_ITER)
    print(f"  Collected {len(ippo_episodes)} IPPO episodes")

    # ── Part 3: Encode and plot t-SNE ──
    print("\nEncoding trajectories for t-SNE...")
    random_latents = encode_episodes(model, train_state, all_episodes, max_seq_len)
    ippo_latents = encode_episodes(model, train_state, ippo_episodes, max_seq_len)
    plot_tsne(
        {"random": random_latents, "ippo_selfplay": ippo_latents},
        save_path="evaluation/tsne_random_vs_ippo.png",
    )
