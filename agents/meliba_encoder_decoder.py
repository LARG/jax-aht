import functools

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import distrax

from agents.rnn_actor_critic import ScannedRNN
from marl.meliba_utils import DecoderScannedRNN, transform_timestep_to_batch_vmap, shift_padding_to_front_vectorized

def sample_gaussian(mu, logvar, prng_key):
    std = jnp.exp(0.5 * logvar)
    eps = jax.random.normal(prng_key, std.shape)
    return (eps * std) + mu

class FeatureExtractor(nn.Module):
    """ Used for extrating features for states/actions/rewards """
    output_size: int
    activation_function: callable = None

    @nn.compact
    def __call__(self, inputs):
        if self.output_size != 0:
            features = nn.Dense(self.output_size)(inputs)
            if self.activation_function is not None:
                features = self.activation_function(features)
            return features
        else:
            return jnp.zeros(0, )

class VariationalEncoderRNNNetwork(nn.Module):
    state_embed_dim: int
    action_embed_dim: int
    reward_embed_dim: int
    layers_before_rnn: int
    layers_after_rnn: int
    latent_dim: int

    @nn.compact
    def __call__(self, hidden, x):
        states, actions, rewards, dones, prng_key = x

        action_embed = FeatureExtractor(self.action_embed_dim, nn.relu)(actions)
        state_embed = FeatureExtractor(self.state_embed_dim, nn.relu)(states)
        reward_embed = FeatureExtractor(self.reward_embed_dim, nn.relu)(rewards)

        embedding = jnp.concatenate((action_embed, state_embed, reward_embed), axis=-1)

        # def n_dense(x, hidden_dim):
        #     x = nn.Dense(hidden_dim)(x)
        #     x = nn.relu(x)
        #     return x

        # embedding = jax.lax.scan(n_dense, embedding, self.layers_before_rnn)
        embedding = nn.Dense(self.layers_before_rnn)(embedding)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        # embedding = jax.lax.scan(n_dense, embedding, self.layers_after_rnn)
        embedding = nn.Dense(self.layers_after_rnn)(embedding)
        embedding = nn.relu(embedding)

        prng_key, agent_character_key, mental_state_key = jax.random.split(prng_key, 3)

        latent_mean = nn.Dense(self.latent_dim)(embedding)
        latent_logvar = nn.Dense(self.latent_dim)(embedding)
        latent_sample = sample_gaussian(latent_mean, latent_logvar, agent_character_key)

        latent_mean_t = nn.Dense(self.latent_dim)(embedding)
        latent_logvar_t = nn.Dense(self.latent_dim)(embedding)
        latent_sample_t = sample_gaussian(latent_mean_t, latent_logvar_t, mental_state_key)

        return hidden, (latent_sample, latent_mean, latent_logvar, latent_sample_t, latent_mean_t, latent_logvar_t)

class DecoderRNNNetwork(nn.Module):
    state_embed_dim: int
    agent_character_embed_dim: int
    hidden_dim: int
    ouput_dim: int

    @nn.compact
    def __call__(self, x):
        state, latent_mean, latent_logvar, latent_mean_t, latent_logvar_t, agent_character, mental_state, partner_actions, dones = x
        # Sizes are (time, batch, dim), except partner_actions and ones which are (time, batch)

        # Compute KL divergence
        latent_mean_all = jnp.concatenate((latent_mean, latent_mean_t), axis=-1)
        latent_logvar_all = jnp.concatenate((latent_logvar, latent_logvar_t), axis=-1)

        gauss_dim = latent_mean_all.shape[-1]
        # add the gaussian prior
        all_means = jnp.concatenate((jnp.zeros((1, *latent_mean_all.shape[1:])), latent_mean_all))
        all_logvars = jnp.concatenate((jnp.zeros((1, *latent_logvar_all.shape[1:])), latent_logvar_all))
        # https://arxiv.org/pdf/1811.09975.pdf
        # KL(N(mu,E)||N(m,S)) = 0.5 * (log(|S|/|E|) - K + tr(S^-1 E) + (m-mu)^T S^-1 (m-mu)))
        mu = all_means[1:]
        m = all_means[:-1]
        logE = all_logvars[1:]
        logS = all_logvars[:-1]
        kl_loss = 0.5 * (jnp.sum(logS, axis=-1) - jnp.sum(logE, axis=-1) - gauss_dim + jnp.sum(
            1 / jnp.exp(logS) * jnp.exp(logE), axis=-1) + ((m - mu) / jnp.exp(logS) * (m - mu)).sum(axis=-1))
        kl_loss = kl_loss.sum(axis=0)
        kl_loss = kl_loss.sum(axis=0).mean()

        # MeLIBA decoder
        state_embed = FeatureExtractor(self.state_embed_dim, nn.relu)(state)
        agent_character_embed = FeatureExtractor(self.agent_character_embed_dim, nn.relu)(agent_character)

        state_agent_embed = jnp.concatenate((state_embed, agent_character_embed), axis=-1)
        state_agent_embed = nn.Dense(self.hidden_dim)(state_agent_embed)

        hidden = jnp.concatenate((agent_character_embed, mental_state), axis=-1)
        hidden = nn.Dense(self.hidden_dim)(hidden)

        # jax.debug.breakpoint()

        # The batch dimension is the second dimension, we want to vmap over that.
        # The batch referes to the different env instances.
        def handle_batch(state_agent_embed, hidden, dones):

            # Construct k trajectories
            k_state_agent_embed, valid_mask = transform_timestep_to_batch_vmap(state_agent_embed, pad_value=0.0, return_mask=True)
            k_hidden = transform_timestep_to_batch_vmap(hidden, pad_value=0.0, return_mask=False)
            k_dones = transform_timestep_to_batch_vmap(dones, pad_value=0.0, return_mask=False)

            # state_agent_embed expected as 3D for RNN (128, 64) -> (128, 1, 64)
            # hidden expected as 2D for RNN (128, 64) -> hidden[0] (1, 64)
            # dones (128, 1)
            def handle_k_trajectories(state_agent_embed, hidden, dones):
                rnn_in = (jnp.expand_dims(state_agent_embed, axis=1), hidden, dones)
                _, embedding = DecoderScannedRNN()(jnp.expand_dims(hidden[0], axis=0), rnn_in)

                # Squeeze the batch dimension
                # Shape: (128, 1, 32) -> (128, 32)
                out = nn.Dense(self.ouput_dim)(embedding)
                out = jnp.squeeze(out, axis=1)

                return out

            # Shape (127, 128, 32)
            vmap_handle_k_trajectories = jax.vmap(handle_k_trajectories, (0, 0, 0), 0)
            out = vmap_handle_k_trajectories(k_state_agent_embed, k_hidden, k_dones)

            # Mask out to only consider valid elements
            out = out * jnp.expand_dims(valid_mask, axis=-1)
            out, _ = shift_padding_to_front_vectorized(out, valid_mask)

            # Reduction: Sum over k trajectories
            # Shape (128, 32)
            out = jnp.sum(out, axis=0)

            return out

            # Reduction

        # Shape (128, 2, 32)
        vmap_handle_batch = jax.vmap(handle_batch, (1, 1, 1), 1)
        out = vmap_handle_batch(state_agent_embed, hidden, jnp.expand_dims(dones, axis=-1))

        # Log likelihood
        pi = distrax.Categorical(logits=out)
        # Shape (128, 2)
        log_prob = pi.log_prob(partner_actions)
        # Shape (128,)
        log_prob_sum = jnp.sum(log_prob, axis=-1)

        return kl_loss, log_prob_sum

class VariationalEncoderRNN():
    """Model wrapper for EncoderRNNNetwork."""

    def __init__(self, state_dim, action_dim, state_embed_dim, action_embed_dim, reward_embed_dim,
                 rnn_hidden_dim, layers_before_rnn, layers_after_rnn, latent_dim):
        """
        Args:
            state_dim: int, dimension of the state space
            action_dim: int, dimension of the action space
            state_embed_dim: int, dimension of the encoder state embedding
            action_embed_dim: int, dimension of the encoder action embedding
            reward_embed_dim: int, dimension of the encoder reward embedding
            rnn_hidden_dim: int, dimension of the RNN hidden layers
            layers_before_rnn: jnp.array, dimensions of the layers before LSTM
            layers_after_rnn: jnp.array, dimensions of the layers after LSTM
            latent_dim: int, dimension of the latent space
        """
        self.model = VariationalEncoderRNNNetwork(state_embed_dim, action_embed_dim, reward_embed_dim,
                                                  layers_before_rnn, layers_after_rnn, latent_dim)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.rnn_hidden_dim = rnn_hidden_dim

    def init_hstate(self, batch_size=1, aux_info=None):
        """Initialize hidden state for the encoder RNN."""
        hstate =  ScannedRNN.initialize_carry(batch_size, self.rnn_hidden_dim)
        hstate = hstate.reshape(1, batch_size, self.rnn_hidden_dim)
        return hstate

    def init_params(self, prng):
        """Initialize parameters for the encoder model."""
        batch_size = 1

        # Initialize hidden state
        init_hstate = self.init_hstate(batch_size=batch_size)

        # Split the random key for sampling
        prng, init_key, sample_key = jax.random.split(prng, 3)

        # Create dummy inputs - add time dimension
        dummy_state = jnp.zeros((1, batch_size, self.state_dim))
        dummy_act = jnp.zeros((1, batch_size, self.action_dim))
        dummy_reward = jnp.zeros((1, batch_size, 1))
        dummy_done = jnp.zeros((1, batch_size))

        dummy_x = (dummy_state, dummy_act, dummy_reward, dummy_done, sample_key)

        init_hstate = init_hstate.reshape(batch_size, -1)

        # Initialize model
        return self.model.init(init_key, init_hstate, dummy_x)

    @functools.partial(jax.jit, static_argnums=(0,))
    def compute_embedding(self, params, hstate, state, act, reward, done, sample_key):
        """Embed observations using the encoder model."""
        batch_size = state.shape[1]

        new_hstate, (latent_sample, latent_mean, latent_logvar, latent_sample_t, latent_mean_t, latent_logvar_t) = self.model.apply(
            params, hstate.squeeze(0), (state, act, reward, done, sample_key))

        return latent_sample, latent_mean, latent_logvar, latent_sample_t, latent_mean_t, latent_logvar_t, new_hstate.reshape(1, batch_size, -1)

class Decoder():
    """Model wrapper for DecoderNetwork."""

    def __init__(self, state_dim, state_embed_dim, agent_character_embed_dim, latent_mean_dim, latent_logvar_dim,
                 latent_mean_t_dim, latent_logvar_t_dim, agent_character_dim, mental_state_dim, partner_action_dim,
                 hidden_dim, ouput_dim, loss_coeff, kl_weight):
        """
        Args:
            obs_dim: int, dimension of the observation space
            action_dim: int, dimension of the action space
            obs_dim: int, dimension of the observation space
            embedding_dim: int, dimension of the embedding space
            hidden_dim: int, dimension of the decoder hidden layers
            ouput_dim1: int, dimension of the decoder output
            ouput_dim2: int, dimension of the decoder probs
        """
        self.model = DecoderRNNNetwork(state_embed_dim, agent_character_embed_dim, hidden_dim, ouput_dim)

        self.state_dim = state_dim
        self.state_embed_dim = state_embed_dim
        self.agent_character_embed_dim = agent_character_embed_dim
        self.latent_mean_dim = latent_mean_dim
        self.latent_logvar_dim = latent_logvar_dim
        self.latent_mean_t_dim = latent_mean_t_dim
        self.latent_logvar_t_dim = latent_logvar_t_dim
        self.agent_character_dim = agent_character_dim
        self.mental_state_dim = mental_state_dim
        self.partner_action_dim = partner_action_dim
        self.hidden_dim = hidden_dim
        self.ouput_dim = ouput_dim
        self.loss_coeff = loss_coeff
        self.kl_weight = kl_weight

    def init_params(self, prng):
        """Initialize parameters for the decoder model."""
        batch_size = 1

        # state, latent_mean, latent_logvar, latent_mean_t, latent_logvar_t,
        # agent_character, mental_state, partner_actions, dones, prng_key = x

        # Split the random key for sampling
        # prng, init_key, sample_key = jax.random.split(rng, 3)

        # Create dummy inputs - add time dimension
        # TODO: Addtional dimensions needed to handle to trajectory of trajectories
        dummy_state = jnp.zeros((1, batch_size, self.state_dim))
        dummy_latent_mean = jnp.zeros((1, batch_size, self.latent_mean_dim))
        dummy_latent_logvar = jnp.zeros((1, batch_size, self.latent_logvar_dim))
        dummy_latent_mean_t = jnp.zeros((1, batch_size, self.latent_mean_t_dim))
        dummy_latent_logvar_t = jnp.zeros((1, batch_size, self.latent_logvar_t_dim))
        dummy_agent_character = jnp.zeros((1, batch_size, self.agent_character_dim))
        dummy_mental_state = jnp.zeros((1, batch_size, self.mental_state_dim))
        dummy_partner_actions = jnp.zeros((1, batch_size, self.ouput_dim))
        dummy_done = jnp.zeros((1, batch_size))

        # TODO: Should be the state instead of obs
        dummy_x = (dummy_state, dummy_latent_mean, dummy_latent_logvar, dummy_latent_mean_t,
                   dummy_latent_logvar_t, dummy_agent_character, dummy_mental_state,
                   dummy_partner_actions, dummy_done)

        # Initialize model
        return self.model.init(prng, dummy_x)

    @functools.partial(jax.jit, static_argnums=(0,))
    def evaluate(self, params, state, latent_mean, latent_logvar, latent_mean_t, latent_logvar_t,
                 agent_character, mental_state, partner_action, done):

        """Evaluate the decoder model with given parameters and inputs."""
        kl_loss, log_prob_pred = self.model.apply(params, (state, latent_mean, latent_logvar,
                                                           latent_mean_t, latent_logvar_t,
                                                           agent_character, mental_state,
                                                           partner_action, done))

        # TODO: Check sign of KL loss
        elbo = (self.loss_coeff * log_prob_pred) - (self.kl_weight * kl_loss)

        return log_prob_pred, kl_loss, elbo.mean()

def initialize_encoder_decoder(config, env, rng):
    """Initialize the Encoder and Decoder models with the given config.

    Args:
        config: dict, config for the agent
        env: gymnasium environment
        rng: jax.random.PRNGKey, random key for initialization

    Returns:
        encoder: Encoder, the model object
        decoder: Decoder, the model object
        params: dict, initial parameters for the encoder and decoder
    """
    # TODO: Should be the state instead of obs
    encoder = VariationalEncoderRNN(
        state_dim=env.observation_space(env.agents[0]).shape[0],
        action_dim=env.action_space(env.agents[0]).n + env.action_space(env.agents[1]).n,
        state_embed_dim=config.get("ENCODER_STATE_EMBED_DIM", 64),
        action_embed_dim=config.get("ENCODER_ACTION_EMBED_DIM", 64),
        reward_embed_dim=config.get("ENCODER_REWARD_EMBED_DIM", 64),
        rnn_hidden_dim=config.get("ENCODER_RNN_HIDDEN_DIM", 64),
        layers_before_rnn=config.get("ENCODER_LAYERS_BEFORE_RNN", 64),
        layers_after_rnn=config.get("ENCODER_LAYERS_AFTER_RNN", 64),
        latent_dim=config.get("ENCODER_LATENT_DIM", 64)
    )

    # TODO: Should be the state instead of obs
    decoder = Decoder(
        state_dim=env.observation_space(env.agents[0]).shape[0],
        state_embed_dim=config.get("DECODER_STATE_EMBED_DIM", 64),
        agent_character_embed_dim=config.get("DECODER_AGENT_CHARACTER_EMBED_DIM", 32),
        latent_mean_dim=config.get("ENCODER_LATENT_DIM", 64),
        latent_logvar_dim=config.get("ENCODER_LATENT_DIM", 64),
        latent_mean_t_dim=config.get("ENCODER_LATENT_DIM", 64),
        latent_logvar_t_dim=config.get("ENCODER_LATENT_DIM", 64),
        agent_character_dim=config.get("ENCODER_LATENT_DIM", 64),
        mental_state_dim=config.get("ENCODER_LATENT_DIM", 64),
        partner_action_dim=env.action_space(env.agents[1]).n,
        hidden_dim=config.get("DECODER_HIDDEN_DIM", 64),
        ouput_dim=config.get("DECODER_OUTPUT_DIM", 64),
        loss_coeff=config.get("DECODER_LOSS_COEFF", 1.0),
        kl_weight=config.get("DECODER_KL_WEIGHT", 0.05)
    )

    rng, init_rng_encoder, init_rng_decoder  = jax.random.split(rng, 3)
    init_params_encoder = encoder.init_params(init_rng_encoder)
    init_params_decoder = decoder.init_params(init_rng_decoder)

    return encoder, decoder, {'encoder': init_params_encoder, 'decoder': init_params_decoder}
