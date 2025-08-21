import functools

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import distrax

from agents.liam_encoder_decoder import ScannedLSTM
from agents.rnn_actor_critic import ScannedRNN
from agents.s5_actor_critic import SequenceLayer

def sample_gaussian(mu, logvar, prng_key):
    std = jnp.exp(0.5 * logvar)
    eps = jax.random.normal(prng_key, std.shape)
    return eps.mul(std).add_(mu)

class FeatureExtractor(nn.Module):
    """ Used for extrating features for states/actions/rewards """
    output_size: int
    activation_function: callable = None

    def __call__(self, inputs):
        if self.output_size != 0:
            features = nn.Dense(self.output_size)(inputs)
            if self.activation_function is not None:
                features = self.activation_function(features)
            return features
        else:
            return jnp.zeros(0, )

class VariationalEncoderLSTMNetwork(nn.Module):
    state_embed_dim: int
    action_embed_dim: int
    reward_embed_dim: int
    layers_before_lstm: jnp.array
    layers_after_lstm: jnp.array
    latent_dim: int

    @nn.compact
    def __call__(self, hidden, x):
        states, actions, rewards, dones, prng_key = x

        action_embed = FeatureExtractor(self.action_embed_dim, nn.relu)(actions)
        state_embed = FeatureExtractor(self.state_embed_dim, nn.relu)(states)
        reward_embed = FeatureExtractor(self.reward_embed_dim, nn.relu)(rewards)
        embedding = jnp.cat((action_embed, state_embed, reward_embed), dim=2)

        def n_dense(x, hidden_dim):
            x = nn.Dense(hidden_dim)(x)
            x = nn.relu(x)
            return x

        embedding = jax.lax.scan(n_dense, embedding, self.layers_before_lstm)

        lstm_in = (embedding, dones)
        hidden, embedding = ScannedLSTM()(hidden, lstm_in)

        embedding = jax.lax.scan(n_dense, embedding, self.layers_after_lstm)

        prng_key, agent_character_key, mental_state_key = jax.random.split(prng_key, 3)

        latent_mean = nn.Dense(self.latent_dim)(embedding)
        latent_logvar = nn.Dense(self.latent_dim)(embedding)
        latent_sample = sample_gaussian(latent_mean, latent_logvar, agent_character_key)

        latent_mean_t = nn.Dense(self.latent_dim)(embedding)
        latent_logvar_t = nn.Dense(self.latent_dim)(embedding)
        latent_sample_t = sample_gaussian(latent_mean_t, latent_logvar_t, mental_state_key)

        return hidden, (latent_sample, latent_mean, latent_logvar, latent_sample_t, latent_mean_t, latent_logvar_t)

class VariationalEncoderRNNNetwork(nn.Module):
    state_embed_dim: int
    action_embed_dim: int
    reward_embed_dim: int
    layers_before_lstm: jnp.array
    layers_after_lstm: jnp.array
    latent_dim: int

    @nn.compact
    def __call__(self, hidden, x):
        states, actions, rewards, dones, prng_key = x

        action_embed = FeatureExtractor(self.action_embed_dim, nn.relu)(actions)
        state_embed = FeatureExtractor(self.state_embed_dim, nn.relu)(states)
        reward_embed = FeatureExtractor(self.reward_embed_dim, nn.relu)(rewards)
        embedding = jnp.cat((action_embed, state_embed, reward_embed), dim=2)

        def n_dense(x, hidden_dim):
            x = nn.Dense(hidden_dim)(x)
            x = nn.relu(x)
            return x

        embedding = jax.lax.scan(n_dense, embedding, self.layers_before_lstm)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        embedding = jax.lax.scan(n_dense, embedding, self.layers_after_lstm)

        prng_key, agent_character_key, mental_state_key = jax.random.split(prng_key, 3)

        latent_mean = nn.Dense(self.latent_dim)(embedding)
        latent_logvar = nn.Dense(self.latent_dim)(embedding)
        latent_sample = sample_gaussian(latent_mean, latent_logvar, agent_character_key)

        latent_mean_t = nn.Dense(self.latent_dim)(embedding)
        latent_logvar_t = nn.Dense(self.latent_dim)(embedding)
        latent_sample_t = sample_gaussian(latent_mean_t, latent_logvar_t, mental_state_key)

        return hidden, (latent_sample, latent_mean, latent_logvar, latent_sample_t, latent_mean_t, latent_logvar_t)

class VariationalEncoderS5Network(nn.Module):
    state_embed_dim: int
    action_embed_dim: int
    reward_embed_dim: int
    layers_before_lstm: jnp.array
    layers_after_lstm: jnp.array
    latent_dim: int

    @nn.compact
    def __call__(self, hidden, x):
        raise NotImplementedError

class DecoderRNNNetwork(nn.Module):
    state_embed_dim: int
    agent_character_embed_dim: int
    hidden_dim: int
    ouput_dim: int

    @nn.compact
    def __call__(self, x):
        state, latent_mean, latent_logvar, latent_mean_t, latent_logvar_t, agent_character, mental_state, partner_actions, dones, prng_key = x

        # Compute KL divergence
        latent_mean = jnp.concatenate((latent_mean, latent_mean_t), axis=-1)
        latent_logvar = jnp.concatenate((latent_logvar, latent_logvar_t), axis=-1)

        gauss_dim = latent_mean.shape[-1]
        # add the gaussian prior
        all_means = jnp.concatenate((jnp.zeros(1, *latent_mean.shape[1:]), latent_mean))
        all_logvars = jnp.concatenate((jnp.zeros(1, *latent_logvar.shape[1:]), latent_logvar))
        # https://arxiv.org/pdf/1811.09975.pdf
        # KL(N(mu,E)||N(m,S)) = 0.5 * (log(|S|/|E|) - K + tr(S^-1 E) + (m-mu)^T S^-1 (m-mu)))
        mu = all_means[1:]
        m = all_means[:-1]
        logE = all_logvars[1:]
        logS = all_logvars[:-1]
        kl_loss = 0.5 * (jnp.sum(logS, dim=-1) - jnp.sum(logE, dim=-1) - gauss_dim + jnp.sum(
            1 / jnp.exp(logS) * jnp.exp(logE), dim=-1) + ((m - mu) / jnp.exp(logS) * (m - mu)).sum(dim=-1))
        kl_loss = kl_loss.sum(dim=0)
        kl_loss = kl_loss.sum(dim=0).mean()

        # MeLIBA decoder
        state_embed = FeatureExtractor(self.state_embed_dim, nn.relu)(state)
        agent_character_embed = FeatureExtractor(self.agent_character_embed_dim, nn.relu)(agent_character)

        out = jnp.concatenate((state_embed, agent_character_embed), axis=-1)
        out = nn.Dense(self.hidden_dim)(out)

        hidden = jnp.concatenate((agent_character_embed, mental_state), axis=-1)
        hidden = nn.Dense(self.hidden_dim)(hidden)

        #TODO: Check vmap, scan, or fori

        rnn_in = (out, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        out = nn.Dense(self.ouput_dim)(embedding)

        # Sum product
        # TODO: check if this is correct
        pi = distrax.Categorical(logits=out)

        # Log likelihood
        #TODO: Input the partner actions
        log_prob = pi.log_prob(partner_actions)
        log_prob = jnp.sum(log_prob, axis=-1)

        return kl_loss, log_prob

class VariationalEncoderLSTM():
    """Model wrapper for EncoderLSTMNetwork."""

    def __init__(self, state_dim, action_dim, reward_dim, state_embed_dim, action_embed_dim, reward_embed_dim, lstm_hidden_dim, layers_before_lstm, layers_after_lstm, latent_dim):
        """
        Args:
            state_dim: int, dimension of the state space
            action_dim: int, dimension of the action space
            reward_dim: int, dimension of the reaward space
            state_embed_dim: int, dimension of the encoder state embedding
            action_embed_dim: int, dimension of the encoder action embedding
            reward_embed_dim: int, dimension of the encoder reward embedding
            lstm_hidden_dim: int, dimension of the LSTM hidden layers
            layers_before_lstm: jnp.array, dimensions of the layers before LSTM
            layers_after_lstm: jnp.array, dimensions of the layers after LSTM
            latent_dim: int, dimension of the latent space
        """
        self.model = VariationalEncoderLSTMNetwork(state_embed_dim, action_embed_dim, reward_embed_dim, layers_before_lstm, layers_after_lstm, latent_dim)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.lstm_hidden_dim = lstm_hidden_dim

    def init_hstate(self, batch_size=1, aux_info=None):
        """Initialize hidden state for the encoder LSTM."""
        hstate =  ScannedLSTM.initialize_carry(batch_size, self.lstm_hidden_dim)
        hstate = (hstate[0].reshape(1, batch_size, self.lstm_hidden_dim),
                  hstate[1].reshape(1, batch_size, self.lstm_hidden_dim))
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
        dummy_reward = jnp.zeros((1, batch_size, self.reward_dim))
        dummy_done = jnp.zeros((1, batch_size))

        dummy_x = (dummy_state, dummy_act, dummy_reward, dummy_done, sample_key)

        init_hstate = (init_hstate[0].reshape(batch_size, -1),
                       init_hstate[1].reshape(batch_size, -1))

        # Initialize model
        return self.model.init(init_key, init_hstate, dummy_x)

    @functools.partial(jax.jit, static_argnums=(0,))
    def compute_embedding(self, params, hstate, state, act, reward, done, sample_key):
        """Embed observations using the encoder model."""
        batch_size = state.shape[1]

        hstate = (hstate[0].squeeze(0),
                  hstate[1].squeeze(0))

        new_hstate, (latent_sample, latent_mean, latent_logvar, latent_sample_t, latent_mean_t, latent_logvar_t) = self.model.apply(
            params, hstate, (state, act, reward, done, sample_key))

        new_hstate = (new_hstate[0].reshape(1, batch_size, -1),
                      new_hstate[1].reshape(1, batch_size, -1))
        return latent_sample, latent_mean, latent_logvar, latent_sample_t, latent_mean_t, latent_logvar_t, new_hstate

class VariationalEncoderRNN():
    """Model wrapper for EncoderRNNNetwork."""

    def __init__(self, state_dim, action_dim, reward_dim, state_embed_dim, action_embed_dim, reward_embed_dim, rnn_hidden_dim, layers_before_lstm, layers_after_lstm, latent_dim):
        """
        Args:
            state_dim: int, dimension of the state space
            action_dim: int, dimension of the action space
            reward_dim: int, dimension of the reaward space
            state_embed_dim: int, dimension of the encoder state embedding
            action_embed_dim: int, dimension of the encoder action embedding
            reward_embed_dim: int, dimension of the encoder reward embedding
            rnn_hidden_dim: int, dimension of the RNN hidden layers
            layers_before_lstm: jnp.array, dimensions of the layers before LSTM
            layers_after_lstm: jnp.array, dimensions of the layers after LSTM
            latent_dim: int, dimension of the latent space
        """
        self.model = VariationalEncoderRNNNetwork(state_embed_dim, action_embed_dim, reward_embed_dim, layers_before_lstm, layers_after_lstm, latent_dim)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim
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
        dummy_reward = jnp.zeros((1, batch_size, self.reward_dim))
        dummy_done = jnp.zeros((1, batch_size))

        dummy_x = (dummy_state, dummy_act, dummy_reward, dummy_done, sample_key)

        init_hstate = (init_hstate[0].reshape(batch_size, -1),
                       init_hstate[1].reshape(batch_size, -1))

        # Initialize model
        return self.model.init(init_key, init_hstate, dummy_x)

    @functools.partial(jax.jit, static_argnums=(0,))
    def compute_embedding(self, params, hstate, state, act, reward, done, sample_key):
        """Embed observations using the encoder model."""
        batch_size = state.shape[1]

        new_hstate, (latent_sample, latent_mean, latent_logvar, latent_sample_t, latent_mean_t, latent_logvar_t) = self.model.apply(
            params, hstate, (state, act, reward, done, sample_key))

        return latent_sample, latent_mean, latent_logvar, latent_sample_t, latent_mean_t, latent_logvar_t, new_hstate.reshape(1, batch_size, -1)

# TODO: Complete S5 encoder
class VariationalEncoderS5():
    """Model wrapper for EncoderS5Network."""

    def __init__(self, action_dim, obs_dim, hidden_dim, ouput_dim):
        """
        Args:
            action_dim: int, dimension of the action space
            obs_dim: int, dimension of the observation space
            hidden_dim: int, dimension of the encoder hidden layers
            ouput_dim: int, dimension of the encoder output
        """
        self.model = VariationalEncoderS5Network(hidden_dim, ouput_dim)
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.s5_hidden_dim = hidden_dim

    def init_hstate(self, batch_size=1, aux_info=None):
        """Initialize hidden state for the encoder S5."""
        hstate =  SequenceLayer.initialize_carry(batch_size, self.s5_hidden_dim)
        return hstate

    def init_params(self, rng):
        """Initialize parameters for the encoder model."""
        batch_size = 1

        # Initialize hidden state
        init_hstate = self.init_hstate(batch_size=batch_size)

        # Split the random key for sampling
        prng, init_key, sample_key = jax.random.split(prng, 3)

        # Create dummy inputs - add time dimension
        dummy_state = jnp.zeros((1, batch_size, self.state_dim))
        dummy_act = jnp.zeros((1, batch_size, self.action_dim))
        dummy_reward = jnp.zeros((1, batch_size, self.reward_dim))
        dummy_done = jnp.zeros((1, batch_size))

        dummy_x = (dummy_state, dummy_act, dummy_reward, dummy_done, sample_key)

        # Initialize model
        return self.model.init(init_key, init_hstate, dummy_x)

    @functools.partial(jax.jit, static_argnums=(0,))
    def compute_embedding(self, params, hstate, state, act, reward, done, sample_key):
        """Embed observations using the encoder model."""
        return self.model.apply(params, hstate, (state, act, reward, done, sample_key))

class Decoder():
    """Model wrapper for DecoderNetwork."""

    def __init__(self, state_dim, state_embed_dim, agent_character_embed_dim, latent_mean_dim, latent_logvar_dim,
                 latent_mean_t_dim, latent_logvar_t_dim, agent_character_dim, mental_state_dim, hidden_dim, ouput_dim,
                 loss_coeff, kl_weight):
        """
        Args:
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
        self.hidden_dim = hidden_dim
        self.ouput_dim = ouput_dim
        self.loss_coeff = loss_coeff
        self.kl_weight = kl_weight

    def init_params(self, rng):
        """Initialize parameters for the decoder model."""
        batch_size = 1

        # Create dummy inputs - add time dimension
        # TODO: Addtional dimensions needed to handle to trajectory of trajectories
        dummy_state = jnp.zeros((1, batch_size, self.state_dim))
        dummy_latent_mean = jnp.zeros((1, batch_size, self.latent_mean_dim))
        dummy_latent_logvar = jnp.zeros((1, batch_size, self.latent_logvar_dim))
        dummy_latent_mean_t = jnp.zeros((1, batch_size, self.latent_mean_t_dim))
        dummy_latent_logvar_t = jnp.zeros((1, batch_size, self.latent_logvar_t_dim))
        dummy_agent_character = jnp.zeros((1, batch_size, self.agent_character_dim))
        dummy_mental_state = jnp.zeros((1, batch_size, self.mental_state_dim))
        dummy_done = jnp.zeros((1, batch_size))

        dummy_x = (dummy_state, dummy_latent_mean, dummy_latent_logvar, dummy_latent_mean_t,
                   dummy_latent_logvar_t, dummy_agent_character, dummy_mental_state, dummy_done)

        # Initialize model
        return self.model.init(rng, dummy_x)

    @functools.partial(jax.jit, static_argnums=(0,))
    def evaluate(self, params, sample):
        """Evaluate the decoder model with given parameters and inputs."""
        kl_loss, log_prob_pred = self.model.apply(params, embedding)

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
    # Create the RNN policy
    encoder_type = config.get("ENCODER_TYPE", "rnn")
    if encoder_type == "lstm":
        encoder = VariationalEncoderLSTM(
            action_dim=env.action_space(env.agents[0]).n,
            obs_dim=env.observation_space(env.agents[0]).shape[0],
            hidden_dim=config.get("ENCODER_HIDDEN_DIM", 64),
            ouput_dim=config.get("ENCODER_OUTPUT_DIM", 64)
        )
    elif encoder_type == "rnn":
        encoder = VariationalEncoderRNN(
            action_dim=env.action_space(env.agents[0]).n,
            obs_dim=env.observation_space(env.agents[0]).shape[0],
            hidden_dim=config.get("ENCODER_HIDDEN_DIM", 64),
            ouput_dim=config.get("ENCODER_OUTPUT_DIM", 64)
        )

    decoder = Decoder(
        action_dim=env.action_space(env.agents[0]).n,
        obs_dim=env.observation_space(env.agents[0]).shape[0],
        embedding_dim=config.get("ENCODER_OUTPUT_DIM", 64),
        hidden_dim=config.get("DECODER_HIDDEN_DIM", 64),
        ouput_dim1=env.observation_space(env.agents[1]).shape[0],
        ouput_dim2=env.action_space(env.agents[1]).n
    )

    rng, init_rng_encoder, init_rng_decoder  = jax.random.split(rng, 3)
    init_params_encoder = encoder.init_params(init_rng_encoder)
    init_params_decoder = decoder.init_params(init_rng_decoder)

    return encoder, decoder, {'encoder': init_params_encoder, 'decoder': init_params_decoder}
