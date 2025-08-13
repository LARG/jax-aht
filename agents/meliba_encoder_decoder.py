import functools

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import optax

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

class StateTransitionDecoderNetwork(nn.Module):
    state_dim: int
    state_embed_dim: int
    action_embed_dim: int
    layers: jnp.array
    pred_type: str = 'deterministic'

    @nn.compact
    def __call__(self, x):
        latent_state, states, actions = x

        state_embed = FeatureExtractor(self.state_embed_dim, nn.relu)(states)
        action_embed = FeatureExtractor(self.action_embed_dim, nn.relu)(actions)

        def n_dense(x, hidden_dim):
            x = nn.Dense(hidden_dim)(x)
            x = nn.relu(x)
            return x

        embedding = jax.lax.scan(n_dense, jnp.concatenate((latent_state, state_embed, action_embed), axis=-1), self.layers)

        if self.pred_type == 'gaussian':
            prediction = nn.Dense(2 * self.state_dim)(embedding)
        else:
            prediction = nn.Dense(self.state_dim)(embedding)

        return prediction

class RewardDecoderNetwork(nn.Module):
    state_dim: int
    state_embed_dim: int
    action_embed_dim: int
    num_states: int
    layers: jnp.array
    pred_type: str = 'deterministic'
    input_prev_state: bool = False
    input_action: bool = False

    @nn.compact
    def __call__(self, x):
        latent_state, next_state, prev_state, actions = x

        state_encoder = FeatureExtractor(self.state_embed_dim, nn.relu)

        state_embed = state_encoder(next_state)
        h = jnp.concatenate((latent_state, state_embed), axis=-1)

        if self.input_action:
            action_embed = FeatureExtractor(self.action_embed_dim, nn.relu)(actions)
            h = jnp.concatenate((h, action_embed), axis=-1)

        if self.input_prev_state:
            prev_state_embed = state_encoder(prev_state)
            h = jnp.concatenate((h, prev_state_embed), axis=-1)

        def n_dense(x, hidden_dim):
            x = nn.Dense(hidden_dim)(x)
            x = nn.relu(x)
            return x

        embedding = jax.lax.scan(n_dense, h, self.layers)

        if self.pred_type == 'gaussian':
            prediction = nn.Dense(2)(embedding)
        else:
            prediction = nn.Dense(1)(embedding)

        return prediction

class DecoderRNNNetwork(nn.Module):
    state_embed_dim: int
    agent_character_embed_dim: int
    hidden_dim: int
    ouput_dim: int

    # Parameters for state transition decoder
    state_dim: int
    action_embed_dim: int
    state_layers: jnp.array
    state_pred_type: str = 'deterministic'

    # Parameters for reward decoder
    reward_layers: jnp.array
    reward_pred_type: str = 'bernoulli'
    input_prev_state: bool = False
    input_action: bool = False

    @nn.compact
    def __call__(self, x):
        state, prev_state, actions, reward, latent_sample, latent_mean, latent_logvar, agent_character, mental_state, dones = x
        state_decode_input = (latent_state, state, actions)

        # State decoder reconstruction loss
        state_pred = StateTransitionDecoderNetwork(self.state_dim, self.state_embed_dim, self.action_embed_dim,
                                                     self.state_layers, self.state_pred_type)((latent_sample, prev_state, actions))

        if self.state_pred_type == 'deterministic':
            loss_state = jnp.mean(jnp.power(state_pred - state, 2), dim=-1)
        else:
            raise NotImplementedError

        # Reward decoder reconstruction loss
        reward_pred = RewardDecoderNetwork(self.state_dim, self.state_embed_dim, self.action_embed_dim,
                                             self.num_states, self.reward_layers, self.reward_pred_type,
                                             self.input_prev_state, self.input_action)((latent_sample, state, prev_state, actions))

        if self.reward_pred_type == 'bernoulli':
            rew_pred = jax.nn.sigmoid(rew_pred)
            rew_target = (reward == 1).astype(jnp.float32)
            loss_rew = jnp.mean(optax.sigmoid_binary_cross_entropy(rew_pred, rew_target), dim=-1)
        elif self.reward_pred_type == 'deterministic':
            loss_rew = jnp.mean(jnp.power(rew_pred - reward, 2), dim=-1)
        else:
            raise NotImplementedError

        # TODO: Implement KL loss
        state_embed = FeatureExtractor(self.state_embed_dim, nn.relu)(state)
        agent_character_embed = FeatureExtractor(self.agent_character_embed_dim, nn.relu)(agent_character)

        out = jnp.concatenate((state_embed, agent_character_embed), axis=-1)
        out = nn.Dense(self.hidden_dim)(out)

        hidden = jnp.concatenate((agent_character_embed, mental_state), axis=-1)
        hidden = nn.Dense(self.hidden_dim)(hidden)

        rnn_in = (out, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        prediction = nn.Dense(self.ouput_dim)(embedding)

        return loss_state, loss_rew

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

    def __init__(self, state_dim, state_embed_dim, action_dim, action_embed_dim, agent_character_embed_dim, hidden_dim, ouput_dim,
                 state_decoder_layers, reward_decoder_layers, state_pred_type, rew_pred_type,
                 input_prev_state, input_action):
        """
        Args:
            action_dim: int, dimension of the action space
            obs_dim: int, dimension of the observation space
            embedding_dim: int, dimension of the embedding space
            hidden_dim: int, dimension of the decoder hidden layers
            ouput_dim1: int, dimension of the decoder output
            ouput_dim2: int, dimension of the decoder probs
        """
        self.model = DecoderRNNNetwork(state_embed_dim, agent_character_embed_dim, hidden_dim, ouput_dim,
                                       state_dim, action_embed_dim, state_decoder_layers, state_pred_type,
                                       reward_decoder_layers, rew_pred_type, input_prev_state, input_action)

        self.latent_state_dim = latent_state_dim
        self.agent_character_dim = agent_character_dim
        self.mental_state_dim = mental_state_dim
        self.state_dim = state_dim
        self.state_embed_dim = state_embed_dim
        self.action_dim = action_dim
        self.action_embed_dim = action_embed_dim
        self.agent_character_embed_dim = agent_character_embed_dim
        self.hidden_dim = hidden_dim
        self.ouput_dim = ouput_dim

    def init_params(self, rng):
        """Initialize parameters for the decoder model."""
        batch_size = 1
        rng, rng_decoder, rng_state_decoder, rng_reward_decoder  = jax.random.split(rng, 4)

        # Create dummy inputs - add time dimension
        dummy_latent_state = jnp.zeros((1, batch_size, self.latent_state_dim))
        dummy_state = jnp.zeros((1, batch_size, self.state_dim))
        dummy_action = jnp.zeros((1, batch_size, self.action_dim))
        dummy_agent_character = jnp.zeros((1, batch_size, self.agent_character_dim))
        dummy_mental_state = jnp.zeros((1, batch_size, self.mental_state_dim))
        dummy_done = jnp.zeros((1, batch_size))

        dummy_state_decoder = (dummy_latent_state, dummy_state, dummy_action)
        dummy_rew_decoder = (dummy_latent_state, dummy_state, dummy_state, dummy_action)
        dummy_main_decoder = (dummy_state, dummy_agent_character, dummy_mental_state, dummy_done)


        # Initialize model
        params = {'main_decoder': self.model.init(rng_decoder, dummy_main_decoder),
                  'state_decoder': self.state_decoder.init(rng_state_decoder, dummy_state_decoder),
                  'reward_decoder': self.reward_decoder.init(rng_reward_decoder, dummy_rew_decoder)}
        return params

    @functools.partial(jax.jit, static_argnums=(0,))
    def evaluate(self, params, sample):
        """Evaluate the decoder model with given parameters and inputs."""
        state_recon_loss =
        rew_recon_loss =
        kl_loss = self.model.apply(params, embedding)
        return state_recon_loss, rew_recon_loss, kl_loss

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
