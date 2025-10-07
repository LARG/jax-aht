import functools

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn

from functools import partial

from agents.agent_interface import AgentPolicy
from agents.rnn_actor_critic import ScannedRNN
from agents.s5_actor_critic import SequenceLayer

class ScannedLSTM(nn.Module):
    """
    A LSTM module that can be scanned over time.

    It resets its state based on the `dones` signal
    """
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        lstm_state = carry
        ins, dones = x
        lstm_state_0 = jnp.where(
            dones[:, np.newaxis],
            self.initialize_carry(*lstm_state[0].shape)[0],
            lstm_state[0],
        )
        lstm_state_1 = jnp.where(
            dones[:, np.newaxis],
            self.initialize_carry(*lstm_state[1].shape)[1],
            lstm_state[1],
        )
        new_lstm_state, y = nn.OptimizedLSTMCell(features=ins.shape[1])((lstm_state_0, lstm_state_1), ins)
        return new_lstm_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.OptimizedLSTMCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))

class EncoderLSTMNetwork(nn.Module):
    """
    LSTM-based encoder network.

    Args:
        hidden_dim: int, dimension of the hidden layers
        output_dim: int, dimension of the output embedding
    """
    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, hidden, x):
        """
        Forward pass of the LSTM-based encoder network.

        Args:
            hidden: tuple of (h, c) where each is jnp.array of shape (batch_size, hidden_dim)
            x: tuple of (obs, dones) where
                obs: jnp.array of shape (time, batch_size, obs_dim)
                dones: jnp.array of shape (time, batch_size)

        Returns:
            new_hidden: tuple of (h, c) where each is jnp.array of shape (batch_size, hidden_dim)
            embedding: jnp.array of shape (time, batch_size, output_dim)
        """
        obs, dones = x
        lstm_in = (obs, dones)
        hidden, embedding = ScannedLSTM()(hidden, lstm_in)
        embedding = nn.Dense(self.hidden_dim)(embedding)
        embedding = nn.relu(embedding)
        embedding = nn.Dense(self.output_dim)(embedding)
        return hidden, embedding

class EncoderRNNNetwork(nn.Module):
    """
    RNN-based encoder network.

    Args:
        hidden_dim: int, dimension of the hidden layers
        output_dim: int, dimension of the output embedding
    """
    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, hidden, x):
        """
        Forward pass of the RNN-based encoder network.

        Args:
            hidden: jnp.array of shape (batch_size, hidden_dim)
            x: tuple of (obs, dones) where
                obs: jnp.array of shape (time, batch_size, obs_dim)
                dones: jnp.array of shape (time, batch_size)

        Returns:
            new_hidden: jnp.array of shape (batch_size, hidden_dim)
            embedding: jnp.array of shape (time, batch_size, output_dim)
        """
        obs, dones = x
        rnn_in = (obs, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)
        embedding = nn.Dense(self.hidden_dim)(embedding)
        embedding = nn.relu(embedding)
        embedding = nn.Dense(self.output_dim)(embedding)
        return hidden, embedding

class DecoderNetwork(nn.Module):
    """
    Decoder network.

    Args:
        hidden_dim: int, dimension of the hidden layers
        output_dim1: int, dimension of the output for the reconstruction of observations
        output_dim2: int, dimension of the output for the reconstruction of partner actions
    """
    hidden_dim: int
    output_dim1: int
    output_dim2: int

    @nn.compact
    def __call__(self, x):
        """
        Forward pass of the decoder network.

        Args:
            x: jnp.array of shape (batch_size, embedding_dim)

        Returns:
            out: jnp.array of shape (batch_size, output_dim1)
            prob1: jnp.array of shape (batch_size, output_dim2)"""
        h1 = nn.Dense(self.hidden_dim)(x)
        h1 = nn.relu(h1)
        h1 = nn.Dense(self.hidden_dim)(h1)
        h1 = nn.relu(h1)
        out = nn.Dense(self.output_dim1)(h1)

        # TODO: Handle more than 1 partner
        h2 = nn.Dense(self.hidden_dim)(x)
        h2 = nn.relu(h2)
        h2 = nn.Dense(self.hidden_dim)(h2)
        h2 = nn.relu(h2)
        prob1 = nn.Dense(self.output_dim2)(h2)
        prob1 = nn.softmax(prob1, axis=-1)

        # To handle more than 1 partner we will have to add more output heads
        # Should find a dynamic way to do this once we extend to more than 1 partner
        # This shows how it would look like for 3 partners
        # prob2 = nn.Dense(self.output_dim2)(h2)
        # prob2 = nn.softmax(prob2, axis=-1)
        # prob3 = nn.Dense(self.output_dim2)(h2)
        # prob3 = nn.softmax(prob3, axis=-1)

        return out, prob1 #, prob2, prob3

class EncoderLSTM():
    """Model wrapper for EncoderLSTMNetwork."""

    def __init__(self, action_dim, obs_dim, hidden_dim, output_dim):
        """
        Args:
            action_dim: int, dimension of the action space
            obs_dim: int, dimension of the observation space
            hidden_dim: int, dimension of the encoder hidden layers
            output_dim: int, dimension of the encoder output
        """
        self.model = EncoderLSTMNetwork(hidden_dim, output_dim)
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.lstm_hidden_dim = hidden_dim

    def init_hstate(self, batch_size=1, aux_info=None):
        """Initialize hidden state for the encoder LSTM."""
        hstate =  ScannedLSTM.initialize_carry(batch_size, self.lstm_hidden_dim)
        hstate = (hstate[0].reshape(1, batch_size, self.lstm_hidden_dim),
                  hstate[1].reshape(1, batch_size, self.lstm_hidden_dim))
        return hstate

    def init_params(self, rng):
        """Initialize parameters for the encoder model."""
        batch_size = 1
        # Initialize hidden state
        init_hstate = self.init_hstate(batch_size=batch_size)

        # Create dummy inputs - add time dimension
        dummy_obs = jnp.zeros((1, batch_size, self.obs_dim))
        dummy_act = jnp.zeros((1, batch_size, self.action_dim))
        dummy_done = jnp.zeros((1, batch_size))
        dummy_x = (jnp.concatenate((dummy_obs, dummy_act), axis=-1), dummy_done)

        init_hstate = (init_hstate[0].reshape(batch_size, -1),
                       init_hstate[1].reshape(batch_size, -1))

        # Initialize model
        return self.model.init(rng, init_hstate, dummy_x)

    @functools.partial(jax.jit, static_argnums=(0,))
    def compute_embedding(self, params, hstate, obs, done):
        """Embed observations using the encoder model."""
        batch_size = obs.shape[1]

        hstate = (hstate[0].squeeze(0),
                  hstate[1].squeeze(0))

        new_hstate, embedding = self.model.apply(params, hstate, (obs, done))

        new_hstate = (new_hstate[0].reshape(1, batch_size, -1),
                      new_hstate[1].reshape(1, batch_size, -1))
        return embedding, new_hstate

class EncoderRNN():
    """Model wrapper for EncoderRNNNetwork."""

    def __init__(self, action_dim, obs_dim, hidden_dim, output_dim):
        """
        Args:
            action_dim: int, dimension of the action space
            obs_dim: int, dimension of the observation space
            hidden_dim: int, dimension of the encoder hidden layers
            output_dim: int, dimension of the encoder output
        """
        self.model = EncoderRNNNetwork(hidden_dim, output_dim)
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.rnn_hidden_dim = hidden_dim

    def init_hstate(self, batch_size=1, aux_info=None):
        """Initialize hidden state for the encoder RNN."""
        hstate =  ScannedRNN.initialize_carry(batch_size, self.rnn_hidden_dim)
        hstate = hstate.reshape(1, batch_size, self.rnn_hidden_dim)
        return hstate

    def init_params(self, rng):
        """Initialize parameters for the encoder model."""
        batch_size = 1
        # Initialize hidden state
        init_hstate = self.init_hstate(batch_size=batch_size)

        # Create dummy inputs - add time dimension
        dummy_obs = jnp.zeros((1, batch_size, self.obs_dim))
        dummy_act = jnp.zeros((1, batch_size, self.action_dim))
        dummy_done = jnp.zeros((1, batch_size))
        dummy_x = (jnp.concatenate((dummy_obs, dummy_act), axis=-1), dummy_done)

        init_hstate = init_hstate.reshape(batch_size, -1)

        # Initialize model
        return self.model.init(rng, init_hstate, dummy_x)

    @functools.partial(jax.jit, static_argnums=(0,))
    def compute_embedding(self, params, hstate, obs, done):
        """Embed observations using the encoder model."""
        batch_size = obs.shape[1]

        new_hstate, embedding = self.model.apply(params, hstate.squeeze(0), (obs, done))

        return embedding, new_hstate.reshape(1, batch_size, -1)

class Decoder():
    """Model wrapper for DecoderNetwork."""

    def __init__(self, action_dim, obs_dim, embedding_dim, hidden_dim, output_dim1, output_dim2):
        """
        Args:
            action_dim: int, dimension of the action space
            obs_dim: int, dimension of the observation space
            embedding_dim: int, dimension of the embedding space
            hidden_dim: int, dimension of the decoder hidden layers
            output_dim1: int, dimension of the decoder output
            output_dim2: int, dimension of the decoder probs
        """
        self.model = DecoderNetwork(hidden_dim, output_dim1, output_dim2)
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.embedding_dim = embedding_dim

    def init_params(self, rng):
        """Initialize parameters for the decoder model."""
        batch_size = 1

        # Create dummy inputs - add time dimension
        dummy_x = jnp.zeros((1, batch_size, self.embedding_dim))

        # Initialize model
        return self.model.init(rng, dummy_x)

    @functools.partial(jax.jit, static_argnums=(0,))
    def compute_losses(self, params, embedding, modelled_agent_obs, modelled_agent_act):
        """Evaluate the decoder model with given parameters and inputs."""
        mean, prob1 = self.model.apply(params, embedding)
        recon_loss_1 = 0.5 * ((modelled_agent_obs - mean) ** 2).sum(-1)
        recon_loss_2 = -jnp.log(jnp.sum(prob1 * modelled_agent_act, axis=-1))
        return recon_loss_1.mean(), recon_loss_2.mean()

def initialize_liam_encoder_decoder(config, env, rng):
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
    encoder_type = config.get("ENCODER_TYPE", "lstm")
    if encoder_type == "lstm":
        encoder = EncoderLSTM(
            action_dim=env.action_space(env.agents[0]).n,
            obs_dim=env.observation_space(env.agents[0]).shape[0],
            hidden_dim=config.get("ENCODER_HIDDEN_DIM", 64),
            output_dim=config.get("ENCODER_OUTPUT_DIM", 64)
        )
    elif encoder_type == "rnn":
        encoder = EncoderRNN(
            action_dim=env.action_space(env.agents[0]).n,
            obs_dim=env.observation_space(env.agents[0]).shape[0],
            hidden_dim=config.get("ENCODER_HIDDEN_DIM", 64),
            output_dim=config.get("ENCODER_OUTPUT_DIM", 64)
        )

    decoder = Decoder(
        action_dim=env.action_space(env.agents[0]).n,
        obs_dim=env.observation_space(env.agents[0]).shape[0],
        embedding_dim=config.get("ENCODER_OUTPUT_DIM", 64),
        hidden_dim=config.get("DECODER_HIDDEN_DIM", 64),
        output_dim1=env.observation_space(env.agents[1]).shape[0],
        output_dim2=env.action_space(env.agents[1]).n
    )

    rng, init_rng_encoder, init_rng_decoder  = jax.random.split(rng, 3)
    init_params_encoder = encoder.init_params(init_rng_encoder)
    init_params_decoder = decoder.init_params(init_rng_decoder)

    return encoder, decoder, {'encoder': init_params_encoder, 'decoder': init_params_decoder}

class LIAMPolicy(AgentPolicy):
    """LIAM inference policy that uses an encoder and decoder to model partner behavior."""

    def __init__(self, policy, encoder, decoder):
        """
        Args:
            policy: the policy model
            encoder: the LIAM encoder model
            decoder: the LIAM decoder model
        """
        super().__init__(action_dim=policy.action_dim, obs_dim=policy.obs_dim)
        self.policy = policy
        self.encoder = encoder
        self.decoder = decoder

    def init_hstate(self, batch_size=1, aux_info=None):
        """
        Initialize hidden state for the LIAM policy.

        Args:
            batch_size: int, the batch size of the hidden state
            aux_info: any auxiliary information needed to initialize the hidden state at the
            start of an episode

        Returns:
            hstate: tuple of (encoder_hstate, policy_hstate)
        """
        encoder_hstate = self.encoder.init_hstate(batch_size=batch_size, aux_info=aux_info)
        policy_hstate = self.policy.init_hstate(batch_size=batch_size, aux_info=aux_info)
        return (encoder_hstate, policy_hstate)

    def init_params(self, rng):
        """
        Initialize parameters for the LIAM policy.

        Args:
            rng: jax.random.PRNGKey, random key for initialization

        Returns:
            params: dict, containing encoder and policy parameters
        """
        rng, init_rng_encoder, init_rng_decoder, init_rng_policy  = jax.random.split(rng, 4)
        encoder_params = self.encoder.init_params(init_rng_encoder)
        decoder_params = self.decoder.init_params(init_rng_decoder)
        policy_params = self.policy.init_params(init_rng_policy)
        return {'encoder': encoder_params, 'decoder': decoder_params, 'policy': policy_params}

    @partial(jax.jit, static_argnums=(0,))
    def get_action(self, params, obs, done, avail_actions, hstate, rng,
                   aux_obs=None, env_state=None, test_mode=False):
        """
        Get actions for the LIAM policy.

        Shape of obs, done, avail_actions should correspond to (seq_len, batch_size, ...)
        Shape of hstate should correspond to (1, batch_size, -1). We maintain the extra first dimension for
        compatibility with the learning codes.

        Args:
            params: dict, containing encoder and policy parameters
            obs: jnp.Array, the observation
            done: jnp.Array, the done flag
            avail_actions: jnp.Array, the available actions
            hstate: tuple(jnp.Array, jnp.Array), the hidden state for the encoder and policy
            rng: jax.random.PRNGKey, random key for action sampling
            aux_obs: tuple of auxiliary observations i.e. (act, joint_act, reward)
            env_state: jnp.Array, the environment state
            test_mode: bool, whether to use deterministic action selection

        Returns:
            action: jnp.Array, the selected action
            new_hstate: tuple(jnp.Array, jnp.Array), the new hidden state for the encoder and policy
        """
        act, _, _ = aux_obs

        embedding, new_encoder_hstate = self.encoder.compute_embedding(
            params=params['encoder'],
            hstate=hstate[0], # Encoder hidden state
            obs=jnp.concatenate((obs, act), axis=-1),
            done=done
        )

        action, new_policy_hstate = self.policy.get_action(
            params=params['policy'],
            obs=jnp.concatenate((obs, embedding), axis=-1),
            done=done,
            avail_actions=avail_actions,
            hstate=hstate[1], # Policy hidden state
            rng=rng,
            aux_obs=aux_obs,
            env_state=env_state,
            test_mode=test_mode
        )

        return action, (new_encoder_hstate, new_policy_hstate)

    @partial(jax.jit, static_argnums=(0,))
    def get_action_value_policy(self, params, obs, done, avail_actions, hstate, rng,
                                aux_obs=None, env_state=None):
        """
        Get actions, values, and policy for the LIAM policy.

        Shape of obs, done, avail_actions should correspond to (seq_len, batch_size, ...)
        Shape of hstate should correspond to (1, batch_size, -1). We maintain the extra first dimension for
        compatibility with the learning codes.

        Args:
            params: dict, containing encoder and policy parameters
            obs: jnp.Array, the observation
            done: jnp.Array, the done flag
            avail_actions: jnp.Array, the available actions
            hstate: tuple(jnp.Array, jnp.Array), the hidden state for the encoder and policy
            rng: jax.random.PRNGKey, random key for action sampling
            aux_obs: tuple of auxiliary observations i.e. (act, joint_act, reward)
            env_state: jnp.Array, the environment state

        Returns:
            action: jnp.Array, the selected action
            val: jnp.Array, the value estimate
            pi: jnp.Array, the policy distribution
            new_hstate: tuple(jnp.Array, jnp.Array), the new hidden state for the encoder and policy
        """
        act, _, _ = aux_obs

        embedding, new_encoder_hstate = self.encoder.compute_embedding(
            params=params['encoder'],
            hstate=hstate[0], # Encoder hidden state
            obs=jnp.concatenate((obs, act), axis=-1),
            done=done
        )

        action, val, pi, new_policy_hstate = self.policy.get_action_value_policy(
            params=params['policy'],
            obs=jnp.concatenate((obs, embedding), axis=-1),
            done=done,
            avail_actions=avail_actions,
            hstate=hstate[1], # Policy hidden state
            rng=rng,
            aux_obs=aux_obs,
            env_state=env_state
        )

        return action, val, pi, (new_encoder_hstate, new_policy_hstate)

    @partial(jax.jit, static_argnums=(0,))
    def compute_decoder_losses(self, params, obs, done, avail_actions, hstate, rng,
                               modelled_agent_obs, modelled_agent_act,
                               aux_obs=None, env_state=None):
        """
        Get actions, values, policy, and decoder reconstruction losses for the LIAM policy.

        Shape of obs, done, avail_actions should correspond to (seq_len, batch_size, ...)
        Shape of hstate should correspond to (1, batch_size, -1). We maintain the extra first dimension for
        compatibility with the learning codes.

        Args:
            params: dict, containing encoder, decoder and policy parameters
            obs: jnp.Array, the observation
            done: jnp.Array, the done flag
            avail_actions: jnp.Array, the available actions
            hstate: tuple(jnp.Array, jnp.Array), the hidden state for the encoder and policy
            rng: jax.random.PRNGKey, random key for action sampling
            modelled_agent_obs: jnp.Array, the observations of the modeled agent for decoder loss
            modelled_agent_act: jnp.Array, the actions of the modeled agent for decoder loss
            aux_obs: tuple of auxiliary observations i.e. (act, joint_act, reward)
            env_state: jnp.Array, the environment state

        Returns:
            action: jnp.Array, the selected action
            val: jnp.Array, the value estimate
            pi: jnp.Array, the policy distribution
            recon_loss1: jnp.Array, the reconstruction loss from the decoder
            recon_loss2: jnp.Array, the partner reconstruction loss from the decoder
            new_hstate: tuple(jnp.Array, jnp.Array), the new hidden state for the encoder and policy
        """
        act, _, _ = aux_obs

        embedding, new_encoder_hstate = self.encoder.compute_embedding(
            params=params['encoder'],
            hstate=hstate[0], # Encoder hidden state
            obs=jnp.concatenate((obs, act), axis=-1),
            done=done
        )

        action, val, pi, new_policy_hstate = self.policy.get_action_value_policy(
            params=params['policy'],
            obs=jnp.concatenate((obs, jax.lax.stop_gradient(embedding)), axis=-1),
            done=done,
            avail_actions=avail_actions,
            hstate=hstate[1], # Policy hidden state
            rng=rng,
            aux_obs=aux_obs,
            env_state=env_state
        )

        # Reconstruction Loss
        recon_loss1, recon_loss2 = self.decoder.compute_losses(
            params=params['decoder'],
            embedding=embedding,
            modelled_agent_obs=modelled_agent_obs,
            modelled_agent_act=modelled_agent_act
        )

        return action, val, pi, recon_loss1, recon_loss2, (new_encoder_hstate, new_policy_hstate)
