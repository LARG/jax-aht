import functools

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray
    partner_obs: jnp.ndarray
    partner_action: jnp.ndarray

class ScannedLSTM(nn.Module):
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
            self.initialize_carry(*lstm_state[0].shape)[1],
            lstm_state[1],
        )
        new_lstm_state, y = nn.OptimizedLSTMCell(features=ins.shape[1])((lstm_state_0, lstm_state_1), ins)
        return new_lstm_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.OptimizedLSTMCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))

class EncoderNetwork(nn.Module):
    hidden_dim: int
    ouput_dim: int

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        lstm_in = (obs, dones)
        hidden, embedding = ScannedLSTM()(hidden, lstm_in)
        embedding = nn.Dense(self.hidden_dim)(embedding)
        embedding = nn.relu(embedding)
        embedding = nn.Dense(self.ouput_dim)(embedding)
        return embedding, hidden

class DecoderNetwork(nn.Module):
    hidden_dim: int
    ouput_dim1: int
    ouput_dim2: int

    @nn.compact
    def __call__(self, x):
        jax.debug.print(f"Decoder input shape: {x.shape}")
        h1 = nn.Dense(self.hidden_dim)(x)
        h1 = nn.relu(h1)
        h1 = nn.Dense(self.hidden_dim)(h1)
        h1 = nn.relu(h1)
        out = nn.Dense(self.ouput_dim1)(h1)

        # TODO: Handle more than 1 partner
        h2 = nn.Dense(self.hidden_dim)(x)
        h2 = nn.relu(h2)
        h2 = nn.Dense(self.hidden_dim)(h2)
        h2 = nn.relu(h2)
        prob1 = nn.Dense(self.ouput_dim2)(h2)
        prob1 = nn.softmax(prob1, axis=-1)
        # prob2 = nn.Dense(self.ouput_dim2)(h2)
        # prob2 = nn.softmax(prob2, axis=-1)
        # prob3 = nn.Dense(self.ouput_dim2)(h2)
        # prob3 = nn.softmax(prob3, axis=-1)
        return out, prob1 #, prob2, prob3
    
class Encoder():
    """Model wrapper for EncoderNetwork."""
    
    def __init__(self, action_dim, obs_dim, hidden_dim, ouput_dim):
        """
        Args:
            action_dim: int, dimension of the action space  
            obs_dim: int, dimension of the observation space
            hidden_dim: int, dimension of the encoder hidden layers
            ouput_dim: int, dimension of the encoder output
        """
        self.model = EncoderNetwork(hidden_dim, ouput_dim)
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.lstm_hidden_dim = hidden_dim

    def init_hstate(self, batch_size=1, aux_info=None):
        """Initialize hidden state for the encoder LSTM."""
        hstate =  ScannedLSTM.initialize_carry(batch_size, self.lstm_hidden_dim)
        return hstate
    
    def init_params(self, rng):
        """Initialize parameters for the encoder model."""
        batch_size = 1
        # Initialize hidden state
        init_hstate = self.init_hstate(batch_size=batch_size)
        
        # Create dummy inputs - add time dimension
        dummy_obs = jnp.zeros((1, batch_size, self.obs_dim))
        dummy_done = jnp.zeros((1, batch_size))
        dummy_x = (dummy_obs, dummy_done)

        init_hstate = (init_hstate[0].reshape(batch_size, -1),
                       init_hstate[1].reshape(batch_size, -1))
        
        # import pdb; pdb.set_trace()
        
        # Initialize model
        return self.model.init(rng, init_hstate, dummy_x)
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def compute_embedding(self, params, hstate, obs, done):
        """Embed observations using the encoder model."""
        return self.model.apply(params, hstate, (obs, done))

class Decoder():
    """Model wrapper for DecoderNetwork."""
    
    def __init__(self, action_dim, obs_dim, embedding_dim, hidden_dim, ouput_dim1, ouput_dim2):
        """
        Args:
            action_dim: int, dimension of the action space  
            obs_dim: int, dimension of the observation space
            embedding_dim: int, dimension of the embedding space
            hidden_dim: int, dimension of the decoder hidden layers
            ouput_dim1: int, dimension of the decoder output
            ouput_dim2: int, dimension of the decoder probs
        """
        self.model = DecoderNetwork(hidden_dim, ouput_dim1, ouput_dim2)
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
    def evaluate(self, params, embeddings, modelled_agent_obs, modelled_agent_act):
        """Evaluate the decoder model with given parameters and inputs."""
        mean, prob1 = self.model.apply(params, embeddings)
        recon_loss_1 = 0.5 * ((modelled_agent_obs - mean) ** 2).sum(-1)
        recon_loss_2 = -jnp.log(jnp.sum(prob1 * modelled_agent_act, axis=-1))
        return recon_loss_1, recon_loss_2

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
    encoder = Encoder(
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
