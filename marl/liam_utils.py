import functools

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn

from agents.rnn_actor_critic import ScannedRNN
from agents.s5_actor_critic import SequenceLayer

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

class EncoderLSTMNetwork(nn.Module):
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
        return hidden, embedding

class EncoderRNNNetwork(nn.Module):
    hidden_dim: int
    ouput_dim: int

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        rnn_in = (obs, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)
        embedding = nn.Dense(self.hidden_dim)(embedding)
        embedding = nn.relu(embedding)
        embedding = nn.Dense(self.ouput_dim)(embedding)
        return hidden, embedding

class EncoderS5Network(nn.Module):
    hidden_dim: int
    ouput_dim: int

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        # TODO: Fix S5 initialization
        hidden, embedding = SequenceLayer()(hidden, obs, dones)
        embedding = nn.Dense(self.hidden_dim)(embedding)
        embedding = nn.relu(embedding)
        embedding = nn.Dense(self.ouput_dim)(embedding)
        return hidden, embedding

class DecoderNetwork(nn.Module):
    hidden_dim: int
    ouput_dim1: int
    ouput_dim2: int

    @nn.compact
    def __call__(self, x):
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
    
class EncoderLSTM():
    """Model wrapper for EncoderLSTMNetwork."""
    
    def __init__(self, action_dim, obs_dim, hidden_dim, ouput_dim):
        """
        Args:
            action_dim: int, dimension of the action space  
            obs_dim: int, dimension of the observation space
            hidden_dim: int, dimension of the encoder hidden layers
            ouput_dim: int, dimension of the encoder output
        """
        self.model = EncoderLSTMNetwork(hidden_dim, ouput_dim)
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
        dummy_done = jnp.zeros((1, batch_size))
        dummy_x = (dummy_obs, dummy_done)

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
    
    def __init__(self, action_dim, obs_dim, hidden_dim, ouput_dim):
        """
        Args:
            action_dim: int, dimension of the action space  
            obs_dim: int, dimension of the observation space
            hidden_dim: int, dimension of the encoder hidden layers
            ouput_dim: int, dimension of the encoder output
        """
        self.model = EncoderRNNNetwork(hidden_dim, ouput_dim)
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
        dummy_done = jnp.zeros((1, batch_size))
        dummy_x = (dummy_obs, dummy_done)

        init_hstate = init_hstate.reshape(batch_size, -1)
        
        # Initialize model
        return self.model.init(rng, init_hstate, dummy_x)
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def compute_embedding(self, params, hstate, obs, done):
        """Embed observations using the encoder model."""
        batch_size = obs.shape[1]

        new_hstate, embedding = self.model.apply(params, hstate.squeeze(0), (obs, done))

        return embedding, new_hstate.reshape(1, batch_size, -1)

# TODO: Fix S5 encoder
class EncoderS5():
    """Model wrapper for EncoderS5Network."""
    
    def __init__(self, action_dim, obs_dim, hidden_dim, ouput_dim):
        """
        Args:
            action_dim: int, dimension of the action space  
            obs_dim: int, dimension of the observation space
            hidden_dim: int, dimension of the encoder hidden layers
            ouput_dim: int, dimension of the encoder output
        """
        self.model = EncoderS5Network(hidden_dim, ouput_dim)
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
        
        # Create dummy inputs - add time dimension
        dummy_obs = jnp.zeros((1, batch_size, self.obs_dim))
        dummy_done = jnp.zeros((1, batch_size))
        dummy_x = (dummy_obs, dummy_done)
        
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
    def evaluate(self, params, embedding, modelled_agent_obs, modelled_agent_act):
        """Evaluate the decoder model with given parameters and inputs."""
        mean, prob1 = self.model.apply(params, embedding)
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
    encoder = EncoderLSTM(
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

def _create_minibatches(traj_batch, advantages, targets, init_hstate, init_encoder_hstate, num_actors, num_minibatches, perm_rng):
    """Create minibatches for PPO updates, where each leaf has shape 
        (num_minibatches, rollout_len, num_actors / num_minibatches, ...) 
    This function ensures that the rollout (time) dimension is kept separate from the minibatch and num_actors 
    dimensions, so that the minibatches are compatible with recurrent ActorCritics.
    """
    # Create batch containing trajectory, advantages, and targets
    batch = (
        init_hstate, # shape (1, num_actors, hidden_dim)
        init_encoder_hstate, # shape (1, num_actors, hidden_dim)
        traj_batch, # pytree: obs is shape (rollout_len, num_actors, feat_shape)
        advantages, # shape (rollout_len, num_actors)
        targets # shape (rollout_len, num_actors)
            )

    permutation = jax.random.permutation(perm_rng, num_actors)

    # each leaf of shuffled batch has shape (rollout_len, num_actors, feat_shape)
    # except for init_hstate which has shape (1, num_actors, hidden_dim)
    shuffled_batch = jax.tree.map(
        lambda x: jnp.take(x, permutation, axis=1), batch
    )
    # each leaf has shape (num_minibatches, rollout_len, num_actors/num_minibatches, feat_shape)
    # except for init_hstate which has shape (num_minibatches, 1, num_actors/num_minibatches, hidden_dim)
    minibatches = jax.tree_util.tree_map(
        lambda x: jnp.swapaxes(
            jnp.reshape(
                x,
                [x.shape[0], num_minibatches, -1] 
                + list(x.shape[2:]),
        ), 1, 0,),
        shuffled_batch,
    )

    return minibatches
