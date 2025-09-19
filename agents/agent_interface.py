import abc
from typing import Tuple, Dict
import chex
from functools import partial
import jax
import jax.numpy as jnp

from agents.mlp_actor_critic import ActorCritic
from agents.mlp_actor_critic import ActorWithDoubleCritic
from agents.mlp_actor_critic import ActorWithConditionalCritic
from agents.s5_actor_critic import S5ActorCritic, StackedEncoderModel, init_S5SSM, make_DPLR_HiPPO
from agents.rnn_actor_critic import RNNActorCritic, ScannedRNN


class AgentPolicy(abc.ABC):
    '''Abstract base class for a policy.'''

    def __init__(self, action_dim, obs_dim):
        '''
        Args:
            action_dim: int, dimension of the action space
            obs_dim: int, dimension of the observation space
        '''
        self.action_dim = action_dim
        self.obs_dim = obs_dim

    @abc.abstractmethod
    @partial(jax.jit, static_argnums=(0,))
    def get_action(self, params, obs, done, avail_actions, hstate, rng,
                   aux_obs=None, env_state=None, test_mode=False) -> Tuple[int, chex.Array]:
        """
        Only computes an action given an observation, done flag, available actions, hidden state, and random key.

        Args:
            params (dict): The parameters of the policy.
            obs (chex.Array): The observation.
            done (chex.Array): The done flag.
            avail_actions (chex.Array): The available actions.
            hstate (chex.Array): The hidden state.
            key (jax.random.PRNGKey): The random key.
            env_state (chex.Array): The environment state.
            aux_obs (chex.Array): an optional auxiliary vector to append to the observation
        Returns:
            Tuple[int, chex.Array]: A tuple containing the action and the new hidden state.
        """
        pass

    @partial(jax.jit, static_argnums=(0,))
    def get_action_value_policy(self, params, obs, done, avail_actions, hstate, rng,
                                aux_obs=None, env_state=None) -> Tuple[int, chex.Array, chex.Array, chex.Array]:
        """
        Computes the action, value, and policy given an observation,
        done flag, available actions, hidden state, and random key.

        Args:
            params (dict): The parameters of the policy.
            obs (chex.Array): The observation.
            done (chex.Array): The done flag.
            avail_actions (chex.Array): The available actions.
            hstate (chex.Array): The hidden state.
            key (jax.random.PRNGKey): The random key.
            aux_obs (chex.Array): an optional auxiliary vector to append to the observation
        Returns:
            Tuple[int, chex.Array, chex.Array, chex.Array]:
                A tuple containing the action, value, policy, and new hidden state.
        """
        pass

    def init_hstate(self, batch_size, aux_info: dict=None) -> chex.Array:
        """Initialize the hidden state for the policy.
        Args:
            batch_size: int, the batch size of the hidden state
            aux_info: any auxiliary information needed to initialize the hidden state at the
            start of an episode (e.g. the agent id).
        Returns:
            chex.Array: the initialized hidden state
        """
        return None

    def init_params(self, rng) -> Dict:
        """Initialize the parameters for the policy."""
        return None


class MLPActorCriticPolicy(AgentPolicy):
    """Policy wrapper for MLP Actor-Critic"""

    def __init__(self, action_dim, obs_dim, activation="tanh"):
        """
        Args:
            action_dim: int, dimension of the action space
            obs_dim: int, dimension of the observation space
            activation: str, activation function to use
        """
        super().__init__(action_dim, obs_dim)
        # self.activation = activation
        self.network = ActorCritic(action_dim, activation=activation)

    @partial(jax.jit, static_argnums=(0,))
    def get_action(self, params, obs, done, avail_actions, hstate, rng,
                   aux_obs=None, env_state=None, test_mode=False):
        """Get actions for the MLP policy."""
        pi, _ = self.network.apply(params, (obs, avail_actions))
        action = jax.lax.cond(test_mode,
                              lambda: pi.mode(),
                              lambda: pi.sample(seed=rng))
        return action, None  # no hidden state

    @partial(jax.jit, static_argnums=(0,))
    def get_action_value_policy(self, params, obs, done, avail_actions, hstate, rng,
                                aux_obs=None, env_state=None):
        """Get actions, values, and policy for the MLP policy."""
        pi, val = self.network.apply(params, (obs, avail_actions))
        action = pi.sample(seed=rng)
        return action, val, pi, None  # no hidden state

    def init_params(self, rng):
        """Initialize parameters for the MLP policy."""
        dummy_obs = jnp.zeros((self.obs_dim,))
        dummy_avail = jnp.ones((self.action_dim,))
        init_x = (dummy_obs, dummy_avail)
        return self.network.init(rng, init_x)


class ActorWithDoubleCriticPolicy(AgentPolicy):
    """Policy wrapper for Actor with Double Critics"""

    def __init__(self, action_dim, obs_dim, activation="tanh"):
        """
        Args:
            action_dim: int, dimension of the action space
            obs_dim: int, dimension of the observation space
            activation: str, activation function to use
        """
        super().__init__(action_dim, obs_dim)
        self.network = ActorWithDoubleCritic(action_dim, activation=activation)

    @partial(jax.jit, static_argnums=(0,))
    def get_action(self, params, obs, done, avail_actions, hstate, rng,
                   aux_obs=None, env_state=None, test_mode=False):
        """Get actions for the policy with double critics.
        """
        pi, _, _ = self.network.apply(params, (obs, avail_actions))
        action = jax.lax.cond(test_mode,
                              lambda: pi.mode(),
                              lambda: pi.sample(seed=rng))
        return action, None  # no hidden state

    @partial(jax.jit, static_argnums=(0,))
    def get_action_value_policy(self, params, obs, done, avail_actions, hstate, rng,
                                aux_obs=None, env_state=None):
        """Get actions, values, and policy for the policy with double critics."""
        # convention: val1 is value of of ego agent, val2 is value of best response agent
        pi, val1, val2 = self.network.apply(params, (obs, avail_actions))
        action = pi.sample(seed=rng)
        return action, (val1, val2), pi, None # no hidden state

    def init_params(self, rng):
        """Initialize parameters for the policy with double critics."""
        dummy_obs = jnp.zeros((self.obs_dim,))
        dummy_avail = jnp.ones((self.action_dim,))
        init_x = (dummy_obs, dummy_avail)
        return self.network.init(rng, init_x)

class PseudoActorWithDoubleCriticPolicy(ActorWithDoubleCriticPolicy):
    """Enables ActorWithDoubleCritic to masquerade as an actor with a single critic."""
    def __init__(self, action_dim, obs_dim, activation="tanh"):
        super().__init__(action_dim, obs_dim, activation)

    def get_action_value_policy(self, params, obs, done, avail_actions, hstate, rng,
                                aux_obs=None, env_state=None):
        action, (val1, _), pi, hidden_state = super().get_action_value_policy(
            params, obs, done, avail_actions, hstate, rng,
            aux_obs, env_state)
        return action, val1, pi, hidden_state

class ActorWithConditionalCriticPolicy(AgentPolicy):
    """Policy wrapper for ActorWithConditionalCritic
    """
    def __init__(self, action_dim, obs_dim, pop_size, activation="tanh"):
        """
        Args:
            action_dim: int, dimension of the action space
            obs_dim: int, dimension of the observation space
            pop_size: int, number of agents in the population that the critic was trained with
            activation: str, activation function to use
        """
        super().__init__(action_dim, obs_dim)
        # self.activation = activation
        self.pop_size = pop_size
        self.network = ActorWithConditionalCritic(action_dim, activation=activation)

    @partial(jax.jit, static_argnums=(0,))
    def get_action(self, params, obs, done, avail_actions, hstate, rng,
                   aux_obs=None, env_state=None, test_mode=False):
        """Get actions."""
        # The agent id is only used by the critic, so we pass in a
        # dummy vector to represent the one-hot agent id
        dummy_agent_id = jnp.zeros(obs.shape[:-1] + (self.pop_size,))
        pi, _ = self.network.apply(params, (obs, dummy_agent_id, avail_actions))
        action = jax.lax.cond(test_mode,
                              lambda: pi.mode(),
                              lambda: pi.sample(seed=rng))
        return action, None  # no hidden state

    @partial(jax.jit, static_argnums=(0,))
    def get_action_value_policy(self, params, obs, done, avail_actions, hstate, rng,
                                aux_obs=None, env_state=None):
        """Get actions, values, and policy for the policy with conditional critics.
        The auxiliary observation should be used to pass in the agent ids that we wish to predict
        values for.
        """
        pi, value = self.network.apply(params, (obs, aux_obs, avail_actions))
        action = pi.sample(seed=rng)
        return action, value, pi, None # no hidden state

    def init_params(self, rng):
        """Initialize parameters for the policy with conditional critics."""
        dummy_obs = jnp.zeros((self.obs_dim,))
        dummy_ids = jnp.zeros((self.pop_size,))
        dummy_avail = jnp.ones((self.action_dim,))
        init_x = (dummy_obs, dummy_ids, dummy_avail)
        return self.network.init(rng, init_x)

class PseudoActorWithConditionalCriticPolicy(ActorWithConditionalCriticPolicy):
    """Enables PseudoActorWithConditionalCriticPolicy to act as an MLPActorCriticPolicy.
    by passing in a dummy agent id.
    """
    def __init__(self, action_dim, obs_dim, pop_size, activation="tanh"):
        super().__init__(action_dim, obs_dim, pop_size, activation)

    def get_action_value_policy(self, params, obs, done, avail_actions, hstate, rng,
                                aux_obs=None, env_state=None):
        dummy_agent_id = jnp.zeros(obs.shape[:-1] + (self.pop_size,))
        action, val, pi, hidden_state = super().get_action_value_policy(
            params, obs, done, avail_actions, hstate, rng,
            dummy_agent_id, env_state)
        return action, val, pi, hidden_state

class RNNActorCriticPolicy(AgentPolicy):
    """Policy wrapper for RNN Actor-Critic"""

    def __init__(self, action_dim, obs_dim,
                 activation="tanh", fc_hidden_dim=64, gru_hidden_dim=64):
        """
        Args:
            action_dim: int, dimension of the action space
            obs_dim: int, dimension of the observation space
            activation: str, activation function to use
            fc_hidden_dim: int, dimension of the feed-forward hidden layers
            gru_hidden_dim: int, dimension of the GRU hidden state
        """
        super().__init__(action_dim, obs_dim)
        self.network = RNNActorCritic(
            action_dim,
            fc_hidden_dim=fc_hidden_dim,
            gru_hidden_dim=gru_hidden_dim,
            activation=activation
        )
        self.gru_hidden_dim = gru_hidden_dim

    @partial(jax.jit, static_argnums=(0,))
    def get_action(self, params, obs, done, avail_actions, hstate, rng,
                   aux_obs=None, env_state=None, test_mode=False):
        """Get actions for the RNN policy.
        Shape of obs, done, avail_actions should correspond to (seq_len, batch_size, ...)
        Shape of hstate should correspond to (1, batch_size, -1). We maintain the extra first dimension for
        compatibility with the learning codes.
        """
        batch_size = obs.shape[1]
        new_hstate, pi, _ = self.network.apply(params, hstate.squeeze(0), (obs, done, avail_actions))
        action = jax.lax.cond(test_mode,
                              lambda: pi.mode(),
                              lambda: pi.sample(seed=rng))
        return action, new_hstate.reshape(1, batch_size, -1)

    @partial(jax.jit, static_argnums=(0,))
    def get_action_value_policy(self, params, obs, done, avail_actions, hstate, rng,
                                aux_obs=None, env_state=None):
        """Get actions, values, and policy for the RNN policy.
        Shape of obs, done, avail_actions should correspond to (seq_len, batch_size, ...)
        Shape of hstate should correspond to (1, batch_size, -1). We maintain the extra first dimension for
        compatibility with the learning codes.
        """
        batch_size = obs.shape[1]
        new_hstate, pi, val = self.network.apply(params, hstate.squeeze(0), (obs, done, avail_actions))
        action = pi.sample(seed=rng)
        return action, val, pi, new_hstate.reshape(1, batch_size, -1)

    def init_hstate(self, batch_size, aux_info=None):
        """Initialize hidden state for the RNN policy."""
        hstate =  ScannedRNN.initialize_carry(batch_size, self.gru_hidden_dim)
        hstate = hstate.reshape(1, batch_size, self.gru_hidden_dim)
        return hstate

    def init_params(self, rng):
        """Initialize parameters for the RNN policy."""
        batch_size = 1
        # Initialize hidden state
        init_hstate = self.init_hstate(batch_size)

        # Create dummy inputs - add time dimension
        dummy_obs = jnp.zeros((1, batch_size, self.obs_dim))
        dummy_done = jnp.zeros((1, batch_size))
        dummy_avail = jnp.ones((1, batch_size, self.action_dim))
        dummy_x = (dummy_obs, dummy_done, dummy_avail)

        # Initialize model
        return self.network.init(rng, init_hstate.reshape(batch_size, -1), dummy_x)


class S5ActorCriticPolicy(AgentPolicy):
    """Policy wrapper for S5 Actor-Critic"""

    def __init__(self, action_dim, obs_dim,
                 d_model=16, ssm_size=16,
                 ssm_n_layers=2, blocks=1,
                 fc_hidden_dim=64,
                 fc_n_layers=2,
                 s5_activation="full_glu",
                 s5_do_norm=True,
                 s5_prenorm=True,
                 s5_do_gtrxl_norm=True,
                 s5_no_reset=False):
        """
        Args:
            action_dim: int, dimension of the action space
            obs_dim: int, dimension of the observation space
            d_model: int, dimension of the model
            ssm_size: int, size of the SSM
            n_layers: int, number of S5 layers
            blocks: int, number of blocks to split SSM parameters
            fc_hidden_dim: int, dimension of the fully connected hidden layers
            s5_activation: str, activation function to use in S5
            s5_do_norm: bool, whether to apply normalization in S5
            s5_prenorm: bool, whether to apply pre-normalization in S5
            s5_do_gtrxl_norm: bool, whether to apply gtrxl normalization in S5
            s5_no_reset: bool, whether to ignore reset signals
        """
        super().__init__(action_dim, obs_dim)
        self.d_model = d_model
        self.ssm_size = ssm_size
        self.ssm_n_layers = ssm_n_layers
        self.blocks = blocks
        self.fc_hidden_dim = fc_hidden_dim
        self.fc_n_layers = fc_n_layers
        self.s5_activation = s5_activation
        self.s5_do_norm = s5_do_norm
        self.s5_prenorm = s5_prenorm
        self.s5_do_gtrxl_norm = s5_do_gtrxl_norm
        self.s5_no_reset = s5_no_reset

        # Initialize SSM parameters
        block_size = int(ssm_size / blocks)
        Lambda, _, _, V, _ = make_DPLR_HiPPO(ssm_size)
        block_size = block_size // 2
        ssm_size_half = ssm_size // 2
        Lambda = Lambda[:block_size]
        V = V[:, :block_size]
        Vinv = V.conj().T

        self.ssm_init_fn = init_S5SSM(
            H=d_model,
            P=ssm_size_half,
            Lambda_re_init=Lambda.real,
            Lambda_im_init=Lambda.imag,
            V=V,
            Vinv=Vinv
        )

        # Initialize the network instance once
        self.network = S5ActorCritic(
            action_dim,
            ssm_init_fn=self.ssm_init_fn,
            fc_hidden_dim=self.fc_hidden_dim,
            fc_n_layers=self.fc_n_layers,
            ssm_hidden_dim=self.ssm_size,
            s5_d_model=self.d_model,
            s5_n_layers=self.ssm_n_layers,
            s5_activation=self.s5_activation,
            s5_do_norm=self.s5_do_norm,
            s5_prenorm=self.s5_prenorm,
            s5_do_gtrxl_norm=self.s5_do_gtrxl_norm,
            s5_no_reset=self.s5_no_reset
        )

    @partial(jax.jit, static_argnums=(0,))
    def get_action(self, params, obs, done, avail_actions, hstate, rng,
                   aux_obs=None, env_state=None, test_mode=False):
        """Get actions for the S5 policy."""
        new_hstate, pi, _ = self.network.apply(params, hstate, (obs, done, avail_actions))
        action = jax.lax.cond(test_mode,
                              lambda: pi.mode(),
                              lambda: pi.sample(seed=rng))
        return action, new_hstate

    @partial(jax.jit, static_argnums=(0,))
    def get_action_value_policy(self, params, obs, done, avail_actions, hstate, rng,
                                aux_obs=None, env_state=None):
        """Get actions, values, and policy for the S5 policy.
        Shape of obs, done, avail_actions should correspond to (seq_len, batch_size, ...)
        Shape of hstate should correspond to (1, batch_size, -1)
        """
        new_hstate, pi, val = self.network.apply(params, hstate, (obs, done, avail_actions))
        action = pi.sample(seed=rng)
        return action, val, pi, new_hstate

    def init_hstate(self, batch_size, aux_info=None):
        """Initialize hidden state for the S5 policy."""

        init_hstate =  StackedEncoderModel.initialize_carry(batch_size, self.ssm_size // 2, self.ssm_n_layers)
        return init_hstate

    def init_params(self, rng):
        """Initialize parameters for the S5 policy."""
        batch_size = 1
        # Initialize hidden state
        init_hstate = self.init_hstate(batch_size)

        # Create dummy inputs
        dummy_obs = jnp.zeros((1, batch_size, self.obs_dim))
        dummy_done = jnp.zeros((1, batch_size))
        dummy_avail = jnp.ones((1, batch_size, self.action_dim))
        dummy_x = (dummy_obs, dummy_done, dummy_avail)

        # Initialize model using the pre-initialized network
        return self.network.init(rng, init_hstate, dummy_x)

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

        embbeding, new_encoder_hstate = self.encoder.compute_embedding(
            params=params['encoder'],
            hstate=hstate[0],
            obs=jnp.concatenate((obs, act), axis=-1),
            done=done
        )

        action, new_policy_hstate = self.policy.get_action(
            params=params['policy'],
            obs=jnp.concatenate((obs, embbeding), axis=-1),
            done=done,
            avail_actions=avail_actions,
            hstate=hstate[1],
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
        Get actions, values, and policy for the lIAM policy.

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

        embbeding, new_encoder_hstate = self.encoder.compute_embedding(
            params=params['encoder'],
            hstate=hstate[0],
            obs=jnp.concatenate((obs, act), axis=-1),
            done=done
        )

        action, val, pi, new_policy_hstate = self.policy.get_action_value_policy(
            params=params['policy'],
            obs=jnp.concatenate((obs, embbeding), axis=-1),
            done=done,
            avail_actions=avail_actions,
            hstate=hstate[1],
            rng=rng,
            aux_obs=aux_obs,
            env_state=env_state
        )

        return action, val, pi, (new_encoder_hstate, new_policy_hstate)

    @partial(jax.jit, static_argnums=(0,))
    def evaluate(self, params, obs, done, avail_actions, hstate, rng,
                modelled_agent_obs, modelled_agent_act,
                aux_obs=None, env_state=None):
        """
        Get actions, values, policy, and decoder reconstruction losses for the lIAM policy.

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

        embbeding, new_encoder_hstate = self.encoder.compute_embedding(
            params=params['encoder'],
            hstate=hstate[0],
            obs=jnp.concatenate((obs, act), axis=-1),
            done=done
        )

        action, val, pi, new_policy_hstate = self.policy.get_action_value_policy(
            params=params['policy'],
            obs=jnp.concatenate((obs, jax.lax.stop_gradient(embbeding)), axis=-1),
            done=done,
            avail_actions=avail_actions,
            hstate=hstate[1],
            rng=rng,
            aux_obs=aux_obs,
            env_state=env_state
        )

        # Reconstruction Loss
        recon_loss1, recon_loss2 = self.decoder.evaluate(
            params=params['decoder'],
            embedding=embbeding,
            modelled_agent_obs=modelled_agent_obs,
            modelled_agent_act=modelled_agent_act
        )

        return action, val, pi, recon_loss1, recon_loss2, (new_encoder_hstate, new_policy_hstate)

class MeLIBAPolicy(AgentPolicy):
    """MeLIBA inference policy that uses an encoder and decoder to model partner behavior."""

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
        Get actions for the MeLIBA policy.

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
        _, joint_act, reward = aux_obs

        rng, policy_rng, sample_key  = jax.random.split(rng, 3)

        _, latent_mean, latent_logvar, _, latent_mean_t, latent_logvar_t, new_encoder_hstate = self.encoder.compute_embedding(
            params=params['encoder'],
            hstate=hstate[0],
            state=obs,
            act=joint_act,
            reward=reward,
            done=done,
            sample_key=sample_key
        )

        action, new_policy_hstate = self.policy.get_action(
            params=params['policy'],
            obs=jnp.concatenate((obs, latent_mean, latent_logvar, latent_mean_t, latent_logvar_t), axis=-1),
            done=done,
            avail_actions=avail_actions,
            hstate=hstate[1],
            rng=policy_rng,
            aux_obs=aux_obs,
            env_state=env_state,
            test_mode=test_mode
        )

        return action, (new_encoder_hstate, new_policy_hstate)

    @partial(jax.jit, static_argnums=(0,))
    def get_action_value_policy(self, params, obs, done, avail_actions, hstate, rng,
                                aux_obs=None, env_state=None):
        """
        Get actions, values, and policy for the MeLIBA policy.

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
        _, joint_act, reward = aux_obs

        rng, policy_rng, sample_key  = jax.random.split(rng, 3)

        _, latent_mean, latent_logvar, _, latent_mean_t, latent_logvar_t, new_encoder_hstate = self.encoder.compute_embedding(
            params=params['encoder'],
            hstate=hstate[0],
            state=obs,
            act=joint_act,
            reward=reward,
            done=done,
            sample_key=sample_key
        )

        action, val, pi, new_policy_hstate = self.policy.get_action_value_policy(
            params=params['policy'],
            obs=jnp.concatenate((obs, latent_mean, latent_logvar, latent_mean_t, latent_logvar_t), axis=-1),
            done=done,
            avail_actions=avail_actions,
            hstate=hstate[1],
            rng=policy_rng,
            aux_obs=aux_obs,
            env_state=env_state
        )

        return action, val, pi, (new_encoder_hstate, new_policy_hstate)

    @partial(jax.jit, static_argnums=(0,))
    def evaluate(self, params, obs, done, avail_actions, hstate, rng,
                 partner_action, aux_obs=None, env_state=None):
        """
        Get actions, values, policy, and decoder reconstructions for the MeLIBA policy.

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
            partner_action: jnp.Array, the actions of the modeled agent for decoder loss
            aux_obs: tuple of auxiliary observations i.e. (act, joint_act, reward)
            env_state: jnp.Array, the environment state

        Returns:
            action: jnp.Array, the selected action
            val: jnp.Array, the value estimate
            pi: jnp.Array, the policy distribution
            kl_loss: jnp.Array, the KL divergence loss from the decoder
            recon_loss: jnp.Array, the reconstruction loss from the decoder
            new_hstate: tuple(jnp.Array, jnp.Array), the new hidden state for the encoder and policy
        """
        _, joint_act, reward = aux_obs

        rng, policy_rng, sample_key  = jax.random.split(rng, 3)

        latent_sample, latent_mean, latent_logvar, latent_sample_t, latent_mean_t, latent_logvar_t, new_encoder_hstate = self.encoder.compute_embedding(
            params=params['encoder'],
            hstate=hstate[0],
            state=obs,
            act=joint_act,
            reward=jnp.expand_dims(reward, axis=-1),
            done=done,
            sample_key=sample_key
        )

        action, val, pi, new_policy_hstate = self.policy.get_action_value_policy(
            params=params['policy'],
            obs=jnp.concatenate((obs, jax.lax.stop_gradient(jnp.concatenate((latent_mean, latent_logvar, latent_mean_t, latent_logvar_t), axis=-1))), axis=-1),
            done=done,
            avail_actions=avail_actions,
            hstate=hstate[1],
            rng=policy_rng,
            aux_obs=aux_obs,
            env_state=env_state
        )

        # Reconstruction Loss
        kl_loss, recon_loss = self.decoder.evaluate(
            params=params['decoder'],
            state=obs,
            latent_mean=latent_mean,
            latent_logvar=latent_logvar,
            latent_mean_t=latent_mean_t,
            latent_logvar_t=latent_logvar_t,
            agent_character=latent_sample,
            mental_state=latent_sample_t,
            partner_action=partner_action,
            done=done
        )

        return action, val, pi, kl_loss, recon_loss, (new_encoder_hstate, new_policy_hstate)
