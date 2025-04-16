import abc
from typing import Tuple, Dict
import chex
from functools import partial
import jax
import jax.numpy as jnp

from common.mlp_actor_critic import ActorCritic
from common.mlp_actor_critic import ActorWithDoubleCritic
from common.s5_actor_critic import S5ActorCritic, StackedEncoderModel, init_S5SSM, make_DPLR_HiPPO
from common.rnn_actor_critic import RNNActorCritic, ScannedRNN

class AgentPopulation:
    '''Base class for a population of identical agents'''
    def __init__(self, pop_size, policy_cls):
        '''
        Args:
            pop_size: int, number of agents in the population
            policy_cls: an instance of the AgentPolicy class. The policy class for the population of agents
        '''
        self.pop_size = pop_size
        self.policy_cls = policy_cls # AgentPolicy class

    def sample_agent_indices(self, n, rng):
        '''Sample n indices from the population, with replacement.'''
        return jax.random.randint(rng, (n,), 0, self.pop_size)
    
    def gather_agent_params(self, pop_params, agent_indices):
        '''Gather the parameters of the agents specified by agent_indices.

        Args:
            pop_params: pytree of parameters for the population of agents of shape (pop_size, ...).
            agent_indices: indices with shape (num_envs,), each in [0, pop_size)
        '''
        def gather_leaf(leaf):
            # leaf shape: (num_envs,  ...)
            return jax.vmap(lambda idx: leaf[idx])(agent_indices)
        return jax.tree.map(gather_leaf, pop_params)
    
    def get_actions(self, pop_params, agent_indices, obs, done, avail_actions, hstate, rng):
        '''
        Get the actions of the agents specified by agent_indices.
        
        Args:
            pop_params: pytree of parameters for the population of agents of shape (pop_size, ...).
            agent_indices: indices with shape (num_envs,), each in [0, pop_size)
            obs: observations with shape (num_envs, ...) 
            done: done flags with shape (num_envs,)
            avail_actions: available actions with shape (num_envs, num_actions)
            hstate: hidden state with shape (num_envs, ...) or None if policy doesn't use hidden state
            rng: random key
            
        Returns:
            actions: actions with shape (num_envs,)
            new_hstate: new hidden state with shape (num_envs, ...) or None
        '''
        gathered_params = self.gather_agent_params(pop_params, agent_indices)
        num_envs = agent_indices.squeeze().shape[0]
        rngs_batched = jax.random.split(rng, num_envs)
        actions, new_hstate = jax.vmap(self.policy_cls.get_action)(
            gathered_params, obs, done, avail_actions, hstate, 
            rngs_batched)
        return actions, new_hstate
    
    def init_hstate(self, n: int):
        '''Initialize the hidden state for n members of the population.'''
        return self.policy_cls.init_hstate(n)

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
    def get_action(self, params, obs, done, avail_actions, hstate, rng) -> Tuple[int, chex.Array]:
        """
        Only computes an action given an observation, done flag, available actions, hidden state, and random key.

        Args:
            params (dict): The parameters of the policy.
            obs (chex.Array): The observation.
            done (chex.Array): The done flag.
            avail_actions (chex.Array): The available actions.
            hstate (chex.Array): The hidden state.
            key (jax.random.PRNGKey): The random key.

        Returns:
            Tuple[int, chex.Array]: A tuple containing the action and the new hidden state.
        """
        pass

    @partial(jax.jit, static_argnums=(0,))
    def get_action_value_policy(self, params, obs, done, avail_actions, hstate, rng) -> Tuple[int, chex.Array, chex.Array, chex.Array]:
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

        Returns:
            Tuple[int, chex.Array, chex.Array, chex.Array]: 
                A tuple containing the action, value, policy, and new hidden state.
        """
        pass

    def init_hstate(self, batch_size) -> chex.Array:
        """Initialize the hidden state for the policy."""
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
        self.activation = activation
        self.network = ActorCritic(action_dim, activation=activation)
    
    @partial(jax.jit, static_argnums=(0,))
    def get_action(self, params, obs, done, avail_actions, hstate, rng):
        """Get actions for the MLP policy."""
        pi, _ = self.network.apply(params, (obs, avail_actions))
        action = pi.sample(seed=rng)
        return action, None  # no hidden state
    
    @partial(jax.jit, static_argnums=(0,))
    def get_action_value_policy(self, params, obs, done, avail_actions, hstate, rng):
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
        self.activation = activation
        # Initialize the network class once
        self.network = ActorWithDoubleCritic(action_dim, activation=activation)
    
    @partial(jax.jit, static_argnums=(0,))
    def get_action(self, params, obs, done, avail_actions, hstate, rng):
        """Get actions for the policy with double critics."""
        pi, _, _ = self.network.apply(params, (obs, avail_actions))
        action = pi.sample(seed=rng)
        return action, None  # no hidden state
    
    @partial(jax.jit, static_argnums=(0,))
    def get_action_value_policy(self, params, obs, done, avail_actions, hstate, rng):
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
        self.activation = activation
        self.fc_hidden_dim = fc_hidden_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.network = RNNActorCritic(
            action_dim, 
            fc_hidden_dim=fc_hidden_dim,
            gru_hidden_dim=gru_hidden_dim,
            activation=activation
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def get_action(self, params, obs, done, avail_actions, hstate, rng):
        """Get actions for the RNN policy. 
        The first dim of obs and done should be the time dimension."""
        new_hstate, pi, _ = self.network.apply(params, hstate, (obs, done, avail_actions))
        action = pi.sample(seed=rng)
        return action, new_hstate
    
    @partial(jax.jit, static_argnums=(0,))
    def get_action_value_policy(self, params, obs, done, avail_actions, hstate, rng):
        """Get actions, values, and policy for the RNN policy.
        The first dim of obs and done should be the time dimension."""
        new_hstate, pi, val = self.network.apply(params, hstate, (obs, done, avail_actions))
        action = pi.sample(seed=rng)
        return action, val, pi, new_hstate
    
    def init_hstate(self, batch_size):
        """Initialize hidden state for the RNN policy."""
        return ScannedRNN.initialize_carry(batch_size, self.gru_hidden_dim)
    
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
        return self.network.init(rng, init_hstate, dummy_x)


class S5ActorCriticPolicy(AgentPolicy):
    """Policy wrapper for S5 Actor-Critic"""
    
    def __init__(self, action_dim, obs_dim, 
                 d_model=16, ssm_size=16, 
                 n_layers=2, blocks=1,
                 fc_hidden_dim=64,
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
        self.n_layers = n_layers
        self.blocks = blocks
        self.fc_hidden_dim = fc_hidden_dim
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
            ssm_hidden_dim=self.ssm_size,
            s5_d_model=self.d_model,
            s5_n_layers=self.n_layers,
            s5_activation=self.s5_activation,
            s5_do_norm=self.s5_do_norm, 
            s5_prenorm=self.s5_prenorm,
            s5_do_gtrxl_norm=self.s5_do_gtrxl_norm,
            s5_no_reset=self.s5_no_reset
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def get_action(self, params, obs, done, avail_actions, hstate, rng):
        """Get actions for the S5 policy."""
        new_hstate, pi, _ = self.network.apply(params, hstate, (obs, done, avail_actions))
        action = pi.sample(seed=rng)
        return action, new_hstate
    
    @partial(jax.jit, static_argnums=(0,))
    def get_action_value_policy(self, params, obs, done, avail_actions, hstate, rng):
        """Get actions, values, and policy for the S5 policy."""
        new_hstate, pi, val = self.network.apply(params, hstate, (obs, done, avail_actions))
        action = pi.sample(seed=rng)
        return action, val, pi, new_hstate
    
    def init_hstate(self, batch_size):
        """Initialize hidden state for the S5 policy."""
        return StackedEncoderModel.initialize_carry(batch_size, self.ssm_size // 2, self.n_layers)
    
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
