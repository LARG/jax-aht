from functools import partial

import jax
import jax.numpy as jnp

from agents.agent_interface import AgentPolicy
from agents.s5_actor_critic_agent import init_S5SSM, make_DPLR_HiPPO, StackedEncoderModel
from agents.s5_q_network import S5QNetwork
from agents.rnn_actor_critic import ScannedRNN


class S5DQNActorCriticFQEPolicy(AgentPolicy):
    """Policy wrapper for DRQN Actor-Critic Fitted Q Estimation"""

    def __init__(self, action_dim, obs_dim, actor_critic_policy, epsilon_start=1.0, epsilon_finish=0.1, epsilon_anneal_time=10000,
                 d_model=16, ssm_size=16,
                 ssm_n_layers=2, blocks=1,
                 s5_activation="full_glu",
                 s5_do_norm=True,
                 s5_prenorm=True,
                 s5_do_gtrxl_norm=True,
                 s5_no_reset=False):
        """
        Args:
            action_dim: int, dimension of the action space
            obs_dim: int, dimension of the observation space
            actor_critic_policy: nn.Module, the actor-critic network to use for value estimation
            epsilon_start: float, initial epsilon for epsilon-greedy exploration
            epsilon_finish: float, final epsilon for epsilon-greedy exploration
            epsilon_anneal_time: int, number of timesteps over which to anneal epsilon
            d_model: int, dimension of the model
            ssm_size: int, size of the SSM
            n_layers: int, number of S5 layers
            blocks: int, number of blocks to split SSM parameters
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
        self.network = S5QNetwork(
            action_dim,
            ssm_init_fn=self.ssm_init_fn,
            ssm_hidden_dim=self.ssm_size,
            s5_d_model=self.d_model,
            s5_n_layers=self.ssm_n_layers,
            s5_activation=self.s5_activation,
            s5_do_norm=self.s5_do_norm,
            s5_prenorm=self.s5_prenorm,
            s5_do_gtrxl_norm=self.s5_do_gtrxl_norm,
            s5_no_reset=self.s5_no_reset
        )

        self.actor_critic = actor_critic_policy
        self.epsilon_start = epsilon_start
        self.epsilon_finish = epsilon_finish
        self.epsilon_anneal_time = epsilon_anneal_time

    def eps_greedy_exploration(self, rng, q_vals, t):
        """Epsilon-greedy exploration strategy."""
        # Keys for sampling random actions and picking actions
        rng_a, rng_e = jax.random.split(rng, 2)

        # Compute epsilon for the current timestep
        eps = jnp.clip(
            ((self.epsilon_finish - self.epsilon_start) / self.epsilon_anneal_time)
            * t + self.epsilon_start, self.epsilon_finish,
        )

        # Greedy argmax actions from the Q-values
        greedy_actions = jnp.argmax(q_vals, axis=-1)
        chosen_actions = jnp.where(
            # Pick the actions that should be random
            jax.random.uniform(rng_e, greedy_actions.shape) < eps,
            # Sample random actions uniformly from the action space
            jax.random.randint(rng_a, shape=greedy_actions.shape, minval=0, maxval=q_vals.shape[-1]),
            greedy_actions,
        )
        return chosen_actions


    def eps_greedy_actor_critic_exploration(self, rng, actions, avail_actions, t):
        """Epsilon-greedy exploration strategy."""
        # Keys for sampling random actions and picking actions
        rng_a, rng_e = jax.random.split(rng, 2)

        # Compute epsilon for the current timestep
        eps = jnp.clip(
            ((self.epsilon_finish - self.epsilon_start) / self.epsilon_anneal_time)
            * t + self.epsilon_start, self.epsilon_finish,
        )

        # Sample random actions only from available actions
        # Generate random scores for all actions, mask unavailable ones
        random_scores = jax.random.uniform(rng_a, shape=avail_actions.shape)
        masked_scores = jnp.where(avail_actions, random_scores, -jnp.inf)
        random_actions = jnp.argmax(masked_scores, axis=-1)

        chosen_actions = jnp.where(
            # Pick the actions that should be random
            jax.random.uniform(rng_e, actions.shape) < eps,
            # Use random available actions
            random_actions,
            actions,
        )
        return chosen_actions

    @partial(jax.jit, static_argnums=(0,))
    def get_action(self, params, obs, done, avail_actions, hstate, rng,
                   aux_obs=None, env_state=None, test_mode=False):
        """Get actions for the DRQN policy."""
        new_hstate, qvals = self.network.apply(params, hstate, (obs, done))

        # Mask out unavailable actions by setting their Q-values to -inf
        masked_qvals = jnp.where(avail_actions, qvals, -jnp.inf)

        actions = jax.lax.cond(test_mode,
                               lambda: jnp.argmax(masked_qvals, axis=-1),
                               lambda: self.eps_greedy_exploration(rng, masked_qvals, env_state.env_state.env_state.timestep))
        return actions, new_hstate

    @partial(jax.jit, static_argnums=(0,))
    def get_actor_critic_action(self, params, obs, done, avail_actions, hstate, rng,
                   aux_obs=None, env_state=None, test_mode=False):
        """Get actions for the DRQN policy."""
        actions, new_ac_hstate = self.actor_critic.get_action(params, obs, done, avail_actions, hstate, rng,
                                                              aux_obs, env_state, test_mode=False)
        actions = jax.lax.stop_gradient(actions)
        new_ac_hstate = jax.lax.stop_gradient(new_ac_hstate)

        actions = jax.lax.cond(test_mode,
                               lambda: actions,
                               lambda: self.eps_greedy_actor_critic_exploration(rng, actions,
                                                                                avail_actions,
                                                                                env_state.env_state.env_state.timestep))

        return actions, new_ac_hstate

    @partial(jax.jit, static_argnums=(0,))
    def get_action_value_policy(self, params, obs, done, avail_actions, hstate, rng,
                                aux_obs=None, env_state=None):
        """Get actions, values, and policy for the DRQN policy."""
        new_hstate, qvals = self.network.apply(params, hstate, (obs, done))

        # Mask out unavailable actions by setting their Q-values to -inf
        masked_qvals = jnp.where(avail_actions, qvals, -jnp.inf)

        # Greedy argmax actions from the Q-values
        actions = jnp.argmax(masked_qvals, axis=-1)

        return actions, qvals, None, new_hstate # no policy

    def init_hstate(self, batch_size, aux_info=None):
        """Initialize hidden state for the DRQN policy."""
        hstate =  StackedEncoderModel.initialize_carry(batch_size, self.ssm_size // 2, self.ssm_n_layers)
        # actor_critic_hstate = self.actor_critic.init_hstate(batch_size, aux_info)
        return hstate #, actor_critic_hstate

    def init_params(self, rng):
        """Initialize parameters for the DRQN policy."""
        batch_size = 1
        # Initialize hidden state
        init_hstate = self.init_hstate(batch_size)

        # Create dummy inputs - add time dimension
        dummy_obs = jnp.zeros((1, batch_size, self.obs_dim))
        dummy_done = jnp.zeros((1, batch_size))
        dummy_x = (dummy_obs, dummy_done)

        # Initialize model
        return self.network.init(rng, init_hstate, dummy_x)
