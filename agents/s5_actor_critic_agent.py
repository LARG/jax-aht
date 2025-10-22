from functools import partial

import jax
import jax.numpy as jnp

from agents.agent_interface import AgentPolicy
from agents.s5_actor_critic import S5ActorCritic, StackedEncoderModel, init_S5SSM, make_DPLR_HiPPO


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
