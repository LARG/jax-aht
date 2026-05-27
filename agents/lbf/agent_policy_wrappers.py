'''Wrap heuristic agent policies in AgentPolicy interface.
TODO: clean up logic by vectorizing init_hstate. See HeuristicPolicyPopulation.
'''
from functools import partial

import distrax
import jax
import jax.numpy as jnp

from agents.agent_interface import AgentPolicy
from agents.bc.bc_lstm import BCLSTMConfig, BCLSTMNetwork
from agents.lbf.random_agent import RandomAgent
from agents.lbf.sequential_fruit_agent import SequentialFruitAgent
from agents.lbf.entitled_agent import EntitledAgent
from agents.lbf.greedy_heuristic_agent import GreedyHeuristicAgent


class LBFBCLSTMPolicyWrapper(AgentPolicy):
    """Adapter that lets LBF BC-LSTM checkpoints run through JaxAHT evaluation."""

    def __init__(self, config: BCLSTMConfig):
        super().__init__(config.action_dim, config.obs_dim)
        self.config = config
        self.network = BCLSTMNetwork(
            action_dim=config.action_dim,
            preprocess_dim=config.preprocess_dim,
            lstm_dim=config.lstm_dim,
            postprocess_dim=config.postprocess_dim,
            dropout_rate=config.dropout_rate,
        )

    def init_hstate(self, batch_size: int, aux_info=None):
        shape = (batch_size, self.config.lstm_dim)
        return (jnp.zeros(shape), jnp.zeros(shape))

    def _preprocess_obs(self, obs):
        if self.config.lbf_feature_mode == "path":
            from human_data_processing.lbf_features import augment_lbf_obs
            if self.config.lbf_grid_size <= 0 or self.config.lbf_num_food <= 0:
                raise ValueError(
                    "LBF feature mode requires lbf_grid_size and lbf_num_food"
                )
            return augment_lbf_obs(
                obs,
                grid_size=self.config.lbf_grid_size,
                num_food=self.config.lbf_num_food,
            )
        return obs

    @partial(jax.jit, static_argnums=(0,))
    def get_action(self, params, obs, done, avail_actions, hstate, rng,
                   aux_obs=None, env_state=None, test_mode=False):
        obs = self._preprocess_obs(obs)
        obs_shape = obs.shape[:-1]
        obs_flat = obs.reshape((-1, obs.shape[-1]))
        if obs_flat.shape[-1] > self.config.obs_dim:
            raise ValueError(
                f"Policy obs_dim={self.config.obs_dim} cannot handle "
                f"obs_dim={obs_flat.shape[-1]}"
            )
        if obs_flat.shape[-1] < self.config.obs_dim:
            obs_flat = jnp.pad(
                obs_flat,
                [(0, 0), (0, self.config.obs_dim - obs_flat.shape[-1])],
            )
        avail_flat = avail_actions.reshape((-1, self.config.action_dim))
        done_flat = done.reshape((-1, 1))

        reset_hstate = self.init_hstate(obs_flat.shape[0])
        hstate = jax.tree.map(
            lambda reset, current: jnp.where(done_flat, reset, current),
            reset_hstate,
            hstate,
        )

        hstate, logits = self.network.apply({'params': params}, hstate, obs_flat)
        logits = jnp.where(avail_flat > 0, logits, -1e9)
        pi = distrax.Categorical(logits=logits)
        action = jax.lax.cond(
            test_mode,
            lambda: pi.mode(),
            lambda: pi.sample(seed=rng),
        )
        return action.reshape(obs_shape), hstate


class LBFRandomPolicyWrapper(AgentPolicy):
    def __init__(self):
        self.policy = RandomAgent() # agent id doesn't matter for the random agent

    def get_action(self, params, obs, done, avail_actions, hstate, rng, 
                   env_state, aux_obs=None, test_mode=False):
        # hstate represents the agent state
        action, new_hstate =  self.policy.get_action(obs, env_state, hstate, rng)
        return action, new_hstate

    def init_hstate(self, batch_size: int, aux_info=None):
        """Initialize the hidden state for the random agent."""
        return self.policy.init_agent_state(aux_info["agent_id"])


class LBFSequentialFruitPolicyWrapper(AgentPolicy):
    """Policy wrapper for the SequentialFruitAgent that visits fruits in a predetermined order."""
    def __init__(self, grid_size: int = 7, num_fruits: int = 3, 
                 ordering_strategy: str = 'lexicographic', using_log_wrapper: bool = False):
        self.policy = SequentialFruitAgent(grid_size, num_fruits, ordering_strategy)
        self.using_log_wrapper = using_log_wrapper

    def get_action(self, params, obs, done, avail_actions, hstate, rng, 
                   env_state, aux_obs=None, test_mode=False):
        # hstate represents the agent state
        if self.using_log_wrapper:
            env_state = env_state.env_state
        action, new_hstate = self.policy.get_action(obs, env_state, hstate, rng)
        # if done, reset the hstate
        new_hstate = jax.lax.cond(done.squeeze(), 
                                  lambda: self.policy.init_agent_state(hstate.agent_id),
                                  lambda: new_hstate)
        return action, new_hstate

    def init_hstate(self, batch_size: int, aux_info):
        return self.policy.init_agent_state(aux_info["agent_id"])


class LBFEntitledPolicyWrapper(AgentPolicy):
    """Policy wrapper for the EntitledAgent that waits for its teammate before collecting fruit."""
    def __init__(self, grid_size: int = 7, num_fruits: int = 3, using_log_wrapper: bool = False):
        self.policy = EntitledAgent(grid_size, num_fruits)
        self.using_log_wrapper = using_log_wrapper

    def get_action(self, params, obs, done, avail_actions, hstate, rng,
                   env_state, aux_obs=None, test_mode=False):
        # hstate represents the agent state
        if self.using_log_wrapper:
            env_state = env_state.env_state
        action, new_hstate = self.policy.get_action(obs, env_state, hstate, rng)
        # if done, reset the hstate
        new_hstate = jax.lax.cond(done.squeeze(),
                                  lambda: self.policy.init_agent_state(hstate.agent_id),
                                  lambda: new_hstate)
        return action, new_hstate

    def init_hstate(self, batch_size: int, aux_info):
        return self.policy.init_agent_state(aux_info["agent_id"])


class LBFGreedyHeuristicPolicyWrapper(AgentPolicy):
    """Policy wrapper for the GreedyHeuristicAgent that greedily targets fruit based on a heuristic."""
    def __init__(self, grid_size: int = 7, num_fruits: int = 3,
                 heuristic: str = 'closest_self', using_log_wrapper: bool = False):
        self.policy = GreedyHeuristicAgent(grid_size, num_fruits, heuristic)
        self.using_log_wrapper = using_log_wrapper

    def get_action(self, params, obs, done, avail_actions, hstate, rng,
                   env_state, aux_obs=None, test_mode=False):
        # hstate represents the agent state
        if self.using_log_wrapper:
            env_state = env_state.env_state
        action, new_hstate = self.policy.get_action(obs, env_state, hstate, rng)
        # if done, reset the hstate
        new_hstate = jax.lax.cond(done.squeeze(),
                                  lambda: self.policy.init_agent_state(hstate.agent_id),
                                  lambda: new_hstate)
        return action, new_hstate

    def init_hstate(self, batch_size: int, aux_info):
        return self.policy.init_agent_state(aux_info["agent_id"])
