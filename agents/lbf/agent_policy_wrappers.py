'''Wrap heuristic agent policies in AgentPolicy interface.
TODO: clean up logic by vectorizing init_hstate. See HeuristicPolicyPopulation.
'''
import os

import jax
import jax.numpy as jnp
import yaml

from agents.agent_interface import AgentPolicy
from agents.bc import BCLSTMAgent, BCLSTMConfig
from agents.lbf.random_agent import RandomAgent
from agents.lbf.sequential_fruit_agent import SequentialFruitAgent
from agents.lbf.entitled_agent import EntitledAgent
from agents.lbf.greedy_heuristic_agent import GreedyHeuristicAgent
from common.save_load_utils import REPO_PATH


class LBFBCLSTMPolicyWrapper(AgentPolicy):
    """Load an LBF BC-LSTM checkpoint and adapt observations for evaluation."""

    def __init__(self, weight_file: str, greedy: bool = True):
        yaml_path = weight_file.rsplit('.', 1)[0] + '.yaml'
        abs_yaml_path = (
            yaml_path if os.path.isabs(yaml_path)
            else os.path.join(REPO_PATH, yaml_path)
        )
        with open(abs_yaml_path) as config_file:
            saved_config = yaml.safe_load(config_file)

        config = BCLSTMConfig(
            obs_dim=saved_config["obs_dim"],
            action_dim=saved_config["action_dim"],
            preprocess_dim=saved_config.get("preprocess_dim", 1024),
            lstm_dim=saved_config.get("lstm_dim", 512),
            postprocess_dim=saved_config.get("postprocess_dim", 256),
            dropout_rate=saved_config.get("dropout_rate", 0.0),
        )
        super().__init__(config.action_dim, config.obs_dim)
        self.agent = BCLSTMAgent(config, weight_path=weight_file)
        self.greedy = greedy

    def _prepare_obs(self, obs):
        obs = jnp.asarray(obs).reshape(-1)
        if obs.shape[0] > self.obs_dim:
            raise ValueError(
                f"BC-LSTM expects at most {self.obs_dim} observation values, "
                f"received {obs.shape[0]}."
            )
        if obs.shape[0] < self.obs_dim:
            obs = jnp.pad(obs, (0, self.obs_dim - obs.shape[0]))
        return obs

    def get_action(self, params, obs, done, avail_actions, hstate, rng,
                   aux_obs=None, env_state=None, test_mode=False):
        del params, env_state, aux_obs
        obs = self._prepare_obs(obs)
        if avail_actions is None:
            legal_mask = jnp.ones(self.action_dim, dtype=jnp.float32)
        else:
            legal_mask = jnp.asarray(avail_actions).reshape(-1).astype(jnp.float32)

        done = jnp.asarray(done).reshape(-1)[0].astype(bool)
        hstate = jax.lax.cond(
            done,
            lambda: self.agent.init_carry(),
            lambda: hstate,
        )
        if self.greedy or test_mode:
            carry, action = self.agent.greedy_act(hstate, obs, legal_mask)
        else:
            carry, action = self.agent.sample_act(hstate, obs, legal_mask, rng)
        return action, carry

    def init_hstate(self, batch_size: int, aux_info=None):
        del batch_size, aux_info
        return self.agent.init_carry()



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
