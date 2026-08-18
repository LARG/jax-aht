'''Wrap heuristic agent policies in AgentPolicy interface.
TODO: clean up logic by vectorizing init_hstate. See HeuristicPolicyPopulation.
'''
import jax
from agents.agent_interface import AgentPolicy
from agents.lbf.random_agent import RandomAgent
from agents.lbf.sequential_fruit_agent import SequentialFruitAgent
from agents.lbf.entitled_agent import EntitledAgent
from agents.lbf.greedy_heuristic_agent import GreedyHeuristicAgent



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

class LBFBCLSTMPolicyWrapper(AgentPolicy):
    """BC-LSTM human proxy for LBF, loaded from a safetensors weight file with a
    sibling .yaml describing the architecture (see agents/bc)."""

    def __init__(self, weight_file: str, using_log_wrapper: bool = False,
                 greedy: bool = True):
        import os, yaml
        from common.save_load_utils import REPO_PATH
        from agents.bc import BCLSTMAgent, BCLSTMConfig

        yaml_path = weight_file.rsplit('.', 1)[0] + '.yaml'
        abs_yaml = yaml_path if os.path.isabs(yaml_path) else os.path.join(REPO_PATH, yaml_path)
        with open(abs_yaml) as f:
            cfg = yaml.safe_load(f)

        config = BCLSTMConfig(
            obs_dim=cfg['obs_dim'], action_dim=cfg['action_dim'],
            preprocess_dim=cfg.get('preprocess_dim', 1024),
            lstm_dim=cfg.get('lstm_dim', 512),
            postprocess_dim=cfg.get('postprocess_dim', 256),
            dropout_rate=cfg.get('dropout_rate', 0.0),
        )
        self.agent = BCLSTMAgent(config, weight_path=weight_file)
        self.action_dim = cfg['action_dim']
        self.using_log_wrapper = using_log_wrapper
        self.greedy = greedy

    def get_action(self, params, obs, done, avail_actions, hstate, rng,
                   env_state, aux_obs=None, test_mode=False):
        import jax.numpy as jnp
        obs_flat = obs.reshape(-1)

        if avail_actions is not None and avail_actions.ndim >= 1:
            legal_mask = avail_actions.reshape(-1).astype(jnp.float32)
        else:
            legal_mask = jnp.ones(self.action_dim, dtype=jnp.float32)

        if self.greedy or test_mode:
            carry, action = self.agent.greedy_act(hstate, obs_flat, legal_mask)
        else:
            carry, action = self.agent.sample_act(hstate, obs_flat, legal_mask, rng)

        carry = jax.lax.cond(
            done.squeeze().astype(bool),
            lambda: self.agent.init_carry(),
            lambda: carry,
        )
        return action, carry

    def init_hstate(self, batch_size: int, aux_info=None):
        return self.agent.init_carry()
