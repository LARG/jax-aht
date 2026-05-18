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