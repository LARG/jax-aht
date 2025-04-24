'''Wrap heuristic agent policies in AgentPolicy interface.'''
from agents.agent_interface import AgentPolicy
from agents.lbf.random_agent import RandomAgent


class LBFRandomPolicyWrapper(AgentPolicy):
    def __init__(self, agent_id: int = 0):
        self.policy = RandomAgent(agent_id) # agent id doesn't matter for the random agent

    def get_action(self, params, obs, done, avail_actions, hstate, rng, 
                   env_state, aux_obs=None, test_mode=False):
        # hstate represents the agent state
        action, new_hstate =  self.policy.get_action(obs, env_state, hstate)
        return action, new_hstate

    def init_hstate(self, batch_size: int):
        return self.policy.initial_state