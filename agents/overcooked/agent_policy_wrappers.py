'''Wrap heuristic agent policies in AgentPolicy interface.'''
from agents.agent_interface import AgentPolicy
from agents.overcooked.independent_agent import IndependentAgent
from agents.overcooked.onion_agent import OnionAgent
from agents.overcooked.plate_agent import PlateAgent
from agents.overcooked.static_agent import StaticAgent
from agents.overcooked.random_agent import RandomAgent


class OvercookedIndependentPolicyWrapper(AgentPolicy):
    """Policy wrapper for the Independent heuristic agent."""
    def __init__(self, layout, agent_id=0, using_log_wrapper=False, 
                 p_onion_on_counter=0, p_plate_on_counter=0.5):
        super().__init__(action_dim=6, obs_dim=None)  # Action dim 6 for Overcooked
        self.policy = IndependentAgent(agent_id, layout, p_onion_on_counter, p_plate_on_counter)
        self.using_log_wrapper = using_log_wrapper

    
    def get_action(self, params, obs, done, avail_actions, hstate, rng, 
                   aux_obs=None, env_state=None, test_mode=False):
        if self.using_log_wrapper:
            env_state = env_state.env_state
        # hstate represents the agent state
        action, new_hstate = self.policy.get_action(obs, env_state, hstate)
        return action, new_hstate

    def init_hstate(self, batch_size):
        return self.policy.initial_state


class OvercookedOnionPolicyWrapper(AgentPolicy):
    """Policy wrapper for the Onion heuristic agent."""
    def __init__(self, layout, agent_id=0, p_onion_on_counter=0.1, using_log_wrapper=False):
        super().__init__(action_dim=6, obs_dim=None)
        self.policy = OnionAgent(agent_id, layout, p_onion_on_counter)
        self.using_log_wrapper = using_log_wrapper

    def get_action(self, params, obs, done, avail_actions, hstate, rng, 
                   aux_obs=None, env_state=None, test_mode=False):
        if self.using_log_wrapper:
            env_state = env_state.env_state
        action, new_hstate = self.policy.get_action(obs, env_state, hstate)
        return action, new_hstate

    def init_hstate(self, batch_size):
        return self.policy.initial_state


class OvercookedPlatePolicyWrapper(AgentPolicy):
    """Policy wrapper for the Plate heuristic agent."""
    def __init__(self, layout, agent_id=0, p_plate_on_counter=0.1, using_log_wrapper=False):
        super().__init__(action_dim=6, obs_dim=None)
        self.policy = PlateAgent(agent_id, layout, p_plate_on_counter)
        self.using_log_wrapper = using_log_wrapper

    def get_action(self, params, obs, done, avail_actions, hstate, rng, 
                   aux_obs=None, env_state=None, test_mode=False):
        if self.using_log_wrapper:
            env_state = env_state.env_state
        action, new_hstate = self.policy.get_action(obs, env_state, hstate)
        return action, new_hstate

    def init_hstate(self, batch_size):
        return self.policy.initial_state


class OvercookedStaticPolicyWrapper(AgentPolicy):
    """Policy wrapper for the Static heuristic agent."""
    def __init__(self, layout, agent_id=0, using_log_wrapper=False):
        super().__init__(action_dim=6, obs_dim=None)
        self.policy = StaticAgent(agent_id, layout)
        self.using_log_wrapper = using_log_wrapper

    def get_action(self, params, obs, done, avail_actions, hstate, rng, 
                   aux_obs=None, env_state=None, test_mode=False):
        if self.using_log_wrapper:
            env_state = env_state.env_state
        action, new_hstate = self.policy.get_action(obs, env_state, hstate)
        return action, new_hstate

    def init_hstate(self, batch_size):
        return self.policy.initial_state


class OvercookedRandomPolicyWrapper(AgentPolicy):
    """Policy wrapper for the Random heuristic agent."""
    def __init__(self, layout, agent_id=0, using_log_wrapper=False):
        super().__init__(action_dim=6, obs_dim=None)
        self.policy = RandomAgent(agent_id, layout)
        self.using_log_wrapper = using_log_wrapper

    def get_action(self, params, obs, done, avail_actions, hstate, rng, 
                   aux_obs=None, env_state=None, test_mode=False):
        if self.using_log_wrapper:
            env_state = env_state.env_state
        action, new_hstate = self.policy.get_action(obs, env_state, hstate)
        return action, new_hstate

    def init_hstate(self, batch_size):
        return self.policy.initial_state