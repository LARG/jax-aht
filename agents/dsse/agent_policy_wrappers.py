'''Wrap DSSE heuristic agent policies in the AgentPolicy interface so they can
be used as held-out partners by evaluation/heldout_evaluator.py. Mirrors the
pattern used by agents/lbf/agent_policy_wrappers.py and
agents/overcooked/agent_policy_wrappers.py.
'''
import jax
from agents.agent_interface import AgentPolicy
from agents.dsse.random_agent import RandomAgent
from agents.dsse.greedy_search_agent import GreedySearchAgent
from agents.dsse.sweep_agent import SweepAgent


class DSSERandomPolicyWrapper(AgentPolicy):
    """Random DSSE agent. Stateless other than agent_id."""

    def __init__(self, using_log_wrapper: bool = False):
        self.policy = RandomAgent()
        self.using_log_wrapper = using_log_wrapper

    def get_action(self, params, obs, done, avail_actions, hstate, rng,
                   env_state, aux_obs=None, test_mode=False):
        if self.using_log_wrapper:
            env_state = env_state.env_state
        # run_episodes passes obs with shape (1, 1, obs_dim); the DSSE
        # heuristic agents expect a flat (obs_dim,) vector.
        obs = obs.reshape(-1)
        action, new_hstate = self.policy.get_action(obs, env_state, hstate, rng)
        return action, new_hstate

    def init_hstate(self, batch_size: int, aux_info=None):
        return self.policy.init_agent_state(aux_info["agent_id"])


class DSSEGreedySearchPolicyWrapper(AgentPolicy):
    """Greedy DSSE agent that walks toward the highest probability cell and
    searches when on top of it."""

    def __init__(self, grid_size: int = 7, using_log_wrapper: bool = False):
        self.policy = GreedySearchAgent(grid_size=grid_size)
        self.using_log_wrapper = using_log_wrapper

    def get_action(self, params, obs, done, avail_actions, hstate, rng,
                   env_state, aux_obs=None, test_mode=False):
        if self.using_log_wrapper:
            env_state = env_state.env_state
        obs = obs.reshape(-1)
        action, new_hstate = self.policy.get_action(obs, env_state, hstate, rng)
        new_hstate = jax.lax.cond(
            done.squeeze(),
            lambda: self.policy.init_agent_state(hstate.agent_id),
            lambda: new_hstate,
        )
        return action, new_hstate

    def init_hstate(self, batch_size: int, aux_info=None):
        return self.policy.init_agent_state(aux_info["agent_id"])


class DSSESweepPolicyWrapper(AgentPolicy):
    """Boustrophedon sweep agent that searches every cell in row-major order."""

    def __init__(self, grid_size: int = 7, using_log_wrapper: bool = False):
        self.policy = SweepAgent(grid_size=grid_size)
        self.using_log_wrapper = using_log_wrapper

    def get_action(self, params, obs, done, avail_actions, hstate, rng,
                   env_state, aux_obs=None, test_mode=False):
        if self.using_log_wrapper:
            env_state = env_state.env_state
        obs = obs.reshape(-1)
        action, new_hstate = self.policy.get_action(obs, env_state, hstate, rng)
        new_hstate = jax.lax.cond(
            done.squeeze(),
            lambda: self.policy.init_agent_state(hstate.agent_id),
            lambda: new_hstate,
        )
        return action, new_hstate

    def init_hstate(self, batch_size: int, aux_info=None):
        return self.policy.init_agent_state(aux_info["agent_id"])
