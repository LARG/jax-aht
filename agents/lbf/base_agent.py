from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct
from jumanji.environments.routing.lbf.types import Agent, Food, State as LBFState
from envs.lbf.lbf_wrapper import WrappedEnvState


@struct.dataclass
class AgentState:
    agent_id: int

class BaseAgent:
    """A base heuristic agent for the LBF environment.
    """
    def __init__(self):
        pass

    def init_agent_state(self, agent_id: int) -> AgentState:
        return AgentState(agent_id=agent_id)

    def get_name(self):
        return self.__class__.__name__

    @partial(jax.jit, static_argnums=(0,))
    def get_action(self, obs: jnp.ndarray, 
                   env_state: WrappedEnvState, 
                   agent_state: AgentState=None, 
                   rng: jax.random.PRNGKey=None) -> Tuple[int, AgentState]:
        """Get action and updated state based on observation and current state.
        
        Args:
            obs: Flattened observation array
            env_state: WrappedEnvState containing the LBF environment state
            agent_state: AgentState containing agent's internal state
            rng: jax.random.PRNGKey for any stochasticity. The rng key is not returned,
                so the user should split the key before passing it to an agent. 
        Returns:
            action, AgentState
        """
        lbf_env_state = env_state.env_state # extract LBFEnvState from the Jumanji wrapped env state
        action, agent_state = self._get_action(obs, lbf_env_state, agent_state, rng)
        return action, agent_state
