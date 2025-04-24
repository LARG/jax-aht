from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct
from jumanji.environments.routing.lbf.types import Agent, Food, State as LBFState
from envs.jumanji_jaxmarl_wrapper import WrappedEnvState


@struct.dataclass
class AgentState:
    rng_key: jax.random.PRNGKey

class BaseAgent:
    """A base heuristic agent for the LBF environment.
    
    Agent ideas: 
    - Agent that goes to the closest fruit
    - Agent that goes to the farthest fruit
    - Agent that goes to the center of the grid
    """

    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.initial_state = AgentState(rng_key=jax.random.PRNGKey(self.agent_id))

    def get_name(self):
        return self.__class__.__name__

    @partial(jax.jit, static_argnums=(0,))
    def get_action(self, obs: jnp.ndarray, 
                   env_state: WrappedEnvState, 
                   agent_state: AgentState=None) -> Tuple[int, AgentState]:
        """Get action and updated state based on observation and current state.
        
        Args:
            obs: Flattened observation array
            env_state: WrappedEnvState containing the LBF environment state
            agent_state: AgentState containing agent's internal state
            
        Returns:
            action, AgentState
        """
        lbf_env_state = env_state.env_state # extract LBFEnvState from the Jumanji wrapped env state
        action, agent_state = self._get_action(obs, lbf_env_state, agent_state)
        return action, agent_state
