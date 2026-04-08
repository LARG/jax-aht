from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct

from envs.base_env import WrappedEnvState
from envs.dsse.dsse_jax import DSSEState


@struct.dataclass
class AgentState:
    agent_id: int
    step_count: int = 0


class BaseAgent:
    """Base heuristic agent for the DSSE environment."""

    def __init__(self):
        pass

    def init_agent_state(self, agent_id: int) -> AgentState:
        return AgentState(agent_id=agent_id)

    def get_name(self):
        return self.__class__.__name__

    @partial(jax.jit, static_argnums=(0,))
    def get_action(
        self,
        obs: jnp.ndarray,
        env_state: WrappedEnvState,
        agent_state: AgentState = None,
        rng: jax.random.PRNGKey = None,
    ) -> Tuple[int, AgentState]:
        dsse_state = env_state.env_state
        action, agent_state = self._get_action(obs, dsse_state, agent_state, rng)
        return action, agent_state
