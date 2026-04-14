from functools import partial
from typing import Tuple, List

import jax
import jax.numpy as jnp
from flax import struct

from envs.base_env import WrappedEnvState


@struct.dataclass
class AgentState:
    agent_id: int


class BaseAgent:

    def __init__(self, num_actions: int = 20,
                 agent_names: List[str] = None):
        self.num_actions = num_actions
        self.agent_names = agent_names or ['agent_0', 'agent_1']

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
        hanabi_state = env_state.env_state
        # stack avail_actions dict into (num_agents, num_actions) and index by id
        avail_dict = env_state.avail_actions
        avail_array = jnp.stack(
            [avail_dict[name] for name in self.agent_names]
        )
        avail_mask = avail_array[agent_state.agent_id]
        action, agent_state = self._get_action(
            obs, hanabi_state, avail_mask, agent_state, rng
        )
        return action, agent_state
