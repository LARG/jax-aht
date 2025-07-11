from functools import partial
from typing import Tuple, Any

import jax
import jax.numpy as jnp
from flax.struct import dataclass
from envs.coins.coins_wrapper import WrappedEnvState

@dataclass
class AgentState:
    # Add per-agent state fields here if needed
    pass

class BaseAgent:
    """A base agent for the Coins environment."""
    def __init__(self):
        pass

    def init_agent_state(self) -> AgentState:
        return AgentState()

    def get_name(self):
        return self.__class__.__name__

    @partial(jax.jit, static_argnums=(0,))
    def get_action(self, obs: jnp.ndarray,
                   env_state: WrappedEnvState,
                   agent_state: AgentState = None,
                   rng: jax.random.PRNGKey = None) -> Tuple[int, AgentState]:
        """Get action and updated state based on observation and current state.

        Args:
            obs: Flattened observation array
            env_state: WrappedEnvState containing the Coins environment state
            agent_state: AgentState containing agent's internal state
            rng: jax.random.PRNGKey for any stochasticity

        Returns:
            action, AgentState
        """
        coins_env_state = env_state.env_state  # extract underlying env state
        action, agent_state = self._get_action(obs, coins_env_state, agent_state, rng)
        return action, agent_state

    def _get_action(self, obs: jnp.ndarray, env_state: Any,
                    agent_state: AgentState, rng: jax.random.PRNGKey) -> Tuple[int, AgentState]:
        """To be implemented by subclasses."""
        raise NotImplementedError