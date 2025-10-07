from functools import partial
from typing import Dict, Any, Tuple, Optional
from abc import ABC, abstractmethod
import chex
import jax
import jax.numpy as jnp
from flax.struct import dataclass


@dataclass
class WrappedEnvState:
    env_state: Any  # Currently can be OvercookedState or an LBF state
    base_return_so_far: jnp.ndarray  # records the original return w/o reward shaping terms
    avail_actions: jnp.ndarray
    step: jnp.array


class BaseEnv(ABC):
    """Abstract base class for multi-agent environments."""
    
    def __init__(self, *args, **kwargs):
        self.agents = None
        self.num_agents = None
        self.observation_spaces = None
        self.action_spaces = None
        self.name = 'BaseEnv'
    
    @abstractmethod
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], WrappedEnvState]:
        """Reset the environment and return initial observations and state."""
        pass
    
    @abstractmethod
    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: WrappedEnvState,
        actions: Dict[str, chex.Array],
        reset_state: Optional[WrappedEnvState] = None,
    ) -> Tuple[Dict[str, chex.Array], WrappedEnvState, Dict[str, float], Dict[str, bool], Dict]:
        """Execute one step in the environment."""
        pass
    
    @abstractmethod
    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: WrappedEnvState) -> Dict[str, jnp.ndarray]:
        """Returns the available actions for each agent."""
        pass
    
    @abstractmethod
    @partial(jax.jit, static_argnums=(0,))
    def get_step_count(self, state: WrappedEnvState) -> jnp.array:
        """Returns the step count of the environment."""
        pass
    
    @abstractmethod
    def observation_space(self, agent: str):
        """Returns the observation space for the given agent."""
        pass
    
    @abstractmethod
    def action_space(self, agent: str):
        """Returns the action space for the given agent."""
        pass


def create_env(env_type: str, *args, **kwargs) -> BaseEnv:
    """Factory function to create environment instances."""
    if env_type == 'lbf':
        from envs.lbf.lbf_wrapper import LBFWrapper
        return LBFWrapper(*args, **kwargs)
    elif env_type == 'overcooked':
        from envs.overcooked.overcooked_wrapper import OvercookedWrapper
        return OvercookedWrapper(*args, **kwargs)
    else:
        raise ValueError(f"Unsupported environment type: {env_type}")
