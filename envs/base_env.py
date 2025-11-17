from abc import ABC, abstractmethod

from typing import Any, Dict, Tuple
import jax.numpy as jnp
import chex
from jaxmarl.environments import spaces
from flax.struct import dataclass

@dataclass
class WrappedEnvState:
    env_state: Any  # Currently can be OvercookedState or an LBF state
    base_return_so_far: jnp.ndarray  # records the original return w/o reward shaping terms
    avail_actions: jnp.ndarray
    step: jnp.array

class BaseEnv(ABC):
    @abstractmethod
    def step(
        self, 
        rng: chex.PRNGKey, 
        env_state: WrappedEnvState, 
        env_act: Dict[str, chex.Array]
    ) -> Tuple[Dict[str, chex.Array], WrappedEnvState, Dict[str, float], Dict[str, bool], Dict]:
        raise NotImplementedError

    @abstractmethod
    def reset(
        self, 
        rng: chex.PRNGKey
    ) -> Tuple[Dict[str, chex.Array], WrappedEnvState]:
        raise NotImplementedError
    
    @abstractmethod
    def get_avail_actions(
        self, 
        env_state: WrappedEnvState
    ) -> Dict[str, jnp.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def observation_space(
        self, 
        agent: str
    ) -> spaces.Box:
        raise NotImplementedError

    @abstractmethod
    def action_space(
        self, 
        agent: str
    ) -> spaces.Discrete:
        raise NotImplementedError

    def __getattr__(self, name):
        return getattr(super(), name)