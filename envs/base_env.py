from abc import ABC, abstractmethod

from typing import Any
import jax.numpy as jnp
from flax.struct import dataclass

@dataclass
class WrappedEnvState:
    env_state: Any  # Currently can be OvercookedState or an LBF state
    base_return_so_far: jnp.ndarray  # records the original return w/o reward shaping terms
    avail_actions: jnp.ndarray
    step: jnp.array

class BaseEnv(ABC):
    @abstractmethod
    def step(self, rng, env_state, env_act):
        raise NotImplementedError

    @abstractmethod
    def reset(self, rng, env_state):
        raise NotImplementedError
    
    @abstractmethod
    def get_avail_actions(self, env_state):
        raise NotImplementedError

    @abstractmethod
    def observation_space(self, agent: str):
        raise NotImplementedError

    @abstractmethod
    def action_space(self, agent: str):
        raise NotImplementedError

    def __getattr__(self, name):
        return getattr(super(), name)