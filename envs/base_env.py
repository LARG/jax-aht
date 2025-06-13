from abc import ABC, abstractmethod

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