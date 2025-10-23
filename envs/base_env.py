from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any
import chex

class BaseEnv(ABC):
    """
    Abstract base class for multi-agent reinforcement learning environments.
    """
    
    @abstractmethod
    def step(
        self, 
        rng: chex.PRNGKey, 
        env_state: Any, 
        env_act: Dict[str, chex.Array]
    ) -> Tuple[Dict[str, chex.Array], Any, Dict[str, float], Dict[str, bool], Dict]:
        """
        Take a step in the environment.
        
        Args:
            rng: JAX random key for stochastic operations
            env_state: Current environment state (type depends on specific environment)
            env_act: Dictionary mapping agent names to actions
            
        Returns:
            obs: Dictionary mapping agent names to observations
            state: New environment state
            rewards: Dictionary mapping agent names to scalar rewards
            dones: Dictionary mapping agent names to done flags (plus '__all__' key)
            info: Dictionary with auxiliary information
        """
        raise NotImplementedError

    @abstractmethod
    def reset(
        self, 
        rng: chex.PRNGKey
    ) -> Tuple[Dict[str, chex.Array], Any]:
        """
        Reset the environment to initial state.
        
        Args:
            rng: JAX random key for stochastic initialization
            
        Returns:
            obs: Dictionary mapping agent names to initial observations
            state: Initial environment state
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_avail_actions(self, env_state: Any) -> Dict[str, chex.Array]:
        """
        Get available (legal) actions for each agent.
        
        Args:
            env_state: Current environment state
            
        Returns:
            Dictionary mapping agent names to binary masks of shape (num_actions,)
            where 1 indicates the action is available and 0 indicates it's not.
        """
        raise NotImplementedError

    @abstractmethod
    def observation_space(self, agent: str):
        """
        Get the observation space for a specific agent.
        
        Args:
            agent: Name of the agent (e.g., "agent_0")
            
        Returns:
            Gym/JaxMARL space object (e.g., spaces.Box, spaces.Discrete)
        """
        raise NotImplementedError

    @abstractmethod
    def action_space(self, agent: str):
        """
        Get the action space for a specific agent.
        
        Args:
            agent: Name of the agent (e.g., "agent_0")
            
        Returns:
            Gym/JaxMARL space object (e.g., spaces.Box, spaces.Discrete)
        """
        raise NotImplementedError

    def __getattr__(self, name):
        """
        Fallback to access attributes from parent class.
        Allows wrappers to expose underlying environment attributes.
        """
        return getattr(super(), name)