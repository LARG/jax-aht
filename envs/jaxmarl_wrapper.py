from functools import partial
from typing import Dict, Tuple, Optional, Any
import inspect

import chex
import jax
import jax.numpy as jnp
from flax.struct import dataclass
from jaxmarl.environments import spaces


@dataclass
class WrappedEnvState:
    env_state: Any  # The underlying JaxMARL environment state
    base_return_so_far: jnp.ndarray  # records the original return w/o reward shaping terms

class JaxMARLWrapper:
    # Generic wrapper for JaxMARL environments to ensure a common interface.

    def __init__(self, env_class, *args, **kwargs):
        """
        Args:
            env_class: A JaxMARL environment class 
            *args: Positional arguments to pass to env_class
            **kwargs: Keyword arguments to pass to env_class
        """
        self.env = env_class(*args, **kwargs)
        self.agents = self.env.agents
        self.num_agents = len(self.agents)
        
        # Detect if observation_space and action_space take agent parameter
        # this is because in some environments, agents have partial observability, causing different action spaces
        obs_sig = inspect.signature(self.env.observation_space)
        self.obs_space_takes_agent = len(obs_sig.parameters) > 0
        act_sig = inspect.signature(self.env.action_space)
        self.act_space_takes_agent = len(act_sig.parameters) > 0
        
        self.observation_spaces = {agent: self.observation_space(agent) for agent in self.agents}
        self.action_spaces = {agent: self.action_space(agent) for agent in self.agents}

        # use this function for get_avail_actions if possible
        self.has_legal_moves = hasattr(self.env, 'get_legal_moves')
                
        # overcooked visualization has us expose this variable
        # let us continue to do so
        if hasattr(self.env, 'agent_view_size'):
            self.agent_view_size = self.env.agent_view_size

    def observation_space(self, agent: str):
        """Returns the observation space with flattened shape."""
        if self.obs_space_takes_agent:
            obs_space = self.env.observation_space(agent)
        else:
            obs_space = self.env.observation_space()

        if isinstance(obs_space, spaces.Box):
            # Flatten Box observation space (works for any dimensionality)
            flat_shape = (int(jnp.prod(jnp.array(obs_space.shape))),)
            return spaces.Box(obs_space.low, obs_space.high, flat_shape)
        elif isinstance(obs_space, spaces.Discrete):
            # For discrete spaces, set shape to (n,) for one-hot encoding
            obs_space.shape = (obs_space.n,)
        
        return obs_space

    def action_space(self, agent: str):
        """Returns the action space."""
        if self.act_space_takes_agent:
            act_space = self.env.action_space(agent)
        else:
            act_space = self.env.action_space()
        
        # For discrete action spaces, set shape for consistency
        if isinstance(act_space, spaces.Discrete):
            act_space.shape = (act_space.n,)
        
        return act_space
    
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], WrappedEnvState]:
        """Reset the environment."""
        obs, env_state = self.env.reset(key)
        
        # Always flatten observations (no-op if already 1D)
        flat_obs = {agent: obs[agent].flatten() for agent in self.agents}

        return flat_obs, WrappedEnvState(env_state, jnp.zeros(self.num_agents))

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: WrappedEnvState) -> Dict[str, jnp.ndarray]:
        """Returns the available actions for each agent."""
        if self.has_legal_moves:
            # Use environment's legal moves if available
            return self.env.get_legal_moves(state.env_state)
        else:
            # Otherwise, create a mask of all 1s so this function still works
            num_actions = self.action_spaces[self.agents[0]].n
            return {agent: jnp.ones(num_actions) for agent in self.agents}
    
    @partial(jax.jit, static_argnums=(0,))
    def get_step_count(self, state: WrappedEnvState) -> jnp.array:
        """Returns the step count for the environment."""
        # Try common step count field names
        if hasattr(state.env_state, 'step_count'):
            return state.env_state.step_count # hanabi uses this
        elif hasattr(state.env_state, 'time'):
            return state.env_state.time       # overcooked uses this
        elif hasattr(state.env_state, 'step'):
            return state.env_state.step       # others could use this
        else:
            raise Exception("Could not compute step count from environment state.")

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: WrappedEnvState,
        actions: Dict[str, chex.Array],
        reset_state: Optional[WrappedEnvState] = None,
    ) -> Tuple[Dict[str, chex.Array], WrappedEnvState, Dict[str, float], Dict[str, bool], Dict]:
        """
        Wrapped step function. The base return is tracked in the info dictionary,
        so that the return can be obtained from the final info.
        """        
        # Call underlying environment step
        obs, env_state, rewards, dones, infos = self.env.step(
            key, state.env_state, actions, reset_state
        )
        
        # Always flatten observations (no-op if already 1D)
        obs = {agent: obs[agent].flatten() for agent in self.agents}
        
        # Extract base reward
        if 'base_reward' in infos:
            # If environment provides base_reward, use it
            base_reward = infos['base_reward']
        else:
            # Otherwise, convert rewards dict to array
            base_reward = jnp.array([rewards[agent] for agent in self.agents])
        
        # Track base return
        base_return_so_far = base_reward + state.base_return_so_far
        new_info = {**infos, 'base_return': base_return_so_far, 'base_reward': base_reward}
        
        # Handle auto-resetting the base return upon episode termination
        base_return_so_far = jax.lax.select(dones['__all__'], jnp.zeros(self.num_agents), base_return_so_far)
        new_state = WrappedEnvState(env_state=env_state, base_return_so_far=base_return_so_far)
        return obs, new_state, rewards, dones, new_info