from functools import partial
from typing import Dict, Tuple, Optional

import chex
import jax
import jax.numpy as jnp
from flax.struct import dataclass
from jaxmarl.environments.overcooked.overcooked import State as OvercookedState
from jaxmarl.environments import spaces

from ..env_wrapper import EnvWrapper
from ..env_wrapper import WrappedEnvState
from envs.overcooked.overcooked_v1 import OvercookedV1

class OvercookedWrapper(EnvWrapper):
    '''Wrapper for the Overcooked-v1 environment to ensure that it follows a common interface 
    with other environments provided in this library.
    
    Main features:
    - Randomized agent order
    - Flattened observations
    - Base return tracking
    '''
    def __init__(self, *args, **kwargs):
        self.env = OvercookedV1(*args, **kwargs)
        self.agents = self.env.agents
        self.num_agents = len(self.agents)

        self.observation_spaces = {agent: self.observation_space(agent) for agent in self.agents}
        self.action_spaces = {agent: self.action_space(agent) for agent in self.agents}
        
        # exposing some variables from underlying environment
        self.agent_view_size = self.env.agent_view_size

    def observation_space(self, agent: str):
        """Returns the flattened observation space."""
        # Calculate flattened observation shape
        flat_obs_shape = (self.env.obs_shape[0] * self.env.obs_shape[1] * self.env.obs_shape[2],)
        return spaces.Box(0, 255, flat_obs_shape)

    def action_space(self, agent: str):
        return self.env.action_space()
    
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], WrappedEnvState]:
        obs, env_state = self.env.reset(key)
        flat_obs = {agent: obs[agent].flatten() for agent in self.agents} # flatten obs
        return flat_obs, WrappedEnvState(env_state, jnp.zeros(self.num_agents), jnp.zeros(self.num_agents), jnp.empty((), dtype=jnp.int32))

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: WrappedEnvState) -> Dict[str, jnp.ndarray]:
        """Returns the available actions for each agent."""
        num_actions = len(self.env.action_set)
        return {agent: jnp.ones(num_actions) for agent in self.agents}
    
    @partial(jax.jit, static_argnums=(0,))
    def get_step_count(self, state: WrappedEnvState) -> jnp.array:
        """Returns the step count for the environment."""
        return state.env_state.time

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: WrappedEnvState,
        actions: Dict[str, chex.Array],
        reset_state: Optional[WrappedEnvState] = None,
    ) -> Tuple[Dict[str, chex.Array], WrappedEnvState, Dict[str, float], Dict[str, bool], Dict]:
        '''Wrapped step function. The base return is 
        tracked in the info dictionary, so that the return can be obtained from the final info.
        '''
        obs, env_state, rewards, dones, infos = self.env.step(key, state.env_state, actions, reset_state)
        flat_obs = {agent: obs[agent].flatten() for agent in self.agents} # flatten obs
        # log the base return in the info
        base_reward = infos['base_reward']
        base_return_so_far = base_reward + state.base_return_so_far
        new_info = {**infos, 'base_return': base_return_so_far}
        
        # handle auto-resetting the base return upon episode termination
        base_return_so_far = jax.lax.select(dones['__all__'], jnp.zeros(self.num_agents), base_return_so_far)
        new_state = WrappedEnvState(env_state=env_state, base_return_so_far=base_return_so_far, avail_actions=jnp.zeros(self.num_agents), step=jnp.empty((), dtype=jnp.int32))
        return flat_obs, new_state, rewards, dones, new_info

