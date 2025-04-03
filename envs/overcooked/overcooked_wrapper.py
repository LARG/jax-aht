from functools import partial
from typing import Dict

import jax
import jax.numpy as jnp
from jaxmarl.environments.overcooked.overcooked import State as OvercookedState
from jaxmarl.environments import spaces

from envs.overcooked.overcooked_v2 import OvercookedV2  


class OvercookedWrapper(OvercookedV2):
    '''Wrapper for the Overcooked environment to ensure that it follows a common interface 
    with other environments provided in this library.'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def observation_space(self, agent: str):
        """Returns the flattened observation space."""
        # Calculate flattened observation shape
        flat_obs_shape = (self.obs_shape[0] * self.obs_shape[1] * self.obs_shape[2],)
        return spaces.Box(0, 255, flat_obs_shape)

    def action_space(self, agent: str):
        return super().action_space()
    
    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: OvercookedState) -> Dict[str, jnp.ndarray]:
        """Returns the available actions for each agent."""
        num_actions = len(self.action_set)
        return {agent: jnp.ones(num_actions) for agent in self.agents}
    
    @partial(jax.jit, static_argnums=(0,))
    def get_step_count(self, state: OvercookedState) -> jnp.array:
        """Returns the step count for the environment."""
        return state.time

    def get_obs(self, state: OvercookedState) -> Dict[str, jnp.ndarray]:
        """Returns flattened observations for each agent."""
        obs = super().get_obs(state)
        # Flatten observations for each agent
        return {agent: obs[agent].flatten() for agent in self.agents}

    @partial(jax.jit, static_argnums=(0,))
    def step_env(self, key: jax.random.PRNGKey, state: OvercookedState, actions: Dict[str, jnp.ndarray]) -> tuple:
        """Override step_env to reshape the info dictionary."""
        obs, state, rewards, dones, info = super().step_env(key, state, actions)
        
        # Reshape shaped_reward into a jnp array
        shaped_rewards = jnp.array([info['shaped_reward'][agent] for agent in self.agents])
        info = {'shaped_reward': shaped_rewards}
        
        return obs, state, rewards, dones, info
