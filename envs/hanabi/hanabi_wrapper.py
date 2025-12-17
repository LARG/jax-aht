from functools import partial
from typing import Dict, Tuple, Optional

import chex
import jax
import jax.numpy as jnp
from flax.struct import dataclass
from jaxmarl.environments.hanabi.hanabi import HanabiEnv
from jaxmarl.environments.hanabi.hanabi import State as HanabiState
from jaxmarl.environments import spaces

from ..base_env import BaseEnv
from ..base_env import WrappedEnvState

class HanabiWrapper(BaseEnv):
    '''Wrapper for the Hanabi environment to ensure that it follows a common interface 
    with other environments provided in this library.
    
    Main features:
    - Randomized agent order
    - Flattened observations
    - Base return tracking
    '''
    def __init__(self, *args, **kwargs):       
        self.env = HanabiEnv(*args, **kwargs)
        self.agents = self.env.agents
        self.num_agents = len(self.agents)

        self.observation_spaces = {agent: self.observation_space(agent) for agent in self.agents}
        self.action_spaces = {agent: self.action_space(agent) for agent in self.agents}

    def observation_space(self, agent: str):
        """Returns the observation space."""
        obs_space = self.env.observation_space(agent)
        obs_space.shape = (obs_space.n,)
        return obs_space

    def action_space(self, agent: str):
        """Returns the action space."""
        act_space = self.env.action_space(agent)
        act_space.shape = (act_space.n,)
        return act_space
    
    def reset(self, key: chex.PRNGKey, ) -> Tuple[Dict[str, chex.Array], WrappedEnvState]:
        obs, env_state = self.env.reset(key)
        # compute avail_actions from the raw env_state
        avail_actions = self.env.get_legal_moves(env_state)
        step = env_state.turn
        return obs, WrappedEnvState(env_state=env_state,
                                     base_return_so_far=jnp.zeros(self.num_agents),
                                     avail_actions=avail_actions,
                                     step=step)

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: WrappedEnvState) -> Dict[str, jnp.ndarray]:
        """Returns the available actions for each agent."""
        return self.env.get_legal_moves(state.env_state)
    
    @partial(jax.jit, static_argnums=(0,))
    def get_step_count(self, state: WrappedEnvState) -> jnp.array:
        """Returns the step count for the environment."""
        return state.env_state.turn

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
        # Pass the unwrapped env_state to the underlying environment
        reset_env_state = reset_state.env_state if reset_state is not None else None
        obs, env_state, rewards, dones, infos = self.env.step(key, state.env_state, actions, reset_env_state)
        
        # Extract base reward (assuming rewards are the base rewards for Hanabi)
        # Convert rewards dict to array for tracking
        base_reward = jnp.array([rewards[agent] for agent in self.agents])
        base_return_so_far = base_reward + state.base_return_so_far
        new_info = {**infos, 'base_return': base_return_so_far, 'base_reward': base_reward}
        
        # handle auto-resetting the base return upon episode termination
        base_return_so_far = jax.lax.select(dones['__all__'], jnp.zeros(self.num_agents), base_return_so_far)
        # compute new avail_actions and step
        avail_actions = self.env.get_legal_moves(env_state)
        step = env_state.turn
        new_state = WrappedEnvState(env_state=env_state,
                                    base_return_so_far=base_return_so_far,
                                    avail_actions=avail_actions,
                                    step=step)
        return obs, new_state, rewards, dones, new_info