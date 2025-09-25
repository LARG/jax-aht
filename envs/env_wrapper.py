from functools import partial
from typing import Dict, Tuple, Optional

import chex
import jax
import jax.numpy as jnp
from flax.struct import dataclass
from jaxmarl.environments.overcooked.overcooked import State as OvercookedState
from jaxmarl.environments import spaces

# TODO: Add type hints for env_state in WrappedEnvState
@dataclass
class WrappedEnvState:
    env_state: Any # Currently can be OvercookedState or an LBF state
    base_return_so_far: jnp.ndarray # records the original return w/o reward shaping terms

class EnvWrapper:
    def __init__(self, env_type, *args, **kwargs):
        if env_type == 'lbf':
            from envs.lbf.lbf_wrapper import LBFWrapper
            self.env = LBFWrapper(*args, **kwargs)
        elif env_type == 'overcooked':
            from envs.overcooked.overcooked_wrapper import OvercookedWrapper
            self.env = OvercookedWrapper(*args, **kwargs)
        else:
            raise ValueError(f"Unsupported environment type: {env_type}")
        
        self.agents = self.env.agents
        self.num_agents = self.env.num_agents
        self.observation_spaces = self.env.observation_spaces
        self.action_spaces = self.env.action_spaces
        self.name = getattr(self.env, 'name', 'CustomEnv')

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey, ) -> Tuple[Dict[str, chex.Array], WrappedEnvState]:
        return self.env.reset(key)

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: WrappedEnvState,
        actions: Dict[str, chex.Array],
        reset_state: Optional[WrappedEnvState] = None,
    ) -> Tuple[Dict[str, chex.Array], WrappedEnvState, Dict[str, float], Dict[str, bool], Dict]:
        return self.env.step(key, state, actions, reset_state)
    
    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: WrappedEnvState) -> Dict[str, jnp.ndarray]:
        """Returns the available actions for each agent."""
        return self.env.get_avail_actions(state)

    @partial(jax.jit, static_argnums=(0,))
    def get_step_count(self, state: WrappedEnvState) -> jnp.array:
        """Returns the step count of the environment."""
        return self.env.get_step_count(state)
    
    def observation_space(self, agent: str):
        return self.env.observation_space()

    def action_space(self, agent: str):
        return self.env.action_space()