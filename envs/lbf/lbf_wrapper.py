from functools import partial
from typing import Dict, Any, List, Tuple, Optional

import chex
from flax.struct import dataclass
import jax
import jax.numpy as jnp
from jumanji.env import Environment as JumanjiEnv
from jumanji import specs as jumanji_specs
from jaxmarl.environments import spaces as jaxmarl_spaces

from ..env_wrapper import BaseEnv
from ..env_wrapper import WrappedEnvState
    
class LBFWrapper(BaseEnv):
    """Use the LBF Jumanji Environment with JaxMARL environments.
    Warning: this wrapper has only been tested with LBF. It also runs with RWare, but has not been tested. 
    
    We add the option to share rewards between agents, since it is 
    shared according to the agent level in the LBF environment.

    Args:
        *args: Positional arguments. First argument must be the JumanjiEnv.
        **kwargs: Keyword arguments.
            share_rewards (bool): Whether to share rewards between agents. Defaults to False.
    """
    def __init__(self, *args, **kwargs):
        if not args or not isinstance(args[0], JumanjiEnv):
            raise ValueError("First argument must be a JumanjiEnv instance")
        
        self.env = args[0]
        self.share_rewards = kwargs.get('share_rewards', False)
        
        self.num_agents = self.env.num_agents
        self.name = self.env.__class__.__name__
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        # warning: this wrapper currently only supports homogeneous agent envs
        self.observation_spaces = {
            agent: self._convert_jumanji_obs_spec_to_jaxmarl_space(self.env.observation_spec, agent_idx)
            for agent_idx, agent in enumerate(self.agents)
        }

        self.action_spaces = {
            agent: self._convert_jumanji_action_spec_to_jaxmarl_space(self.env.action_spec, agent_idx)
            for agent_idx, agent in enumerate(self.agents)
        }

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey):
        env_state, timestep = self.env.reset(key)
        obs = self._extract_observations(timestep.observation)
        state = WrappedEnvState(env_state, 
                                jnp.zeros(self.num_agents),
                                self._extract_avail_actions(timestep),
                                timestep.observation.step_count)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: WrappedEnvState,
        actions: Dict[str, chex.Array],
        reset_state: Optional[WrappedEnvState] = None,
    ) -> Tuple[Dict[str, chex.Array], WrappedEnvState, Dict[str, float], Dict[str, bool], Dict]:
        '''Performs step transitions in the environment. 
        In compliance with JaxMARL MultiAgentEnv interface, auto-resets the environment if done.
        '''
        key, key_reset = jax.random.split(key)
        # Convert dict of actions to array
        actions_array = self._actions_to_array(actions)
        env_state, timestep = self.env.step(state.env_state, actions_array)
        avail_actions = self._extract_avail_actions(timestep)

        state_st = WrappedEnvState(env_state, jnp.zeros(self.num_agents), avail_actions, timestep.observation.step_count)
        obs_st = self._extract_observations(timestep.observation)
        reward = self._extract_rewards(timestep.reward)
        done = self._extract_dones(timestep)
        info  = self._extract_infos(timestep)
        # Auto-reset environment based on termination
        obs, state = jax.tree.map(
            lambda x, y: jax.lax.select(done["__all__"], x, y), 
            self.reset(key_reset), 
            (obs_st, state_st)
        )
        return obs, state, reward, done, info

    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: WrappedEnvState) -> Dict[str, jnp.ndarray]:
        """Returns the available actions for each agent."""
        return state.avail_actions

    @partial(jax.jit, static_argnums=(0,))
    def get_step_count(self, state: WrappedEnvState) -> jnp.array:
        """Returns the step count of the environment."""
        return state.step

    def _extract_observations(self, observation):
        '''Extract per-agent observations and flatten them into arrays'''
        obs = {}
        for i in range(self.num_agents):
            agent_view = observation.agents_view[i].flatten()
            # action_mask = observation.action_mask[i].astype(jnp.float32)  # Convert bool to float
            # step_count = jnp.array([observation.step_count], dtype=jnp.float32)
            # Concatenate all components into a single array
            # agent_obs = jnp.concatenate([agent_view, action_mask, step_count])
            agent_obs = jnp.concatenate([agent_view])

            obs[self.agents[i]] = agent_obs
        return obs

    def _actions_to_array(self, actions: Dict[str, Any]):
        '''Convert dict of actions to array'''
        actions_array = jnp.array([actions[agent] for agent in self.agents], dtype=jnp.int32)
        return actions_array

    def _extract_rewards(self, reward):
        '''Extract per-agent rewards'''
        if self.share_rewards:
            tot_reward = jnp.mean(reward)
            rewards = {agent: tot_reward for agent in self.agents}
        else: 
            rewards = {agent: reward[i] for i, agent in enumerate(self.agents)}
        return rewards

    def _extract_dones(self, timestep):
        '''Extract per-agent done flags'''
        done = timestep.last() # jumanji lbf returns a single boolean done for all agents
        dones = {agent: done for agent in self.agents}
        dones["__all__"] = done
        return dones

    def _extract_infos(self, timestep):
        '''Broadcast info into per-agent shape'''
        info = {}
        for k, v in timestep.extras.items():
            info[k] = jnp.array([v for _ in range(self.num_agents)])
        return info
    
    def _extract_avail_actions(self, timestep):
        '''Extract per-agent avail_actions'''
        avail_actions = {agent: timestep.observation.action_mask[i] for i, agent in enumerate(self.agents)}
        return avail_actions

    def _convert_jumanji_obs_spec_to_jaxmarl_space(self, spec: jumanji_specs.Spec, agent_idx: int):
        """Converts the observation spec for each agent to a JaxMARL space."""
        # Extract specs for 'agents_view', 'action_mask', and 'step_count'
        agents_view_spec = spec.agents_view

        # Get per-agent specs
        per_agent_view_spec = self._get_per_agent_spec(agents_view_spec, agent_idx)

        # Flatten shapes
        view_shape = int(jnp.prod(jnp.array(per_agent_view_spec.shape)))

        # Total observation length
        total_shape = (view_shape,)

        # Determine low and high bounds
        # For simplicity, use -inf and inf; adjust if you have specific bounds
        if hasattr(per_agent_view_spec, "minimum"):
            low = per_agent_view_spec.minimum
            high = per_agent_view_spec.maximum
        else:
            low = -jnp.inf * jnp.ones(total_shape, dtype=jnp.float32)
            high = jnp.inf * jnp.ones(total_shape, dtype=jnp.float32)

        # Create Box space
        observation_space = jaxmarl_spaces.Box(
            low=low,
            high=high,
            shape=total_shape,
            dtype=jnp.float32
        )
        return observation_space

    def _get_per_agent_spec(self, spec: jumanji_specs.Spec, agent_idx: int):
        """Extracts the per-agent spec from a batched spec."""
        if isinstance(spec, jumanji_specs.BoundedArray):
            per_agent_shape = spec.shape[1:]

            # Adjust minimum and maximum
            if isinstance(spec.minimum, jnp.ndarray) and spec.minimum.shape == spec.shape:
                per_agent_min = spec.minimum[1:]
            else:
                per_agent_min = spec.minimum  # scalar or broadcastable

            if isinstance(spec.maximum, jnp.ndarray) and spec.maximum.shape == spec.shape:
                per_agent_max = spec.maximum[1:]
            else:
                per_agent_max = spec.maximum  # scalar or broadcastable

            return jumanji_specs.BoundedArray(
                shape=per_agent_shape,
                dtype=spec.dtype,
                minimum=per_agent_min,
                maximum=per_agent_max,
                name=spec.name
            )
        elif isinstance(spec, jumanji_specs.Array):
            # Assuming the first dimension is num_agents
            per_agent_shape = spec.shape[1:]
            return jumanji_specs.Array(
                shape=per_agent_shape,
                dtype=spec.dtype,
                name=spec.name
            )
        else:
            raise NotImplementedError(f"Spec type {type(spec)} not supported for per-agent extraction.")

    def _convert_jumanji_action_spec_to_jaxmarl_space(self, spec: jumanji_specs.Spec, agent_idx: int):
        """Converts the action spec for each agent to a JaxMARL space."""
        if isinstance(spec, jumanji_specs.MultiDiscreteArray):
            num_actions = spec.num_values[agent_idx]
            return jaxmarl_spaces.Discrete(num_categories=int(num_actions), dtype=spec.dtype)
        elif isinstance(spec, jumanji_specs.DiscreteArray):
            return jaxmarl_spaces.Discrete(num_categories=spec.num_values, dtype=spec.dtype)
        else:
            raise NotImplementedError(f"Spec type {type(spec)} not supported for action spaces.")

    def render(self, state: WrappedEnvState):
        self.env.render(state.env_state)
    
    def animate(self, states: List[WrappedEnvState], interval=100):
        return self.env.animate([s.env_state for s in states], interval=interval)
