from functools import partial
from typing import Dict, Any, List, Tuple, Optional

import chex
import hydra
import math
import os
import jax
import jax.numpy as jnp
import numpy as np
import pyRDDLGym_jax

# from pyRDDLGym.core.visualizer.movie import MovieGenerator
from pyRDDLGym_jax.core.env import JaxRDDLEnv, EnvState
from jumanji import specs as jumanji_specs
from jaxmarl.environments import spaces as jaxmarl_spaces

from envs.base_env import BaseEnv
from envs.base_env import WrappedEnvState

from envs.rddl.grid_4x4.Grid4x4MultiAgentViz import Grid4x4MultiAgentVisualizer

class Grid4x4Wrapper(BaseEnv):
    """Use the RDDL JAX Environment with JaxMARL environments.

    Args:
        *args: Positional arguments. First argument must be the JumanjiEnv.
        **kwargs: Keyword arguments.
            share_rewards (bool): Whether to share rewards between agents. Defaults to False.
    """
    def __init__(self, *args, **kwargs):
        if not args or not isinstance(args[0], JaxRDDLEnv):
            raise ValueError("First argument must be a JaxRDDLEnv instance")

        self.env = args[0]
        self.vectorized = kwargs.get('vectorized', True)
        self.share_rewards = kwargs.get('share_rewards', False)
        self._render = kwargs.get('render', False)
        self._render_name = kwargs.get('render_name', "grid_4x4")
        self._render_dir = kwargs.get('render_dir', "grid_4x4")

        self._ego_centric_obs = kwargs.get('ego_centric_obs', False)

        # Stochastic movement: probability that horizontal movement (left-right) becomes vertical (up-down)
        self.stochastic_movement_prob = kwargs.get('stochastic_movement_prob', 0.0)

        self.horizon = self.env.horizon
        self.name = self.env.__class__.__name__
        self.rddl_agent_names = self.env.model.type_to_objects['agent']
        self.rddl_action_keys = sorted(self.env.action_space.keys())
        self.num_agents = len(self.rddl_agent_names)
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]

        controllable = self.env.model.non_fluents['CONTROLLABLE']
        self.uncontrolled_agents_exist = not all(controllable)
        self.controllable = {}
        self.controlled_agents = []
        self.uncontrolled_agents = []
        for agent_idx, agent in enumerate(self.agents):
            if controllable[agent_idx]:
                self.controllable[agent] = True
                self.controlled_agents.append(agent_idx)
            else:
                self.controllable[agent] = False
                self.uncontrolled_agents.append(agent_idx)
        self.single_agent_projection = len(self.controlled_agents) == 1 and self.uncontrolled_agents_exist

        self._xpos_list = list(self.env.model.type_to_objects['xpos'])
        self._ypos_list = list(self.env.model.type_to_objects['ypos'])
        self._restriction_type = self.env.model.non_fluents['RESTRICTION-TYPE']

        left_half_array = self.env.model.non_fluents['LEFT-HALF']
        right_half_array = self.env.model.non_fluents['RIGHT-HALF']
        bottom_half_array = self.env.model.non_fluents['BOTTOM-HALF']
        top_half_array = self.env.model.non_fluents['TOP-HALF']

        self._left_half = [x for i, x in enumerate(self._xpos_list) if i < len(left_half_array) and left_half_array[i]]
        self._right_half = [x for i, x in enumerate(self._xpos_list) if i < len(right_half_array) and right_half_array[i]]
        self._bottom_half = [y for i, y in enumerate(self._ypos_list) if i < len(bottom_half_array) and bottom_half_array[i]]
        self._top_half = [y for i, y in enumerate(self._ypos_list) if i < len(top_half_array) and top_half_array[i]]

        agent_half_array = self.env.model.non_fluents['AGENT-HALF']
        self._agent_halves = {}
        for i, agent in enumerate(self.agents):
            self._agent_halves[agent] = agent_half_array[i]

        if self._render:
            self.render_name = kwargs.get('render_name', "grid_4x4")
            self.render_dir = os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, kwargs.get('render_dir', "render"))
            if not os.path.exists(self.render_dir):
                os.makedirs(self.render_dir)
            # movie_gen = MovieGenerator(save_dir=self.render_dir, env_name=self.render_name, max_frames=1000, frame_duration=1000)
            viz=Grid4x4MultiAgentVisualizer(self.env.model)
            # self.env.set_visualizer(visualizer=viz, movie_gen=movie_gen)
            self.env.set_visualizer(visualizer=viz)

        # WARNING: This wrapper currently only supports homogeneous agent envs
        self.observation_spaces = {}
        self.observation_full_spaces = {}
        for agent_idx, agent in enumerate(self.agents):
            agent_obs, full_obs = self._convert_rddl_obs_spec_to_jaxmarl_space(self.env.observation_space)
            self.observation_spaces[agent] = agent_obs
            self.observation_full_spaces[agent] = full_obs

        self.action_spaces = {
            agent: self._convert_rddl_action_spec_to_jaxmarl_space(self.env.action_space[self.rddl_action_keys[0]])
            for agent_idx, agent in enumerate(self.agents)
        }

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey):
        reset_key, randomize_key = jax.random.split(key)
        env_state, timestep = self.env.reset(reset_key)
        env_state, timestep = self._randomize_initial_positions(randomize_key, env_state, timestep)
        obs = self._extract_observations(timestep.observation)
        state = WrappedEnvState(env_state,
                                jnp.zeros(self.num_agents),
                                self._extract_avail_actions(env_state),
                                env_state.timestep)
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
        actions_rddl = self._actions_to_rddl(actions)

        env_state, timestep = self.env.step(state.env_state, actions_rddl)
        avail_actions = self._extract_avail_actions(env_state)
        state_st = WrappedEnvState(env_state, jnp.zeros(self.num_agents), avail_actions, env_state.timestep)
        obs_st = self._extract_observations(timestep.observation)
        reward = self._extract_rewards(timestep.reward)
        done = self._extract_dones(timestep)
        info = self._extract_infos(timestep)
        # Save the state before reset to info dict for rendering purposes
        info['pre_reset_state'] = state_st
        info['pre_reset_obs'] = obs_st
        # Auto-reset environment based on termination
        obs, state = jax.tree.map(
            lambda x, y: jax.lax.select(done["__all__"], x, y),
            self.reset(key_reset),
            (obs_st, state_st)
        )
        return obs, state, reward, done, info

    def observation_space(self, agent: str, observation_type: str = "agent") -> jaxmarl_spaces.Space:
        if observation_type == "agent":
            return self.observation_spaces[agent]
        elif observation_type == "full":
            return self.observation_full_spaces[agent]
        else:
            raise ValueError(f"Unknown observation_type: {observation_type}")

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
        if self.vectorized:
            obs_agent = {}
            obs_full = {}
            for agent_idx, agent in enumerate(self.agents):
                obs_agent_values = []
                obs_full_values = []

                if self._ego_centric_obs:
                    # Ego-centric: reorder so this agent's information comes first
                    # Reorder agent-at: put agent_idx first, then others
                    agent_at_reordered = jnp.concatenate([
                        observation['agent-at'][agent_idx:agent_idx+1],  # This agent
                        observation['agent-at'][:agent_idx],             # Agents before
                        observation['agent-at'][agent_idx+1:]            # Agents after
                    ], axis=0).flatten()

                    goal_at_reordered = jnp.concatenate([
                        observation['goal-at'][agent_idx:agent_idx+1],  # This agent
                        observation['goal-at'][:agent_idx],             # Agents before
                        observation['goal-at'][agent_idx+1:]            # Agents after
                    ], axis=0)

                    # Reorder goal-reached: put agent_idx first, then others
                    goal_reached_reordered = jnp.concatenate([
                        observation['goal-reached'][agent_idx:agent_idx+1],  # This agent
                        observation['goal-reached'][:agent_idx],             # Agents before
                        observation['goal-reached'][agent_idx+1:]            # Agents after
                    ], axis=0).flatten()

                    obs_agent_values.append(agent_at_reordered)
                    obs_agent_values.append(goal_at_reordered[0].flatten())
                    obs_agent_values.append(goal_reached_reordered)

                    obs_full_values.append(agent_at_reordered)
                    obs_full_values.append(goal_at_reordered.flatten())
                    obs_full_values.append(goal_reached_reordered)

                    obs_agent[agent] = jnp.concatenate(obs_agent_values, dtype=self.observation_spaces[agent].dtype)
                    obs_full[agent] = jnp.concatenate(obs_full_values, dtype=self.observation_spaces[agent].dtype)
                else:
                    # Non-ego-centric: keep original ordering
                    agent_at = observation['agent-at'].flatten()
                    goal_reached = observation['goal-reached'].flatten()
                    obs_agent_values.append(agent_at)
                    obs_agent_values.append(observation['goal-at'][agent_idx].flatten())
                    obs_agent_values.append(goal_reached)

                    # Full observation include all goal positions
                    obs_full_values.append(agent_at)
                    obs_full_values.append(observation['goal-at'].flatten())
                    obs_full_values.append(goal_reached)

                    obs_agent[agent] = jnp.concatenate(obs_agent_values, dtype=self.observation_spaces[agent].dtype)
                    obs_full[agent] = jnp.concatenate(obs_full_values, dtype=self.observation_spaces[agent].dtype)
            return obs_agent, obs_full
        else:
            raise NotImplementedError("Non-vectorized observations not implemented yet.")

    def _actions_to_rddl(self, actions: Dict[str, Any], key: Optional[chex.PRNGKey] = None):
        '''Convert dict of actions to array with optional stochastic movement'''
        if self.vectorized:
            action_array = jnp.zeros((self.num_agents,), dtype=jnp.int32)
            for agent_idx, agent in enumerate(self.agents):
                # Ensure action is scalar by squeezing if needed
                action_array = action_array.at[agent_idx].set(actions[agent].squeeze())

            # Apply stochastic movement if enabled
            # if self.stochastic_movement_prob > 0.0 and key is not None:
            #     action_array = self._apply_stochastic_movement(action_array, key)

            vectorized_actions = {self.rddl_action_keys[0]: action_array}
            return vectorized_actions
        else:
            raise NotImplementedError("Non-vectorized actions not implemented yet.")

    def _apply_stochastic_movement(self, action_array: jnp.ndarray, key: chex.PRNGKey) -> jnp.ndarray:
        """Apply stochastic movement where horizontal actions can become vertical.

        Actions: 0=noop, 1=west, 2=east, 3=south, 4=north
        With probability stochastic_movement_prob:
        - west (1) → south (3)
        - east (2) → north (4)

        Args:
            action_array: Array of actions for each agent
            key: PRNG key for randomness

        Returns:
            Modified action array with stochastic movements applied
        """
        # Generate Bernoulli random variables for each agent
        should_flip = jax.random.bernoulli(key, self.stochastic_movement_prob, shape=action_array.shape)

        # Identify which movements should be flipped
        is_west = (action_array == 1)
        is_east = (action_array == 2)
        should_flip_west = is_west & should_flip
        should_flip_east = is_east & should_flip

        # # Generate random values for each agent
        # flip_probs = jax.random.uniform(key, shape=action_array.shape)

        # # Identify which movements should be flipped
        # is_west = (action_array == 1)
        # is_east = (action_array == 2)
        # should_flip_west = is_west & (flip_probs < self.stochastic_movement_prob)
        # should_flip_east = is_east & (flip_probs < self.stochastic_movement_prob)

        action_array = jnp.where(should_flip_west, 3, action_array)
        action_array = jnp.where(should_flip_east, 4, action_array)

        return action_array

    def _extract_rewards(self, reward):
        '''Extract per-agent rewards'''
        if self.share_rewards:
            total_reward = jnp.sum(reward)
            rewards = {agent: total_reward for agent in self.agents}
        else:
            rewards = {agent: reward[i] for i, agent in enumerate(self.agents)}
        return rewards

    def _extract_dones(self, timestep):
        '''Extract per-agent done flags'''
        done = timestep.done # Epsiode terminal condition reached
        terminal = timestep.truncated # Episode limit reached

        # Per-agent dones based on goal-reached
        if self.vectorized:
            dones = {agent: timestep.observation['goal-reached'][agent_idx] for agent_idx, agent in enumerate(self.agents)}
        else:
            dones = {agent: timestep.observation[f'goal-reached___{self.rddl_agent_names[agent_idx]}'] for agent_idx, agent in enumerate(self.agents)}
        dones["__all__"] = done | terminal

        # For single agent projection, only the controlled agent's done flag matters for the "__all__" key
        if self.single_agent_projection:
            dones["__all__"] = dones[self.agents[self.controlled_agents[0]]] | terminal

        return dones

    def _extract_infos(self, timestep):
        '''Broadcast info into per-agent shape'''
        info = {}
        for k, v in timestep.info.items():
            info[k] = jnp.array([v for _ in range(self.num_agents)])
        return info

    def _extract_avail_actions(self, env_state):
        '''Extract per-agent avail_actions'''
        if self.vectorized:
            avail_action_mask = self.env.get_available_actions(env_state)
            avail_actions = {agent: avail_action_mask[self.rddl_action_keys[0]][i] for i, agent in enumerate(self.agents)}
            return avail_actions
        else:
            raise NotImplementedError("Non-vectorized avail_actions not implemented yet.")

    def _convert_rddl_obs_spec_to_jaxmarl_space(self, space: pyRDDLGym_jax.core.spaces.Dict):
        """Converts the observation spec for each agent to a JaxMARL space."""
        # Remove collision indicators from the space
        if self.vectorized:
            obs_size = 0
            obs_full_size = 0
            for key in space.keys():
                if 'collision' in key:
                    pass
                elif 'goal-at' in key:
                    obs_size += math.prod(space[key].shape[1:])
                    obs_full_size += math.prod(space[key].shape)
                else:
                    obs_size += math.prod(space[key].shape)
                    obs_full_size += math.prod(space[key].shape)

            # Create Box space
            observation_space = jaxmarl_spaces.Box(
                low=0,
                high=1,
                shape=(obs_size,),
                dtype=jnp.float32
            )

            observation_full_space = jaxmarl_spaces.Box(
                low=0,
                high=1,
                shape=(obs_full_size,),
                dtype=jnp.float32
            )
        else:
            raise NotImplementedError("Non-vectorized observation spaces not implemented yet.")

        return observation_space, observation_full_space

    def _convert_rddl_action_spec_to_jaxmarl_space(self, space: pyRDDLGym_jax.core.spaces.Dict):
        """Converts the action spec for each agent to a JaxMARL space."""
        if self.vectorized:
            return jaxmarl_spaces.Discrete(num_categories=int(space.high[0]) + 1, dtype=space.dtype)
        else:
            raise NotImplementedError("Non-vectorized action spaces not implemented yet.")

    def _randomize_initial_positions(self, key: chex.PRNGKey, env_state, timestep):
        '''Randomizes initial positions of agents and goals using JAX-compatible operations'''

        # Get grid dimensions
        num_x = len(self._xpos_list)
        num_y = len(self._ypos_list)

        # Create obstacle mask
        obstacle_mask = self.env.init_values['OBSTACLE']  # Shape: (num_x, num_y)

        # Initialize position arrays
        agent_at = jnp.zeros_like(env_state.obs['agent-at']) # Shape: (num_agents, num_x, num_y)
        goal_at = jnp.zeros_like(env_state.obs['goal-at']) # Shape: (num_agents, num_x, num_y)
        goal = jnp.zeros_like(env_state.subs['GOAL'])  # Shape: (num_agents, num_x, num_y)

        # Create a mask for occupied positions (start with obstacles)
        occupied = jnp.broadcast_to(obstacle_mask, (num_x, num_y))

        # Process each agent
        for agent_idx in range(self.num_agents):
            key, agent_key, goal_key = jax.random.split(key, 3)

            # Get valid positions for this agent based on restriction type
            agent_half = self._agent_halves[f"agent_{agent_idx}"]

            # Create mask for valid positions
            valid_mask = jnp.ones((num_x, num_y), dtype=jnp.bool_)

            # Apply restrictions
            if self._restriction_type == 1:  # Vertical split
                if agent_half == 0:  # Left half
                    left_indices = jnp.array([self._xpos_list.index(x) for x in self._left_half])
                    x_mask = jnp.zeros(num_x, dtype=jnp.bool_).at[left_indices].set(True)
                else:  # Right half
                    right_indices = jnp.array([self._xpos_list.index(x) for x in self._right_half])
                    x_mask = jnp.zeros(num_x, dtype=jnp.bool_).at[right_indices].set(True)
                valid_mask = valid_mask & x_mask[:, None]

            elif self._restriction_type == 2:  # Horizontal split
                if agent_half == 0:  # Bottom half
                    bottom_indices = jnp.array([self._ypos_list.index(y) for y in self._bottom_half])
                    y_mask = jnp.zeros(num_y, dtype=jnp.bool_).at[bottom_indices].set(True)
                else:  # Top half
                    top_indices = jnp.array([self._ypos_list.index(y) for y in self._top_half])
                    y_mask = jnp.zeros(num_y, dtype=jnp.bool_).at[top_indices].set(True)
                valid_mask = valid_mask & y_mask[None, :]

            # Combine with obstacle and occupied masks
            available_agent = valid_mask & ~occupied

            # Sample agent position
            flat_available = available_agent.flatten()
            flat_probs = flat_available.astype(jnp.float32) / jnp.sum(flat_available)
            agent_pos_idx = jax.random.choice(agent_key, num_x * num_y, p=flat_probs)
            agent_x = agent_pos_idx // num_y
            agent_y = agent_pos_idx % num_y

            # Update agent position
            agent_at = agent_at.at[agent_idx, agent_x, agent_y].set(True)
            occupied = occupied.at[agent_x, agent_y].set(True)

            # Sample goal position (exclude the agent position we just placed)
            available_goal = valid_mask & ~occupied
            flat_available_goal = available_goal.flatten()
            flat_probs_goal = flat_available_goal.astype(jnp.float32) / jnp.sum(flat_available_goal)
            goal_pos_idx = jax.random.choice(goal_key, num_x * num_y, p=flat_probs_goal)
            goal_x = goal_pos_idx // num_y
            goal_y = goal_pos_idx % num_y

            # Update goal position
            goal_at = goal_at.at[agent_idx, goal_x, goal_y].set(True)
            goal = goal.at[agent_idx, goal_x, goal_y].set(True)
            occupied = occupied.at[goal_x, goal_y].set(True)

        # Update environment state - create new dicts with updated values
        obs = {**env_state.obs, 'goal-at': goal_at, 'agent-at': agent_at}
        subs = {**env_state.subs, 'agent-at': agent_at, 'goal-at': goal_at, 'GOAL': goal}

        env_state = env_state.replace(obs=obs, state=obs, subs=subs)
        timestep = timestep.replace(observation=obs)

        return env_state, timestep

    def render(self, env_state: EnvState, save_frame: bool = True) -> Any:
        """Render the current environment state.

        This method extracts the state from the EnvState and calls the
        visualizer's render method. It's designed to be called outside
        JIT-compiled loops for visualization purposes.

        Args:
            env_state: The current environment state from step() or reset()
            save_frame: If True and movie_generator is set, saves the frame

        Returns:
            image: The rendered image (typically a PIL Image), or None if
                   no visualizer is set

        Example:
            ```python
            env_state, timestep = env.reset(key)
            image = env.render(env_state)

            # Save or display the image
            if image is not None:
                image.save('state.png')
            ```
        """
        if self.env._visualizer is None:
            return None

        # Extract state dictionary from EnvState
        state = env_state.state

        # Always convert to grounded format for visualizers
        # (JAX env internally uses vectorized representation)
        state = self.env.model.ground_vars_with_values(state)

        # Convert JAX arrays to NumPy arrays for visualizer compatibility
        state = {k: (np.asarray(v) if hasattr(v, '__array__') else v)
                 for k, v in state.items()}

        # Call visualizer's render method
        image = self.env._visualizer.render(state, env_state.actions)

        # Save frame to movie generator if enabled
        if save_frame and self.env._movie_generator is not None and image is not None:
            self.env._movie_generator.save_frame(image)

        return image

    def reset_render(self):
        self.env.reset_render()

    def animate(self, states, dones, num_episodes, extra_dir=None, fps=1, loop_count=0, debug=False):

        init_state, state = states

        if extra_dir is not None:
            os.makedirs(os.path.join(self._render_dir, extra_dir), exist_ok=True)

        for i in range(num_episodes):
            if debug:
                # Also save individual frames for debugging
                debug_dir = os.path.join(self._render_dir, extra_dir if extra_dir is not None else "", f"{self._render_name}_ep_{i}_frames")
                os.makedirs(debug_dir, exist_ok=True)

            # List of EnvState for each timestep in the episode
            init_env_state = self.unbatch_init_envstate(init_state, idx1=0, idx2=i)
            env_states = self.unbatch_envstate(state, idx1=0, idx2=i)

            # Generate frames and optionally save them for debugging
            frames = []
            frame = self.render(init_env_state)
            frames.append(frame)
            if debug:
                # Also save individual frames for debugging
                frame_path = os.path.join(debug_dir, f"frame_{0:04d}.png")
                frame.save(frame_path)

            for j in range(self.env.model.horizon):
                frame = self.render(env_states[j])
                frames.append(frame)
                if debug:
                    # Also save individual frames for debugging
                    frame_path = os.path.join(debug_dir, f"frame_{j+1:04d}.png")
                    frame.save(frame_path)

                if dones[0,i,j]:
                    break

            # Save animation
            frames[0].save(
                os.path.join(self._render_dir, extra_dir if extra_dir is not None else "", f"{self._render_name}_ep_{i}.gif"),
                save_all=True,
                append_images=frames[1:], # Append all frames from the second one onwards
                duration=1000 // fps,
                loop=loop_count
            )

            self.reset_render()

    @staticmethod
    def unbatch_init_envstate(batched_state, idx1=None, idx2=None):
        """Convert an Init EnvState with batched arrays into a single EnvState for the specified indices.

        Args:
            batched_state: EnvState with arrays of shape (batch_size, ...)
            idx1: Optional index for first dimension (if provided, slice into this dimension first)
            idx2: Optional index for second dimension (if provided, slice into this dimension second)

        Returns:
            EnvState object, each with arrays of shape (...)
        """
        # First, index into specified dimensions if provided
        state = batched_state
        if idx1 is not None:
            state = jax.tree.map(lambda x: x[idx1], state)
        if idx2 is not None:
            state = jax.tree.map(lambda x: x[idx2], state)

        return state

    @staticmethod
    def unbatch_envstate(batched_state, idx1=None, idx2=None, unbatch_axis=0):
        """Convert an EnvState with batched arrays into a list of EnvStates.

        Args:
            batched_state: EnvState with arrays of shape (batch_size, ...)
            idx1: Optional index for first dimension (if provided, slice into this dimension first)
            idx2: Optional index for second dimension (if provided, slice into this dimension second)
            unbatch_axis: Which axis to unbatch along (default 0, or 2 if idx1 and idx2 provided)

        Returns:
            List of EnvState objects, each with arrays of shape (...)
        """
        # First, index into specified dimensions if provided
        state = batched_state
        if idx1 is not None:
            state = jax.tree.map(lambda x: x[idx1], state)
        if idx2 is not None:
            state = jax.tree.map(lambda x: x[idx2], state)

        # Get batch size from the unbatch axis
        batch_size = jax.tree.leaves(state)[0].shape[unbatch_axis]

        # Create a list of unbatched states by indexing each array at position i along unbatch_axis
        if unbatch_axis == 0:
            return [jax.tree.map(lambda x: x[i], state) for i in range(batch_size)]
        else:
            # For non-zero axes, need to use dynamic slicing
            return [jax.tree.map(lambda x: jnp.take(x, i, axis=unbatch_axis), state) for i in range(batch_size)]
