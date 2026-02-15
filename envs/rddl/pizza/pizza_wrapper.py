from functools import partial
from typing import Dict, Any, List, Tuple, Optional

import chex
import hydra
import math
import os
import jax
import jax.numpy as jnp

import pyRDDLGym_jax

from pyRDDLGym_jax.core.env import JaxRDDLEnv, EnvState
from jaxmarl.environments import spaces as jaxmarl_spaces

from envs.base_env import BaseEnv
from envs.base_env import WrappedEnvState

class PizzaWrapper(BaseEnv):
    """Use the RDDL JAX Environment with JaxMARL environments.

    Args:
        *args: Positional arguments. First argument must be the JaxRDDLEnv.
        **kwargs: Keyword arguments.
            share_rewards (bool): Whether to share rewards between agents. Defaults to False.
    """
    def __init__(self, *args, **kwargs):
        if not args or not isinstance(args[0], JaxRDDLEnv):
            raise ValueError("First argument must be a JaxRDDLEnv instance")

        self.env = args[0]
        self.vectorized = kwargs.get('vectorized', True)
        self.share_rewards = kwargs.get('share_rewards', False)
        self._render = False # kwargs.get('render', False)
        self._render_name = kwargs.get('render_name', "pizza")
        self._render_dir = kwargs.get('render_dir', "pizza")

        self._ego_centric_obs = kwargs.get('ego_centric_obs', False)

        self.horizon = self.env.horizon
        self.name = self.env.__class__.__name__
        self.rddl_agent_names = self.env.model.type_to_objects['truck']
        self.rddl_action_keys = list(self.env.action_space.keys())
        self.num_agents = len(self.rddl_agent_names)
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]

        # Domain-specific non-fluents
        self.reward_goal = self.env.model.non_fluents.get('DELIVERY-REWARD', 1.0)
        self.rddl_location_names = self.env.model.type_to_objects['location']
        order_list = self.env.model.non_fluents['ORDERS']
        self.num_orders = {loc_name: order_list[i] for i, loc_name in enumerate(self.rddl_location_names)}

        self.pizzas_list = self.env.model.type_to_objects['pizza']
        self.locations_list = self.env.model.type_to_objects['location']

        if self._render:
            self.render_name = kwargs.get('render_name', "pizza")
            self.render_dir = os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, kwargs.get('render_dir', "render"))
            if not os.path.exists(self.render_dir):
                os.makedirs(self.render_dir)
            # Visualization setup can be added here if needed

        # WARNING: This wrapper currently only supports homogeneous agent envs
        self.observation_spaces = {}
        self.observation_full_spaces = {}
        for agent_idx, agent in enumerate(self.agents):
            agent_obs, full_obs = self._convert_rddl_obs_spec_to_jaxmarl_space(self.env.observation_space)
            self.observation_spaces[agent] = agent_obs
            self.observation_full_spaces[agent] = full_obs

        self.action_spaces = {
            agent: self._convert_rddl_action_spec_to_jaxmarl_space(self.env.action_space)
            for agent_idx, agent in enumerate(self.agents)
        }

        # Precompute action structure for unbatchification (avoid JIT issues)
        self._action_type_info = []
        self._action_type_sizes = []
        for action_name in self.rddl_action_keys:
            action_space = self.env.action_space[action_name]
            spec_shape = action_space.shape

            if len(spec_shape) == 1:
                param_shape = ()
                size = 1
            else:
                param_shape = spec_shape[1:]
                size = int(math.prod(param_shape))

            output_shape = (self.num_agents,) + param_shape
            self._action_type_info.append((action_name, param_shape, output_shape, size))
            self._action_type_sizes.append(size)

        # Precompute cumulative boundaries as static array
        self._cumsum_sizes = jnp.array([0] + list(jnp.cumsum(jnp.array(self._action_type_sizes))))

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

    @partial(jax.jit, static_argnums=(0,))
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
                    # Reorder truckAt: put agent_idx first, then others
                    truck_at_reordered = jnp.concatenate([
                        observation['truckAt'][agent_idx:agent_idx+1],  # This agent
                        observation['truckAt'][:agent_idx],             # Agents before
                        observation['truckAt'][agent_idx+1:]            # Agents after
                    ], axis=0).flatten()

                    delivered = observation['delivered'].flatten()
                    pizzaInTruck = observation['pizzaInTruck'].flatten()
                    hot = observation['hot'].flatten()
                    disposed = observation['disposed'].flatten()

                    obs_agent_values.append(truck_at_reordered)
                    obs_agent_values.append(delivered)
                    obs_agent_values.append(pizzaInTruck)
                    obs_agent_values.append(hot)
                    obs_agent_values.append(disposed)

                    obs_full_values.append(truck_at_reordered)
                    obs_full_values.append(delivered)
                    obs_full_values.append(pizzaInTruck)
                    obs_full_values.append(hot)
                    obs_full_values.append(disposed)

                    obs_agent[agent] = jnp.concatenate(obs_agent_values, dtype=self.observation_spaces[agent].dtype)
                    obs_full[agent] = jnp.concatenate(obs_full_values, dtype=self.observation_spaces[agent].dtype)
                else:
                    # Non-ego-centric: keep original ordering
                    truck_at = observation['truckAt'].flatten()
                    delivered = observation['delivered'].flatten()
                    pizzaInTruck = observation['pizzaInTruck'].flatten()
                    hot = observation['hot'].flatten()
                    disposed = observation['disposed'].flatten()

                    obs_agent_values.append(truck_at)
                    obs_agent_values.append(delivered)
                    obs_agent_values.append(pizzaInTruck)
                    obs_agent_values.append(hot)
                    obs_agent_values.append(disposed)

                    obs_full_values.append(truck_at)
                    obs_full_values.append(delivered)
                    obs_full_values.append(pizzaInTruck)
                    obs_full_values.append(hot)
                    obs_full_values.append(disposed)

                    obs_agent[agent] = jnp.concatenate(obs_agent_values, dtype=self.observation_spaces[agent].dtype)
                    obs_full[agent] = jnp.concatenate(obs_full_values, dtype=self.observation_spaces[agent].dtype)
            return obs_agent, obs_full
        else:
            raise NotImplementedError("Non-vectorized observations not implemented yet.")

    @partial(jax.jit, static_argnums=(0,))
    def _actions_to_rddl(self, actions: Dict[str, jnp.array]) -> Dict[str, jnp.array]:
        """Convert agent actions (flat indices) to RDDL action dictionary.

        Args:
            actions: Dict {agent_id: flat_action_index} where flat_action_index
                    is an integer indexing into the flattened action space

        Returns:
            Dictionary with action names as keys and boolean arrays indicating
            which specific actions are taken (format expected by RDDL environment)
        """
        # Stack all action indices into a single array
        action_indices = jnp.stack([actions[agent] for agent in self.agents])

        # Unbatchify to get RDDL action dictionary
        rddl_actions = self._unbatchify_actions(action_indices)

        return rddl_actions

    @partial(jax.jit, static_argnums=(0,))
    def _unbatchify_actions(self, action_indices: jnp.array) -> Dict[str, jnp.array]:
        """Convert flat action indices back to RDDL action dictionary format.

        Args:
            action_indices: Array of shape (num_agents,) with flat action indices
                           Each element is a scalar integer representing which action
                           that agent selected from the flattened action space.

        Returns:
            Dictionary with action names as keys and multi-dimensional boolean arrays
            indicating which actions are selected. The format matches what the RDDL
            environment expects (last dimension is [False, True] pair).
        """
        # Ensure action_indices is 1D array of shape (num_agents,)
        action_indices = jnp.atleast_1d(action_indices).squeeze()

        # Use precomputed action structure from __init__
        rddl_actions = {}

        for idx, (action_name, param_shape, output_shape, size) in enumerate(self._action_type_info):
            start_idx = self._cumsum_sizes[idx]
            end_idx = self._cumsum_sizes[idx + 1]

            # For each agent, check if their action index falls in this action type's range
            in_range = (action_indices >= start_idx) & (action_indices < end_idx)

            # Get the relative index within this action type
            relative_indices = action_indices - start_idx

            # Create the action array with correct output shape
            # Initialize with zeros (False/not selected)
            action_array = jnp.zeros(output_shape, dtype=jnp.bool)

            if len(param_shape) == 0:
                # Simple action with no parameters (e.g., 'noop')
                # Output shape: (num_agents,)
                # Set to True for agents selecting this action, False otherwise
                action_array = in_range
            else:
                # Action with parameters (e.g., 'drive' with 3 options)
                # Output shape: (num_agents, *param_shape)

                # Convert flat relative indices to multi-dimensional parameter indices for all agents
                # Only matters for agents where in_range is True
                multi_indices = jnp.unravel_index(relative_indices, param_shape)

                # Create one-hot encoding for each agent's selected parameter
                # For each agent, create a mask array with True at the selected index
                for agent_idx in range(self.num_agents):
                    # Only set if this agent is selecting this action type
                    selected_param_indices = tuple(mi[agent_idx] for mi in multi_indices)
                    full_indices = (agent_idx,) + selected_param_indices
                    action_array = jnp.where(
                        in_range[agent_idx],
                        action_array.at[full_indices].set(True),
                        action_array
                    )

            rddl_actions[action_name] = action_array

        return rddl_actions

    @partial(jax.jit, static_argnums=(0,))
    def _extract_rewards(self, reward: float) -> Dict[str, float]:
        """Extract and process rewards for each agent.

        Args:
            reward: Scalar reward from the environment

        Returns:
            A dictionary {agent: reward_value}
        """
        if self.share_rewards:
            total_reward = jnp.sum(reward)
            rewards = {agent: total_reward for agent in self.agents}
        else:
            rewards = {agent: reward[i] for i, agent in enumerate(self.agents)}
        return rewards

    @partial(jax.jit, static_argnums=(0,))
    def _extract_dones(self, timestep) -> Dict[str, bool]:
        """Create done dict based on timestep termination status."""
        # The pizza domain does not use a per-agent "goal-reached" boolean
        # in the same way as the grid domain. For now return per-agent
        # False and rely on the environment termination for "__all__".
        done = timestep.done # Epsiode terminal condition reached
        terminal = timestep.truncated # Episode limit reached
        dones = {agent: False for agent in self.agents}
        dones["__all__"] = done | terminal
        return dones

    @partial(jax.jit, static_argnums=(0,))
    def _extract_infos(self, timestep) -> Dict:
        """Extract additional info from timestep."""
        # Check if all orders are fulfilled (goal achieved)
        # This requires checking the environment state
        return {}

    @partial(jax.jit, static_argnums=(0,))
    def _extract_avail_actions(self, env_state: EnvState) -> Dict[str, jnp.ndarray]:
        """Returns a dict {agent: jnp.array([0/1,...])} indicating which actions are available.
        The array has one element per action in the action space.

        Convert available actions dictionary to flat vector for each agent.

        The input dictionary has actions with shapes like:
        - 'noop': (num_agents, 2) - last dim is [False, True]
        - 'drive': (num_agents, 3, 2) - 3 drive options
        - 'deliver': (num_agents, 2, 3, 2) - 2 locations × 3 pizzas
        - 'load': (num_agents, 2, 2) - 2 load locations
        - 'dispose': (num_agents, 2, 2) - 2 dispose locations

        We extract the True values (index 1 of last dimension) and flatten all
        action dimensions to create a flat mask for each agent.

        Returns:
            Dictionary {agent: flat_mask} where flat_mask is a 1D boolean array
        """
        if self.vectorized:
            avail_action_mask = self.env.get_available_actions(env_state)
            # Process all action types at once for all agents (vectorized)
            action_masks_per_type = []

            for action_name in self.rddl_action_keys:
                action_avail = avail_action_mask[action_name]

                # Extract the True values (index 1 of last dimension) for all agents
                # Shape: (num_agents, *action_dims)
                true_values = action_avail[..., 1]

                # Flatten action dimensions for each agent
                # Shape: (num_agents, num_actions_of_this_type)
                flattened = true_values.reshape(self.num_agents, -1)

                action_masks_per_type.append(flattened)

            # Concatenate all action types along the last dimension
            # Shape: (num_agents, total_num_actions)
            all_actions_batched = jnp.concatenate(action_masks_per_type, axis=1).astype(jnp.float32)

            # Create dictionary mapping agent names to their action masks
            batched_avail = {agent: all_actions_batched[i] for i, agent in enumerate(self.agents)}

            return batched_avail

        else:
            raise NotImplementedError("Non-vectorized avail_actions not implemented yet.")

    def _convert_rddl_action_spec_to_jaxmarl_space(self, space: pyRDDLGym_jax.core.spaces.Dict) -> jaxmarl_spaces.Space:
        """Convert RDDL action space to JaxMARL discrete space.
        Each agent has N possible actions indexed 0..N-1.
        """
        if self.vectorized:
            num_actions = 0
            for key, value in space.items():
                if len(value.shape) == 1:
                    num_actions += 1
                else:
                    num_actions += math.prod(value.shape[1:])
            return jaxmarl_spaces.Discrete(num_categories=num_actions, dtype=jnp.int32)
        else:
            raise NotImplementedError("Non-vectorized action spaces not implemented yet.")

    def _convert_rddl_obs_spec_to_jaxmarl_space(self, space: pyRDDLGym_jax.core.spaces.Dict) -> Tuple[jaxmarl_spaces.Space, jaxmarl_spaces.Space]:
        """Converts the observation spec for each agent to a JaxMARL space."""
        # Remove collision indicators from the space
        if self.vectorized:
            obs_size = 0
            obs_full_size = 0
            for key in space.keys():
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

    def _randomize_initial_positions(self, key: chex.PRNGKey, env_state, timestep):
        '''Randomizes initial positions of agents and goals using JAX-compatible operations'''
        # The pizza domain has different init structures than the grid domain.
        # For now we do not modify sampler init values; let the environment's
        # default reset behavior apply. Implement domain-aware randomization
        # later if needed.
        return env_state, timestep

    def render(self, env_state: EnvState, save_frame: bool = True) -> Any:
        """
        Render the environment state. For now, this is a placeholder that can be expanded
        with actual visualization code.
        """
        pass

    def reset_render(self):
        """Reset any internal state related to rendering. Placeholder for now."""
        pass

    def animate(self, states, dones, num_episodes, extra_dir=None, fps=1, loop_count=0, debug=False):
        """
        This method can be implemented to create animations from a sequence of states and dones.
        For now, we can rely on the environment's built-in movie generator if available.
        """
        pass

if __name__ == "__main__":
    # Simple test of the PizzaWrapper with JAX
    import os
    from pyRDDLGym_jax.core.env import JaxRDDLEnv

    domain_path = os.path.join(os.path.dirname(__file__), 'pizza_domain_new.rddl')
    instance_path = os.path.join(os.path.dirname(__file__), 'pizza_instance0.rddl')

    # Create the JAX RDDL environment
    jax_env = JaxRDDLEnv(domain=domain_path, instance=instance_path)

    # Wrap it
    wrapper = PizzaWrapper(jax_env, render=False)

    # Test reset and step
    key = jax.random.PRNGKey(0)
    obs, state = wrapper.reset(key)

    print(f"Number of agents: {wrapper.num_agents}")
    for agent in wrapper.agents:
        print(f"Agent: {agent}")
        print(f"  Observation space: {wrapper.observation_space(agent, 'agent')}")
        print(f"  Action space: {wrapper.action_space(agent)}")

    print(f"\nInitial observation keys: {obs.keys()}")
    print(f"Initial observation shape: {obs[wrapper.agents[0]].shape}")

    # Take a few random steps
    for step_i in range(5):
        key, *step_keys = jax.random.split(key, wrapper.num_agents + 1)

        # Sample random actions
        actions = {}
        for i, agent in enumerate(wrapper.agents):
            action_key = step_keys[i]
            actions[agent] = jax.random.randint(action_key, (), 0, wrapper.action_space(agent).n)

        print(f"\nStep {step_i + 1}:")
        print(f"  Actions: {actions}")

        key, step_key = jax.random.split(key)
        obs, state, rewards, dones, info = wrapper.step(step_key, state, actions)

        print(f"  Rewards: {rewards}")
        print(f"  Done: {dones['__all__']}")

        if dones["__all__"]:
            print("  Episode finished!")
            break

    print("\nTest completed successfully!")
