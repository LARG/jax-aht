from functools import partial
from typing import Dict, Any, List, Tuple, Optional

import chex
import hydra
import math
import os
import jax
import jax.numpy as jnp
import numpy as np


from flax.struct import dataclass
import pyRDDLGym_jax
from pyRDDLGym_jax.core.env import JaxRDDLEnv, EnvState
from jaxmarl.environments import spaces as jaxmarl_spaces

from envs.base_env import BaseEnv
# from envs.base_env import WrappedEnvState
from envs.rddl.pizza_v2.PizzaV2MultiAgentViz import PizzaV2MultiAgentVisualizer

from pyRDDLGym.core.visualizer.movie import MovieGenerator

@dataclass
class WrappedEnvState:
    env_state: Any
    base_return_so_far: jnp.ndarray  # records the original return w/o reward shaping terms
    avail_actions: jnp.ndarray
    step: jnp.array
    collisions_so_far: jnp.ndarray  # records the number of collisions so far in the episode
    

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
        self._render = kwargs.get('render', False)
        self._render_name = kwargs.get('render_name', "pizza")
        self._render_dir = kwargs.get('render_dir', "pizza")
        self._debug_rewards = kwargs.get('debug_rewards', False)
        self._debug_actions = kwargs.get('debug_actions', False)
        self._enforce_action_constraints = kwargs.get('enforce_action_constraints', True)

        self._ego_centric_obs = kwargs.get('ego_centric_obs', False)

        self.horizon = self.env.horizon
        self.name = self.env.__class__.__name__
        self.rddl_agent_names = self.env.model.type_to_objects['truck']
        self.rddl_action_keys = list(self.env.action_space.keys())
        self.num_agents = len(self.rddl_agent_names)
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]

        # Domain-specific non-fluents
        self.reward_goal = self.env.model.non_fluents.get('DELIVERY-REWARD', 10.0)
        self.collision_penalty = self.env.model.non_fluents.get('COLLISION-PENALTY', -20.0)
        self.step_penalty = self.env.model.non_fluents.get('STEP-PENALTY', -0.1)
        self.rddl_location_names = self.env.model.type_to_objects['location']
        self.locations_list = self.env.model.type_to_objects['location']
        # Precompute CAN-DELIVER non-fluent as a (num_agents, num_locations) boolean matrix
        try:
            non_fluents = self.env.model.ground_vars_with_values(self.env.model.non_fluents)
        except Exception:
            # Fallback to raw non_fluents dict if grounding not available
            non_fluents = self.env.model.non_fluents
        num_locs = len(self.locations_list)
        can_deliver_mat = np.zeros((self.num_agents, num_locs), dtype=np.bool_)
        for k, v in non_fluents.items():
            try:
                var, objects = self.env.model.parse_grounded(k)
            except Exception:
                # If key is already grounded or parsing fails, skip
                continue
            if var == 'CAN-DELIVER':
                truck, loc = objects
                if truck in self.rddl_agent_names and loc in self.locations_list:
                    t_idx = self.rddl_agent_names.index(truck)
                    l_idx = self.locations_list.index(loc)
                    can_deliver_mat[t_idx, l_idx] = bool(v)
        self.can_deliver = jnp.asarray(can_deliver_mat)
        self._obs_keys = [
            'truckAt',
            'numShopPizzas',
            'numOrdersRemaining',
            'numPizzasInTruck',
            'collision',
            'doneDelivering',
        ]
        obs_space_keys = set(self.env.observation_space.keys())
        missing_obs_keys = [key for key in self._obs_keys if key not in obs_space_keys]
        if missing_obs_keys:
            raise ValueError(f"Missing pizza_v2 observation keys: {missing_obs_keys}")

        if self._render:
            self.render_name = kwargs.get('render_name', "pizza")
            self.render_dir = os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, kwargs.get('render_dir', "render"))
            if not os.path.exists(self.render_dir):
                os.makedirs(self.render_dir)
            viz = PizzaV2MultiAgentVisualizer(self.env.model)
            self.env.set_visualizer(visualizer=viz)


        # WARNING: This wrapper currently only supports homogeneous agent envs
        self.observation_spaces = {}
        self.observation_full_spaces = {}
        for agent_idx, agent in enumerate(self.agents):
            agent_obs, full_obs = self._convert_rddl_obs_spec_to_jaxmarl_space(self.env.observation_space)
            self.observation_spaces[agent] = agent_obs
            self.observation_full_spaces[agent] = full_obs

        action_space = self.env.action_space[self.rddl_action_keys[0]]
        self.action_spaces = {
            agent: self._convert_rddl_action_spec_to_jaxmarl_space(action_space)
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
                                env_state.timestep,
                                jnp.zeros(self.num_agents, dtype=jnp.int32))
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
        action_inputs = actions

        # Optional wrapper-side guard for invalid selected actions.
        # If the selected flat action is masked out, project it to noop index 0.
        if self._enforce_action_constraints:
            action_inputs = {}
            for idx, agent in enumerate(self.agents):
                selected = actions[agent]
                # Normalize to scalar where possible to avoid 1-element-array vs scalar
                # inconsistencies between test and training code paths.
                selected = jnp.squeeze(selected)
                is_valid = state.avail_actions[agent][selected] > 0
                safe_selected = jnp.where(is_valid, selected, jnp.array(0, dtype=selected.dtype))
                action_inputs[agent] = safe_selected

            if self._debug_actions:
                jax.debug.print(
                    "[PizzaWrapper.step] projected invalid actions to noop. original={orig}, projected={proj}",
                    orig=jnp.stack([actions[a] for a in self.agents]),
                    proj=jnp.stack([action_inputs[a] for a in self.agents]),
                )

                # Also print shapes to help diagnose scalar vs 1-element-array issues
                jax.debug.print(
                    "[PizzaWrapper.step] action shapes: {sh}",
                    sh=jnp.stack([jnp.asarray(actions[a]).shape[0] if jnp.asarray(actions[a]).ndim>0 else 0 for a in self.agents])
                )

        # Convert dict of actions to array
        actions_rddl = self._actions_to_rddl(action_inputs)

        if self._debug_actions:
            test_subs = {**state.env_state.subs, **actions_rddl}
            precond_ok = jnp.array(False, dtype=jnp.bool_)
            if hasattr(self.env, '_check_preconditions'):
                precond_ok = self.env._check_preconditions(test_subs, state.env_state.model_aux, key)
            jax.debug.print("[PizzaWrapper.step] selected flat actions: a0={a0}, a1={a1}",
                            a0=action_inputs[self.agents[0]], a1=action_inputs[self.agents[1]])
            jax.debug.print("[PizzaWrapper.step] preconditions satisfied for submitted joint action: {ok}",
                            ok=precond_ok)
            jax.debug.print("[PizzaWrapper.step] submitted action-num={act}",
                            act=actions_rddl.get(self.rddl_action_keys[0], jnp.array([])))

        env_state, timestep = self.env.step(state.env_state, actions_rddl)
        if self._debug_rewards:
            reward_array = jnp.asarray(timestep.reward)
            jax.debug.print("[PizzaWrapper.step] raw timestep.reward = {r}", r=reward_array)
            jax.debug.print(
                "[PizzaWrapper.step] reward ndim = {nd}, size = {sz}",
                nd=jnp.asarray(reward_array.ndim),
                sz=jnp.asarray(reward_array.size),
            )
        if self._debug_actions:
            raw_avail = self.env.get_available_actions(env_state)
            jax.debug.print(
                "[PizzaWrapper.step] raw available (unsanitized) action-num={act}",
                act=raw_avail.get(self.rddl_action_keys[0], jnp.array([]))
            )
        
        avail_actions = self._extract_avail_actions(env_state)
        done = self._extract_dones(timestep, timestep.observation)
        # extract collision info
        new_episode_collisions = state.collisions_so_far + timestep.observation['collision'].astype(jnp.int32)
        
        state_st = WrappedEnvState(
            env_state, 
            jnp.zeros(self.num_agents), 
            avail_actions, 
            env_state.timestep,
            state.collisions_so_far * (1 - done["__all__"]) + new_episode_collisions * done["__all__"]
        )
        obs_st = self._extract_observations(timestep.observation)
        reward = self._extract_rewards(timestep.reward)
        info = self._extract_infos(timestep)
        # Save the state before reset to info dict for rendering purposes
        info['pre_reset_state'] = state_st
        info['pre_reset_obs'] = obs_st
        info['returned_episode_collisions'] = state_st.collisions_so_far
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

            # Shared features are global in pizza_v2 and are identical for each agent.
            shop_pizzas = jnp.atleast_1d(observation['numShopPizzas']).flatten()
            orders_remaining = jnp.atleast_1d(observation['numOrdersRemaining']).flatten()

            def _ego_reorder_truck_axis(arr: jnp.ndarray, idx: int) -> jnp.ndarray:
                """Move the current agent's truck slice to the front of axis 0."""
                return jnp.concatenate([
                    arr[idx:idx+1],
                    arr[:idx],
                    arr[idx+1:]
                ], axis=0)

            for agent_idx, agent in enumerate(self.agents):
                obs_agent_values = []
                obs_full_values = []

                if self._ego_centric_obs:
                    # Ego-centric: reorder all truck-dependent fluents so this agent is first.
                    truck_at_reordered = _ego_reorder_truck_axis(observation['truckAt'], agent_idx).flatten()
                    pizzas_in_truck_reordered = _ego_reorder_truck_axis(
                        jnp.atleast_1d(observation['numPizzasInTruck']), agent_idx
                    ).flatten()
                    collision_reordered = _ego_reorder_truck_axis(
                        jnp.atleast_1d(observation['collision']), agent_idx
                    ).flatten()
                    done_delivering_reordered = _ego_reorder_truck_axis(
                        jnp.atleast_1d(observation['doneDelivering']), agent_idx
                    ).flatten()

                    # Include CAN-DELIVER non-fluents (per-truck x per-location)
                    can_deliver_reordered = _ego_reorder_truck_axis(
                        jnp.atleast_2d(self.can_deliver), agent_idx
                    ).flatten()

                    obs_agent_values.append(truck_at_reordered)
                    obs_agent_values.append(shop_pizzas)
                    obs_agent_values.append(orders_remaining)
                    obs_agent_values.append(pizzas_in_truck_reordered)
                    obs_agent_values.append(collision_reordered)
                    obs_agent_values.append(done_delivering_reordered)
                    obs_agent_values.append(can_deliver_reordered)

                    obs_full_values.append(truck_at_reordered)
                    obs_full_values.append(shop_pizzas)
                    obs_full_values.append(orders_remaining)
                    obs_full_values.append(pizzas_in_truck_reordered)
                    obs_full_values.append(collision_reordered)
                    obs_full_values.append(done_delivering_reordered)
                    obs_full_values.append(can_deliver_reordered)

                    obs_agent[agent] = jnp.concatenate(obs_agent_values, dtype=self.observation_spaces[agent].dtype)
                    obs_full[agent] = jnp.concatenate(obs_full_values, dtype=self.observation_spaces[agent].dtype)
                else:
                    # Non-ego-centric: keep original ordering
                    truck_at = observation['truckAt'].flatten()
                    pizzas_in_truck = jnp.atleast_1d(observation['numPizzasInTruck']).flatten()
                    collision = jnp.atleast_1d(observation['collision']).flatten()
                    done_delivering = jnp.atleast_1d(observation['doneDelivering']).flatten()

                    can_deliver = jnp.atleast_2d(self.can_deliver).flatten()

                    obs_agent_values.append(truck_at)
                    obs_agent_values.append(shop_pizzas)
                    obs_agent_values.append(orders_remaining)
                    obs_agent_values.append(pizzas_in_truck)
                    obs_agent_values.append(collision)
                    obs_agent_values.append(done_delivering)
                    obs_agent_values.append(can_deliver)

                    obs_full_values.append(truck_at)
                    obs_full_values.append(shop_pizzas)
                    obs_full_values.append(orders_remaining)
                    obs_full_values.append(pizzas_in_truck)
                    obs_full_values.append(collision)
                    obs_full_values.append(done_delivering)
                    obs_full_values.append(can_deliver)

                    obs_agent[agent] = jnp.concatenate(obs_agent_values, dtype=self.observation_spaces[agent].dtype)
                    obs_full[agent] = jnp.concatenate(obs_full_values, dtype=self.observation_spaces[agent].dtype)
            return obs_agent, obs_full
        else:
            raise NotImplementedError("Non-vectorized observations not implemented yet.")

    @partial(jax.jit, static_argnums=(0,))
    def _actions_to_rddl(self, actions: Dict[str, jnp.array]) -> Dict[str, jnp.array]:
        """Convert agent actions (discrete indices) to RDDL action dictionary.

        Args:
            actions: Dict {agent_id: action_index} where action_index selects
                    the integer-valued action-num for that agent.

        Returns:
            Dictionary with action-num array keyed by the action fluent name.
        """
        action_array = jnp.zeros((self.num_agents,), dtype=jnp.int32)
        for agent_idx, agent in enumerate(self.agents):
            action_array = action_array.at[agent_idx].set(actions[agent].squeeze())

        if self._debug_actions:
            jax.debug.print("[PizzaWrapper._actions_to_rddl] action-num={idx}", idx=action_array)

        return {self.rddl_action_keys[0]: action_array}

    @partial(jax.jit, static_argnums=(0,))
    def _extract_rewards(self, reward: float) -> Dict[str, float]:
        """Extract and process rewards for each agent.

        Args:
            reward: Scalar reward from the environment

        Returns:
            A dictionary {agent: reward_value}
        """
        reward_array = jnp.asarray(reward)
        if self.share_rewards:
            total_reward = jnp.sum(reward_array)
            rewards = {agent: total_reward for agent in self.agents}
        else:
            if reward_array.ndim == 0 or reward_array.size == 1:
                scalar_reward = reward_array.reshape(-1)[0]
                rewards = {agent: scalar_reward for agent in self.agents}
            else:
                rewards = {agent: reward_array[i] for i, agent in enumerate(self.agents)}
        return rewards

    @partial(jax.jit, static_argnums=(0,))
    def _extract_dones(self, timestep, observation: Optional[Dict[str, jnp.ndarray]] = None) -> Dict[str, bool]:
        """Create done dict using per-truck completion and episode termination.

        Per-agent done is driven by `doneDelivering` and forced True when the
        episode ends (`done` or `truncated`). The global `__all__` flag remains
        the episode-level termination signal.
        """
        done = timestep.done  # Episode terminal condition reached
        terminal = timestep.truncated  # Episode limit reached
        episode_done = done | terminal

        if observation is not None and 'doneDelivering' in observation:
            per_agent_done = jnp.atleast_1d(observation['doneDelivering']).astype(jnp.bool_).reshape(-1)
        else:
            per_agent_done = jnp.zeros((self.num_agents,), dtype=jnp.bool_)

        dones = {
            agent: jnp.logical_or(per_agent_done[i], episode_done)
            for i, agent in enumerate(self.agents)
        }
        dones["__all__"] = episode_done
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

        The input dictionary has action masks with shape:
        - 'action-num': (num_agents, num_actions)

        Returns:
            Dictionary {agent: flat_mask} where flat_mask is a 1D boolean array
        """
        if self.vectorized:
            # pyRDDLGym_jax can retain previous action-fluent assignments in subs,
            # which contaminates availability checks. Reset action fluents to noop
            # before querying available actions.
            noop_actions = self.env.noop_actions if hasattr(self.env, 'noop_actions') else {}
            sanitized_subs = {**env_state.subs, **noop_actions}
            sanitized_state = env_state.replace(subs=sanitized_subs)
            avail_action_mask = self.env.get_available_actions(sanitized_state)

            if self._debug_actions:
                jax.debug.print(
                    "[PizzaWrapper._extract_avail_actions] sanitized action-num={act}",
                    act=avail_action_mask.get(self.rddl_action_keys[0], jnp.array([]))
                )

            action_mask = avail_action_mask[self.rddl_action_keys[0]]
            return {agent: action_mask[i] for i, agent in enumerate(self.agents)}

        else:
            raise NotImplementedError("Non-vectorized avail_actions not implemented yet.")

    def _convert_rddl_action_spec_to_jaxmarl_space(self, space: pyRDDLGym_jax.core.spaces.Dict) -> jaxmarl_spaces.Space:
        """Convert RDDL action space to JaxMARL discrete space.
        Each agent has N possible actions indexed 0..N-1.
        """
        if self.vectorized:
            return jaxmarl_spaces.Discrete(num_categories=int(space.high[0]) + 1, dtype=space.dtype)
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

            # Include CAN-DELIVER non-fluent: per-truck x per-location boolean matrix
            # which we append to each agent's observation vector in _extract_observations.
            num_locs = len(self.locations_list)
            obs_size += (self.num_agents * num_locs)
            obs_full_size += (self.num_agents * num_locs)

            # Create Box space
            observation_space = jaxmarl_spaces.Box(
                low=0,
                high=jnp.inf,
                shape=(obs_size,),
                dtype=jnp.float32
            )

            observation_full_space = jaxmarl_spaces.Box(
                low=0,
                high=jnp.inf,
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
        Render the current environment state.

        Args:
            env_state: The current environment state.
            save_frame: If True and a MovieGenerator exists, persist this frame.

        Returns:
            PIL image for the rendered frame, or None when visualization is disabled.
        """
        if self.env._visualizer is None:
            return None

        # pyRDDLGym_jax versions differ in state extraction APIs.
        if hasattr(self.env, 'get_state'):
            state = self.env.get_state(env_state)
        elif hasattr(env_state, 'state'):
            state = env_state.state
        elif hasattr(env_state, 'subs'):
            state = env_state.subs
        else:
            raise AttributeError("Unable to extract render state from env_state: expected get_state/state/subs")

        try:
            state = self.env.model.ground_vars_with_values(state)
        except Exception:
            # Fallback for environments where extracted state is already grounded.
            pass

        state = {k: (np.asarray(v) if hasattr(v, '__array__') else v) for k, v in state.items()}

        subs = getattr(env_state, 'subs', {})
        image = self.env._visualizer.render(state, subs)

        if save_frame and self.env._movie_generator is not None and image is not None:
            self.env._movie_generator.save_frame(image)

        return image

    def reset_render(self):
        if self.env._visualizer is not None and hasattr(self.env._visualizer, 'reset'):
            self.env._visualizer.reset()

    def animate(self, states, dones, num_episodes, extra_dir=None, fps=1, loop_count=0, debug=False):
        """
        Create episode animations from a batch of states.

        Saves GIF for each episode and attempts MP4 export when moviepy is available.
        """
        init_state, state = states

        base_dir = self._render_dir
        if extra_dir is not None:
            base_dir = os.path.join(base_dir, extra_dir)
        os.makedirs(base_dir, exist_ok=True)

        for ep_idx in range(num_episodes):
            episode_frames = []

            init_env_state = self.unbatch_init_envstate(init_state, idx1=0, idx2=ep_idx)
            env_states = self.unbatch_envstate(state, idx1=0, idx2=ep_idx)

            first_frame = self.render(init_env_state, save_frame=False)
            if first_frame is not None:
                episode_frames.append(first_frame)

            for step_idx in range(self.env.model.horizon):
                frame = self.render(env_states[step_idx], save_frame=False)
                if frame is not None:
                    episode_frames.append(frame)

                if bool(dones[0, ep_idx, step_idx]):
                    break

            if len(episode_frames) == 0:
                continue

            gif_path = os.path.join(base_dir, f"{self._render_name}_ep_{ep_idx}.gif")
            episode_frames[0].save(
                gif_path,
                save_all=True,
                append_images=episode_frames[1:],
                duration=1000 // max(1, fps),
                loop=loop_count,
            )

            try:
                from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

                mp4_path = os.path.join(base_dir, f"{self._render_name}_ep_{ep_idx}.mp4")
                frame_arrays = [np.asarray(frame) for frame in episode_frames]
                clip = ImageSequenceClip(frame_arrays, fps=max(1, fps))
                clip.write_videofile(mp4_path, codec='libx264', audio=False, logger=None)
                clip.close()
            except Exception:
                # MP4 export is optional; GIF remains the canonical fallback.
                pass

            if debug:
                debug_dir = os.path.join(base_dir, f"{self._render_name}_ep_{ep_idx}_frames")
                os.makedirs(debug_dir, exist_ok=True)
                for frame_idx, frame in enumerate(episode_frames):
                    frame.save(os.path.join(debug_dir, f"frame_{frame_idx:04d}.png"))

            self.reset_render()

    @staticmethod
    def unbatch_init_envstate(batched_state, idx1=None, idx2=None):
        state = batched_state
        if idx1 is not None:
            state = jax.tree.map(lambda x: x[idx1], state)
        if idx2 is not None:
            state = jax.tree.map(lambda x: x[idx2], state)
        return state

    @staticmethod
    def unbatch_envstate(batched_state, idx1=None, idx2=None, unbatch_axis=0):
        state = batched_state
        if idx1 is not None:
            state = jax.tree.map(lambda x: x[idx1], state)
        if idx2 is not None:
            state = jax.tree.map(lambda x: x[idx2], state)

        batch_size = jax.tree.leaves(state)[0].shape[unbatch_axis]
        if unbatch_axis == 0:
            return [jax.tree.map(lambda x: x[i], state) for i in range(batch_size)]
        return [jax.tree.map(lambda x: jnp.take(x, i, axis=unbatch_axis), state) for i in range(batch_size)]

if __name__ == "__main__":
    # Simple test of the PizzaWrapper with JAX
    import os
    from pyRDDLGym_jax.core.env import JaxRDDLEnv

    domain_path = os.path.join(os.path.dirname(__file__), 'pizza_v2_domain.rddl')
    instance_path = os.path.join(os.path.dirname(__file__), 'pizza_v2_instance_all.rddl')

    # Create the JAX RDDL environment
    jax_env = JaxRDDLEnv(domain=domain_path, instance=instance_path)

    # Wrap it
    wrapper = PizzaWrapper(jax_env, render=False)

    # Test reset and step
    key = jax.random.PRNGKey(0)
    obs, state = wrapper.reset(key)
    obs_agent, obs_full = obs

    print(f"Number of agents: {wrapper.num_agents}")
    for agent in wrapper.agents:
        print(f"Agent: {agent}")
        print(f"  Observation space: {wrapper.observation_space(agent, 'agent')}")
        print(f"  Action space: {wrapper.action_space(agent)}")

    print(f"\nInitial observation keys: {obs_agent.keys()}")
    print(f"Initial observation shape: {obs_agent[wrapper.agents[0]].shape}")

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
        obs_agent, obs_full = obs

        print(f"  Rewards: {rewards}")
        print(f"  Done: {dones['__all__']}")
        print(f"  Observation shape: {obs_agent[wrapper.agents[0]].shape}")

        if dones["__all__"]:
            print("  Episode finished!")
            break

    print("\nTest completed successfully!")
