from functools import partial
from typing import Dict, Any, Tuple, Optional

import chex
import hydra
import math
import os
import jax
import jax.numpy as jnp
import numpy as np

from flax.struct import dataclass
from envs.rddl.jax_rddl_env import JaxRDDLEnv, EnvState
from jaxmarl.environments import spaces as jaxmarl_spaces

from envs.base_env import BaseEnv
from envs.rddl.pizza_v2.PizzaV2MultiAgentViz import PizzaV2MultiAgentVisualizer


@dataclass
class WrappedEnvState:
    env_state: Any
    base_return_so_far: jnp.ndarray
    avail_actions: jnp.ndarray
    step: jnp.array
    collisions_so_far: jnp.ndarray  # accumulated collision count per truck, zeroed on reset


class PizzaWrapper(BaseEnv):
    """RDDL pizza_v2 environment wrapped to the JaxMARL interface.

    Expects a new-style domain where each truck has a single integer action
    (like the grid/rover domains).  The action key is read from the RDDL action
    space at init time so no hard-coded name is needed.

    Args:
        args[0]: JaxRDDLEnv instance.
        kwargs:
            share_rewards (bool): sum rewards across agents. Default False.
            vectorized (bool): use vectorized obs/action format. Default True.
            render (bool): enable visualizer. Default False.
            ego_centric_obs (bool): reorder truck axis so own truck is first. Default True.
    """

    def __init__(self, *args, **kwargs):
        if not args or not isinstance(args[0], JaxRDDLEnv):
            raise ValueError("First argument must be a JaxRDDLEnv instance")

        self.env = args[0]
        self.vectorized = kwargs.get('vectorized', True)
        self.share_rewards = kwargs.get('share_rewards', False)
        self._render = kwargs.get('render', False)
        self._render_name = kwargs.get('render_name', "pizza_v2")
        self._render_dir = kwargs.get('render_dir', "pizza_v2")
        self._ego_centric_obs = kwargs.get('ego_centric_obs', True)

        self.horizon = self.env.horizon
        self.name = self.env.__class__.__name__
        self.rddl_agent_names = self.env.model.type_to_objects['truck']
        # Single action key for new integer-action domain (sorted for determinism)
        self.rddl_action_keys = sorted(self.env.action_space.keys())
        self.num_agents = len(self.rddl_agent_names)
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]

        # --- SAP controllability detection (mirrors grid_10x10_wrapper) ---
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
        # True when exactly one truck is controllable — this is a SAP instance
        self.single_agent_projection = (
            len(self.controlled_agents) == 1 and self.uncontrolled_agents_exist
        )

        # --- Domain constants ---
        self.collision_penalty = self.env.model.non_fluents.get('COLLISION-PENALTY', -220.0)
        self.step_penalty = self.env.model.non_fluents.get('STEP-PENALTY', -1.0)
        self.rddl_location_names = self.env.model.type_to_objects['location']
        # action-num encoding: 0=noop, 1=load, 2=deliver, 3..2+MAX-CONNECTIONS=drive
        self._max_connections = int(self.env.model.non_fluents.get('MAX-CONNECTIONS', 4))
        self._num_actions = 3 + self._max_connections

        # --- Observation key validation ---
        self._obs_keys = [
            'truckAt',
            'numShopPizzas',
            'numOrdersRemaining',
            'numPizzasInTruck',
            'collision',
            'doneDelivering',
        ]
        obs_space_keys = set(self.env.observation_space.keys())
        missing = [k for k in self._obs_keys if k not in obs_space_keys]
        if missing:
            raise ValueError(f"pizza_v2 env is missing observation keys: {missing}")

        # --- Render ---
        if self._render:
            self._render_dir = os.path.join(
                hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
                kwargs.get('render_dir', "render"),
            )
            os.makedirs(self._render_dir, exist_ok=True)
            viz = PizzaV2MultiAgentVisualizer(self.env.model)
            self.env.set_visualizer(visualizer=viz)

        # --- Spaces ---
        self.observation_spaces = {}
        self.observation_full_spaces = {}
        for agent in self.agents:
            agent_obs, full_obs = self._convert_rddl_obs_spec_to_jaxmarl_space(
                self.env.observation_space
            )
            self.observation_spaces[agent] = agent_obs
            self.observation_full_spaces[agent] = full_obs

        self.action_spaces = {
            agent: self._convert_rddl_action_spec_to_jaxmarl_space(
                self.env.action_space[self.rddl_action_keys[0]]
            )
            for agent in self.agents
        }

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey):
        env_state, timestep = self.env.reset(key)
        obs = self._extract_observations(timestep.observation)
        state = WrappedEnvState(
            env_state,
            jnp.zeros(self.num_agents),
            self._extract_avail_actions(env_state),
            env_state.timestep,
            jnp.zeros(self.num_agents, dtype=jnp.int32),
        )
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: WrappedEnvState,
        actions: Dict[str, chex.Array],
        reset_state: Optional[WrappedEnvState] = None,
    ) -> Tuple[Dict[str, chex.Array], WrappedEnvState, Dict[str, float], Dict[str, bool], Dict]:
        key, key_reset = jax.random.split(key)

        actions_rddl = self._actions_to_rddl(actions)
        env_state, timestep = self.env.step(state.env_state, actions_rddl)

        avail_actions = self._extract_avail_actions(env_state)
        done = self._extract_dones(timestep, timestep.observation)

        # Accumulate collisions every step; auto-reset zeroes the counter on episode start.
        new_episode_collisions = (
            state.collisions_so_far
            + jnp.atleast_1d(timestep.observation['collision']).astype(jnp.int32)
        )

        state_st = WrappedEnvState(
            env_state,
            jnp.zeros(self.num_agents),
            avail_actions,
            env_state.timestep,
            new_episode_collisions,
        )
        obs_st = self._extract_observations(timestep.observation)
        reward = self._extract_rewards(timestep.reward)

        info = {}
        info['pre_reset_state'] = state_st
        info['pre_reset_obs'] = obs_st
        info['returned_episode_collisions'] = state_st.collisions_so_far

        # Auto-reset on episode end
        obs, state = jax.tree.map(
            lambda x, y: jax.lax.select(done["__all__"], x, y),
            self.reset(key_reset),
            (obs_st, state_st),
        )
        return obs, state, reward, done, info

    # ------------------------------------------------------------------
    # Space accessors
    # ------------------------------------------------------------------

    def observation_space(self, agent: str, observation_type: str = "agent") -> jaxmarl_spaces.Space:
        if observation_type == "agent":
            return self.observation_spaces[agent]
        elif observation_type == "full":
            return self.observation_full_spaces[agent]
        raise ValueError(f"Unknown observation_type: {observation_type}")

    def action_space(self, agent: str):
        return self.action_spaces[agent]

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: WrappedEnvState) -> Dict[str, jnp.ndarray]:
        return state.avail_actions

    @partial(jax.jit, static_argnums=(0,))
    def get_step_count(self, state: WrappedEnvState) -> jnp.array:
        return state.step

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @partial(jax.jit, static_argnums=(0,))
    def _actions_to_rddl(self, actions: Dict[str, jnp.array]) -> Dict[str, jnp.array]:
        """Stack per-agent integer actions into a single (num_agents,) array."""
        action_array = jnp.zeros((self.num_agents,), dtype=jnp.int32)
        for agent_idx, agent in enumerate(self.agents):
            action_array = action_array.at[agent_idx].set(actions[agent].squeeze())
        return {self.rddl_action_keys[0]: action_array}

    @partial(jax.jit, static_argnums=(0,))
    def _extract_avail_actions(self, env_state: EnvState) -> Dict[str, jnp.ndarray]:
        """Return per-agent available-action masks from the RDDL env."""
        avail_action_mask = self.env.get_available_actions(env_state)
        return {
            agent: avail_action_mask[self.rddl_action_keys[0]][i]
            for i, agent in enumerate(self.agents)
        }

    @partial(jax.jit, static_argnums=(0,))
    def _extract_observations(self, observation) -> Tuple[Dict, Dict]:
        """Flatten RDDL observations into per-agent float32 vectors."""
        obs_agent = {}
        obs_full = {}

        shop_pizzas = jnp.atleast_1d(observation['numShopPizzas']).flatten()
        orders_remaining = jnp.atleast_1d(observation['numOrdersRemaining']).flatten()

        for agent_idx, agent in enumerate(self.agents):
            if self._ego_centric_obs:
                truck_at = jnp.concatenate([
                    observation['truckAt'][agent_idx:agent_idx + 1],
                    observation['truckAt'][:agent_idx],
                    observation['truckAt'][agent_idx + 1:],
                ], axis=0).flatten()
                pizzas_in_truck = jnp.concatenate([
                    jnp.atleast_1d(observation['numPizzasInTruck'])[agent_idx:agent_idx + 1],
                    jnp.atleast_1d(observation['numPizzasInTruck'])[:agent_idx],
                    jnp.atleast_1d(observation['numPizzasInTruck'])[agent_idx + 1:],
                ]).flatten()
                collision = jnp.concatenate([
                    jnp.atleast_1d(observation['collision'])[agent_idx:agent_idx + 1],
                    jnp.atleast_1d(observation['collision'])[:agent_idx],
                    jnp.atleast_1d(observation['collision'])[agent_idx + 1:],
                ]).flatten()
                done_delivering = jnp.concatenate([
                    jnp.atleast_1d(observation['doneDelivering'])[agent_idx:agent_idx + 1],
                    jnp.atleast_1d(observation['doneDelivering'])[:agent_idx],
                    jnp.atleast_1d(observation['doneDelivering'])[agent_idx + 1:],
                ]).flatten()
            else:
                truck_at = observation['truckAt'].flatten()
                pizzas_in_truck = jnp.atleast_1d(observation['numPizzasInTruck']).flatten()
                collision = jnp.atleast_1d(observation['collision']).flatten()
                done_delivering = jnp.atleast_1d(observation['doneDelivering']).flatten()

            vec = jnp.concatenate([
                truck_at,
                shop_pizzas,
                orders_remaining,
                pizzas_in_truck,
                collision,
                done_delivering,
            ]).astype(self.observation_spaces[agent].dtype)

            obs_agent[agent] = vec
            obs_full[agent] = vec  # no partial observability in current setup

        return obs_agent, obs_full

    @partial(jax.jit, static_argnums=(0,))
    def _extract_rewards(self, reward) -> Dict[str, float]:
        reward_array = jnp.asarray(reward)
        if self.share_rewards:
            total = jnp.sum(reward_array)
            return {agent: total for agent in self.agents}
        if reward_array.ndim == 0 or reward_array.size == 1:
            scalar = reward_array.reshape(-1)[0]
            return {agent: scalar for agent in self.agents}
        return {agent: reward_array[i] for i, agent in enumerate(self.agents)}

    @partial(jax.jit, static_argnums=(0,))
    def _extract_dones(self, timestep, observation=None) -> Dict[str, bool]:
        done = timestep.done
        terminal = timestep.truncated
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

        # In a SAP instance end the episode as soon as the focal agent finishes
        # its delivery zone, rather than waiting for RDDL termination (which
        # requires ALL locations to empty — impossible when the partner's orders
        # never decrease because that truck is uncontrolled).
        if self.single_agent_projection:
            dones["__all__"] = (
                dones[self.agents[self.controlled_agents[0]]] | terminal | done
            )

        return dones

    def _convert_rddl_action_spec_to_jaxmarl_space(
        self, space: Any
    ) -> jaxmarl_spaces.Space:
        """Discrete action space with 3 + MAX-CONNECTIONS categories.

        action-num encoding: 0=noop, 1=load, 2=deliver, 3..2+MAX-CONNECTIONS=drive.
        We compute num_categories from the MAX-CONNECTIONS non-fluent directly rather
        than relying on space.high, because pyRDDLGym may not parse the compound
        action-precondition bound (action-num <= 2 + MAX-CONNECTIONS) automatically.
        """
        return jaxmarl_spaces.Discrete(num_categories=self._num_actions, dtype=jnp.int32)

    def _convert_rddl_obs_spec_to_jaxmarl_space(
        self, space: Any
    ) -> Tuple[jaxmarl_spaces.Space, jaxmarl_spaces.Space]:
        """Compute flat obs size from the known observation keys only."""
        obs_size = sum(
            math.prod(space[k].shape) for k in self._obs_keys if k in space
        )
        box = jaxmarl_spaces.Box(low=0, high=jnp.inf, shape=(obs_size,), dtype=jnp.float32)
        return box, box  # agent obs == full obs (no partial observability)

    # ------------------------------------------------------------------
    # Render / animate
    # ------------------------------------------------------------------

    def render(self, env_state: EnvState, save_frame: bool = True) -> Any:
        if self.env._visualizer is None:
            return None
        if hasattr(self.env, 'get_state'):
            state = self.env.get_state(env_state)
        elif hasattr(env_state, 'subs'):
            state = env_state.subs
        else:
            raise AttributeError("Cannot extract render state from env_state")
        try:
            state = self.env.model.ground_vars_with_values(state)
        except Exception:
            pass
        state = {k: (np.asarray(v) if hasattr(v, '__array__') else v) for k, v in state.items()}
        image = self.env._visualizer.render(state, env_state.subs)
        if save_frame and self.env._movie_generator is not None and image is not None:
            self.env._movie_generator.save_frame(image)
        return image

    def reset_render(self):
        if self.env._visualizer is not None and hasattr(self.env._visualizer, 'reset'):
            self.env._visualizer.reset()

    def animate(self, states, dones, num_episodes, extra_dir=None, fps=1, loop_count=0, debug=False):
        init_state, state = states
        base_dir = self._render_dir
        if extra_dir is not None:
            base_dir = os.path.join(base_dir, extra_dir)
        os.makedirs(base_dir, exist_ok=True)

        for ep_idx in range(num_episodes):
            frames = []
            init_env_state = self.unbatch_init_envstate(init_state, idx1=0, idx2=ep_idx)
            env_states = self.unbatch_envstate(state, idx1=0, idx2=ep_idx)

            frame = self.render(init_env_state, save_frame=False)
            if frame is not None:
                frames.append(frame)

            for step_idx in range(self.env.model.horizon):
                frame = self.render(env_states[step_idx], save_frame=False)
                if frame is not None:
                    frames.append(frame)
                if bool(dones[0, ep_idx, step_idx]):
                    break

            if not frames:
                continue

            gif_path = os.path.join(base_dir, f"{self._render_name}_ep_{ep_idx}.gif")
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=1000 // max(1, fps),
                loop=loop_count,
            )

            try:
                from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
                mp4_path = os.path.join(base_dir, f"{self._render_name}_ep_{ep_idx}.mp4")
                clip = ImageSequenceClip([np.asarray(f) for f in frames], fps=max(1, fps))
                clip.write_videofile(mp4_path, codec='libx264', audio=False, logger=None)
                clip.close()
            except Exception:
                pass

            if debug:
                debug_dir = os.path.join(base_dir, f"{self._render_name}_ep_{ep_idx}_frames")
                os.makedirs(debug_dir, exist_ok=True)
                for i, f in enumerate(frames):
                    f.save(os.path.join(debug_dir, f"frame_{i:04d}.png"))

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
