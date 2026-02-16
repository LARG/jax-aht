from functools import partial
from typing import Dict, Any, Tuple, Optional

import chex
import jax
import jax.numpy as jnp
from flax.struct import dataclass
from jaxmarl.environments import spaces as jaxmarl_spaces

from envs.base_env import BaseEnv, WrappedEnvState


@dataclass
class CoopReconEnvState:
    """Internal state for the cooperative reconnaissance environment."""
    positions: jnp.ndarray  # (2, 2) - agent positions [x, y]
    velocities: jnp.ndarray  # (2, 2) - agent velocities [vx, vy]
    goal_pos: jnp.ndarray  # (2,) - goal position
    detected_water: jnp.ndarray  # scalar bool
    detected_life: jnp.ndarray  # scalar bool
    picture_taken: jnp.ndarray  # scalar bool
    timestep: jnp.ndarray  # scalar int


class CoopReconContinuousWrapper(BaseEnv):
    """JAX-based wrapper for cooperative reconnaissance continuous environment.

    Args:
        **kwargs: Keyword arguments.
            share_rewards (bool): Whether to share rewards between agents. Defaults to False.
            dt (float): Time step for physics. Defaults to 0.05.
            max_speed (float): Maximum agent speed. Defaults to 0.2.
            detection_radius (float): Radius for detection actions. Defaults to 0.15.
            horizon (int): Maximum episode length. Defaults to 30.
    """

    def __init__(self, **kwargs):
        self.share_rewards = kwargs.get('share_rewards', False)
        self.dt = kwargs.get('dt', 0.05)
        self.max_speed = kwargs.get('max_speed', 0.2)
        self.detection_radius = kwargs.get('detection_radius', 0.15)
        self.horizon = kwargs.get('horizon', 30)

        self._render = False #kwargs.get('render', False)

        self._ego_centric_obs = kwargs.get('ego_centric_obs', False)

        self.num_agents = 2
        self.agents = ["agent_0", "agent_1"]
        self.name = "CoopReconContinuous"

        # Observation: [x1, y1, vx1, vy1, x2, y2, vx2, vy2, goal_x, goal_y, water, life, pic]
        self.observation_spaces = {
            agent: jaxmarl_spaces.Box(
                low=jnp.array([0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0], dtype=jnp.float32),
                high=jnp.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=jnp.float32),
                shape=(13,),
                dtype=jnp.float32
            )
            for agent in self.agents
        }

        # Observation: [x1, y1, vx1, vy1, x2, y2, vx2, vy2, goal_x, goal_y, water, life, pic]
        # Observation Full: [x1, y1, vx1, vy1, x2, y2, vx2, vy2, goal_x, goal_y, water, life, pic]
        self.observation_spaces = {}
        self.observation_full_spaces = {}
        for agent_idx, agent in enumerate(self.agents):
            obs_space = jaxmarl_spaces.Box(
                low=jnp.array([0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0], dtype=jnp.float32),
                high=jnp.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=jnp.float32),
                shape=(13,),
                dtype=jnp.float32
            )
            obs_full_space = jaxmarl_spaces.Box(
                low=jnp.array([0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0], dtype=jnp.float32),
                high=jnp.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=jnp.float32),
                shape=(13,),
                dtype=jnp.float32
            )
            self.observation_spaces[agent] = obs_space
            self.observation_full_spaces[agent] = obs_full_space

        # Actions: 8 discrete
        # 0=noop, 1=north, 2=south, 3=east, 4=west, 5=detect_water, 6=detect_life, 7=picture
        self.action_spaces = {
            agent: jaxmarl_spaces.Discrete(num_categories=8, dtype=jnp.int32)
            for agent in self.agents
        }

        # Action to velocity mapping (stored as jax array)
        self.action_to_vel = jnp.array([
            [0, 0],      # 0: noop
            [0, 0.1],    # 1: north
            [0, -0.1],   # 2: south
            [0.1, 0],    # 3: east
            [-0.1, 0],   # 4: west
            [0, 0],      # 5: detect water
            [0, 0],      # 6: detect life
            [0, 0],      # 7: picture
        ], dtype=jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, jnp.ndarray], WrappedEnvState]:
        """Reset the environment to initial state.

        Args:
            key: PRNG key for random initialization

        Returns:
            obs: Dictionary of observations for each agent
            state: Wrapped environment state
        """
        key, pos_key, goal_key = jax.random.split(key, 3)

        # Random initial positions for both agents
        positions = jax.random.uniform(pos_key, shape=(2, 2), minval=0.0, maxval=1.0)
        velocities = jnp.zeros((2, 2), dtype=jnp.float32)

        # Random goal position
        goal_pos = jax.random.uniform(goal_key, shape=(2,), minval=0.0, maxval=1.0)

        # Task state
        detected_water = jnp.array(False)
        detected_life = jnp.array(False)
        picture_taken = jnp.array(False)
        timestep = jnp.array(0, dtype=jnp.int32)

        env_state = CoopReconEnvState(
            positions=positions,
            velocities=velocities,
            goal_pos=goal_pos,
            detected_water=detected_water,
            detected_life=detected_life,
            picture_taken=picture_taken,
            timestep=timestep
        )

        obs = self._get_obs(env_state)
        avail_actions = self._get_avail_actions(env_state)

        state = WrappedEnvState(
            env_state=env_state,
            base_return_so_far=jnp.zeros(self.num_agents, dtype=jnp.float32),
            avail_actions=avail_actions,
            step=timestep
        )

        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: WrappedEnvState,
        actions: Dict[str, chex.Array],
        reset_state: Optional[WrappedEnvState] = None,
    ) -> Tuple[Dict[str, jnp.ndarray], WrappedEnvState, Dict[str, float], Dict[str, bool], Dict]:
        """Performs step transitions in the environment.

        Args:
            key: PRNG key for stochastic transitions
            state: Current wrapped environment state
            actions: Dictionary of actions for each agent
            reset_state: Optional state to reset to (for auto-reset)

        Returns:
            obs: Observations for each agent
            state: New wrapped environment state
            rewards: Rewards for each agent
            dones: Done flags for each agent
            info: Additional information
        """
        key, key_reset = jax.random.split(key)
        env_state = state.env_state

        # Extract actions as array - use stack to ensure proper shape
        actions_array = jnp.stack([actions[agent] for agent in self.agents], axis=0).astype(jnp.int32).squeeze()

        # Update velocities based on actions
        new_velocities = self._update_velocities(env_state.velocities, actions_array)

        # Update positions
        new_positions = env_state.positions + new_velocities * self.dt
        new_positions = jnp.clip(new_positions, 0.0, 1.0)

        # Process detection/picture actions and compute rewards
        key_water, key_life = jax.random.split(key, 2)
        new_detected_water, new_detected_life, new_picture_taken, rewards = self._process_actions(
            key_water, key_life,
            new_positions,
            env_state.goal_pos,
            actions_array,
            env_state.detected_water,
            env_state.detected_life,
            env_state.picture_taken
        )

        new_timestep = env_state.timestep + 1

        # Build next state
        env_state_next = CoopReconEnvState(
            positions=new_positions,
            velocities=new_velocities,
            goal_pos=env_state.goal_pos,
            detected_water=new_detected_water,
            detected_life=new_detected_life,
            picture_taken=new_picture_taken,
            timestep=new_timestep
        )

        obs_st = self._get_obs(env_state_next)
        avail_actions = self._get_avail_actions(env_state_next)
        dones = self._get_dones(env_state_next)

        state_st = WrappedEnvState(
            env_state=env_state_next,
            base_return_so_far=jnp.zeros(self.num_agents, dtype=jnp.float32), # Log wrapper handles return tracking, so we can set this to zero here
            avail_actions=avail_actions,
            step=new_timestep
        )

        info = {'pre_reset_state': state_st, 'pre_reset_obs': obs_st}

        # Auto-reset environment based on termination
        obs, state = jax.tree.map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y),
            self.reset(key_reset),
            (obs_st, state_st)
        )

        return obs, state, rewards, dones, info

    @partial(jax.jit, static_argnums=(0,))
    def _update_velocities(self, velocities: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        """Update agent velocities based on actions.

        Args:
            velocities: Current velocities (2, 2)
            actions: Actions for each agent (2,)

        Returns:
            new_velocities: Updated velocities (2, 2)
        """
        # Get target velocities from actions
        target_vels = self.action_to_vel[actions]  # (2, 2)

        # Smooth velocity transition
        new_velocities = 0.7 * velocities + 0.3 * target_vels

        # Clip to max speed
        speeds = jnp.linalg.norm(new_velocities, axis=1, keepdims=True)  # (2, 1)
        scale = jnp.minimum(1.0, self.max_speed / (speeds + 1e-8))
        new_velocities = new_velocities * scale

        return new_velocities

    @partial(jax.jit, static_argnums=(0,))
    def _process_actions(
        self,
        key_water: chex.PRNGKey,
        key_life: chex.PRNGKey,
        positions: jnp.ndarray,
        goal_pos: jnp.ndarray,
        actions: jnp.ndarray,
        detected_water: jnp.ndarray,
        detected_life: jnp.ndarray,
        picture_taken: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, float]]:
        """Process detection and picture actions, return new states and rewards.

        Args:
            key_water, key_life: PRNG keys for stochastic detections
            positions: Agent positions (2, 2)
            goal_pos: Goal position (2,)
            actions: Agent actions (2,)
            detected_water, detected_life, picture_taken: Current task state

        Returns:
            new_detected_water, new_detected_life, new_picture_taken: Updated task state
            rewards: Dictionary of rewards for each agent
        """
        # Compute distances to goal
        dists_to_goal = jnp.linalg.norm(positions - goal_pos[None, :], axis=1)  # (2,)
        at_goal = dists_to_goal < self.detection_radius  # (2,)

        # Initialize rewards with step penalty
        reward_values = jnp.full(self.num_agents, -0.1, dtype=jnp.float32)

        # Split keys for each agent
        key_water_0, key_water_1 = jax.random.split(key_water)
        key_life_0, key_life_1 = jax.random.split(key_life)
        water_keys = jnp.array([key_water_0, key_water_1])
        life_keys = jnp.array([key_life_0, key_life_1])

        # Vectorized detection checks
        is_detect_water = (actions == 5) & at_goal & ~detected_water  # (2,)
        is_detect_life = (actions == 6) & at_goal & detected_water & ~detected_life  # (2,)
        is_picture = (actions == 7) & detected_life & ~picture_taken  # (2,)

        # Stochastic success for water and life detection (vectorized)
        water_success = jax.vmap(lambda k: jax.random.bernoulli(k, 0.8))(water_keys)  # (2,)
        life_success = jax.vmap(lambda k: jax.random.bernoulli(k, 0.8))(life_keys)  # (2,)

        # Determine which detections succeeded
        water_detected = is_detect_water & water_success  # (2,)
        life_detected = is_detect_life & life_success  # (2,)
        picture_succeeded = is_picture  # (2,)

        # Update global state (any agent can trigger these)
        new_detected_water = detected_water | jnp.any(water_detected)
        new_detected_life = detected_life | jnp.any(life_detected)
        new_picture_taken = picture_taken | jnp.any(picture_succeeded)

        # Update rewards
        reward_values = reward_values + jnp.where(water_detected, 1.0, 0.0)
        reward_values = reward_values + jnp.where(life_detected, 1.0, 0.0)
        reward_values = reward_values + jnp.where(picture_succeeded, 10.0, 0.0)

        # Convert to dictionary
        rewards = {agent: reward_values[i] for i, agent in enumerate(self.agents)}

        # Apply reward sharing if enabled
        if self.share_rewards:
            total_reward = jnp.sum(reward_values)
            rewards = {agent: total_reward for agent in self.agents}

        return new_detected_water, new_detected_life, new_picture_taken, rewards

    def _get_obs(self, env_state: CoopReconEnvState) -> Dict[str, jnp.ndarray]:
        """Extract observations for each agent.

        Args:
            env_state: Current environment state

        Returns:
            obs: Dictionary of observations for each agent
        """
        obs_full_dict = {}
        for agent_idx, agent in enumerate(self.agents):
            if self._ego_centric_obs:
                obs_full_dict[agent] = jnp.concatenate([
                    env_state.positions[agent_idx:agent_idx+1].flatten(),  # This agent
                    env_state.positions[:agent_idx].flatten(),             # Agents before
                    env_state.positions[agent_idx+1:].flatten(),           # Agents after
                    env_state.velocities[agent_idx:agent_idx+1].flatten(), # This agent
                    env_state.velocities[:agent_idx].flatten(),            # Agents before
                    env_state.velocities[agent_idx+1:].flatten(),          # Agents after
                    env_state.goal_pos,                                    # [goal_x, goal_y]
                    jnp.array([
                        env_state.detected_water.astype(jnp.float32),
                        env_state.detected_life.astype(jnp.float32),
                        env_state.picture_taken.astype(jnp.float32)
                    ])
                ], dtype=jnp.float32)
            else:
                obs_full_dict[agent] = jnp.concatenate([
                    env_state.positions.flatten(),      # [x1, y1, x2, y2]
                    env_state.velocities.flatten(),     # [vx1, vy1, vx2, vy2]
                    env_state.goal_pos,                 # [goal_x, goal_y]
                    jnp.array([
                        env_state.detected_water.astype(jnp.float32),
                        env_state.detected_life.astype(jnp.float32),
                        env_state.picture_taken.astype(jnp.float32)
                    ])
                ], dtype=jnp.float32)
        return obs_full_dict, obs_full_dict

    def _get_dones(self, env_state: CoopReconEnvState) -> Dict[str, bool]:
        """Compute done flags for each agent.

        Args:
            env_state: Current environment state

        Returns:
            dones: Dictionary of done flags
        """
        done = env_state.picture_taken | (env_state.timestep >= self.horizon)
        dones = {agent: done for agent in self.agents}
        dones["__all__"] = done
        return dones

    def _get_avail_actions(self, env_state: CoopReconEnvState) -> Dict[str, jnp.ndarray]:
        """Get available actions for each agent (all actions always available).

        Args:
            env_state: Current environment state

        Returns:
            avail_actions: Dictionary of available action masks
        """
        return {agent: jnp.ones(8, dtype=jnp.float32) for agent in self.agents}

    def observation_space(self, agent: str, observation_type: str = "agent") -> jaxmarl_spaces.Space:
        if observation_type == "agent":
            return self.observation_spaces[agent]
        elif observation_type == "full":
            return self.observation_full_spaces[agent]
        else:
            raise ValueError(f"Unknown observation_type: {observation_type}")

    def action_space(self, agent: str) -> jaxmarl_spaces.Discrete:
        """Get action space for an agent."""
        return self.action_spaces[agent]

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: WrappedEnvState) -> Dict[str, jnp.ndarray]:
        """Returns the available actions for each agent."""
        return state.avail_actions

    @partial(jax.jit, static_argnums=(0,))
    def get_step_count(self, state: WrappedEnvState) -> jnp.ndarray:
        """Returns the step count of the environment."""
        return state.step

    def render(self, env_state, save_frame: bool = True) -> Any:
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
