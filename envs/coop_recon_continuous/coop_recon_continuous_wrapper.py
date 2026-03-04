from functools import partial
from typing import Dict, Any, Tuple, Optional
import os

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
            max_speed (float): Maximum agent speed. Defaults to 1.0. (increased speed, used to be 0.2)
            detection_radius (float): Radius for detection actions. Defaults to 0.2. (increased radius, used to be 0.15)
            horizon (int): Maximum episode length. Defaults to 40. (increased horizon, used to be 30)
    """

    def __init__(self, **kwargs):
        self.share_rewards = kwargs.get('share_rewards', False)
        self.dt = kwargs.get('dt', 0.05)
        self.max_speed = kwargs.get('max_speed', 1.0)
        self.detection_radius = kwargs.get('detection_radius', 0.2)
        self.horizon = kwargs.get('horizon', 40)
        # "any": done when ANY picture taken (use for SAP — frozen agent never takes a pic)
        # "all": done when ALL pictures taken (use for joint — both agents must complete)
        self.done_condition = kwargs.get('done_condition', 'all')
        # Gaussian movement noise std as a fraction of max step size (max_speed * dt).
        # e.g. 0.1 = 10% noise, so actual std = 0.1 * 1.0 * 0.05 = 0.005 units. 0.0 = off.
        self.movement_noise_std = kwargs.get('movement_noise_std', 0.0)

        self._render = kwargs.get('render', False)
        self._render_name = kwargs.get('render_name', "coop_recon_continuous")
        self._render_dir = kwargs.get('render_dir', "render")

        self._ego_centric_obs = kwargs.get('ego_centric_obs', False)

        self.num_agents = 2
        self.agents = ["agent_0", "agent_1"]
        self.name = "CoopReconContinuous"

        obs_size = 16 if self._ego_centric_obs else 18
        # Observation Box bounds simplified for dynamically sized array
        self.observation_spaces = {}
        self.observation_full_spaces = {}
        for agent_idx, agent in enumerate(self.agents):
            obs_space = jaxmarl_spaces.Box(
                low=-1.0, high=1.0,
                shape=(obs_size,),
                dtype=jnp.float32
            )
            obs_full_space = jaxmarl_spaces.Box(
                low=-1.0, high=1.0,
                shape=(18,),
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
            [0, 0.5],    # 1: north
            [0, -0.5],   # 2: south
            [0.5, 0],    # 3: east
            [-0.5, 0],   # 4: west
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

        # Random initial positions with spatial partitioning
        key_x0, key_y0, key_x1, key_y1 = jax.random.split(pos_key, 4)
        
        pos_x_0 = jax.random.uniform(key_x0, shape=(1,), minval=0.0, maxval=0.49)
        pos_y_0 = jax.random.uniform(key_y0, shape=(1,), minval=0.0, maxval=1.0)
        
        pos_x_1 = jax.random.uniform(key_x1, shape=(1,), minval=0.51, maxval=1.0)
        pos_y_1 = jax.random.uniform(key_y1, shape=(1,), minval=0.0, maxval=1.0)
        
        positions = jnp.stack([
            jnp.concatenate([pos_x_0, pos_y_0]),
            jnp.concatenate([pos_x_1, pos_y_1])
        ])
        
        velocities = jnp.zeros((2, 2), dtype=jnp.float32)

        # Random goal positions (One for each agent per PI specs)
        key_gx0, key_gy0, key_gx1, key_gy1 = jax.random.split(goal_key, 4)
        goal_x_0 = jax.random.uniform(key_gx0, shape=(1,), minval=0.0, maxval=0.49)
        goal_y_0 = jax.random.uniform(key_gy0, shape=(1,), minval=0.0, maxval=1.0)
        
        goal_x_1 = jax.random.uniform(key_gx1, shape=(1,), minval=0.51, maxval=1.0)
        goal_y_1 = jax.random.uniform(key_gy1, shape=(1,), minval=0.0, maxval=1.0)
        
        goal_pos = jnp.stack([
            jnp.concatenate([goal_x_0, goal_y_0]),
            jnp.concatenate([goal_x_1, goal_y_1])
        ])

        # Task state arrays (shape 2 for 2 goals)
        detected_water = jnp.array([False, False])
        detected_life = jnp.array([False, False])
        picture_taken = jnp.array([False, False])
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

        # Add Gaussian movement noise (scaled by dt so movement_noise_std is a fraction of max step size)
        # e.g. movement_noise_std=0.1 → actual std = 0.1 * dt = 0.005 units (10% of max step 0.05)
        key, key_noise = jax.random.split(key)
        noise = jax.random.normal(key_noise, shape=new_positions.shape) * self.movement_noise_std * self.dt
        new_positions = new_positions + noise
        
        # Apply spatial partitioning Constraints
        # Agent 0: x in [0.0, 0.5]
        # Agent 1: x in [0.5, 1.0]
        new_pos_x = jnp.array([
            jnp.clip(new_positions[0, 0], 0.0, 0.49),
            jnp.clip(new_positions[1, 0], 0.51, 1.0)
        ])
        new_pos_y = jnp.clip(new_positions[:, 1], 0.0, 1.0)
        
        new_positions = jnp.stack([new_pos_x, new_pos_y], axis=1)

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
        dists_to_goal = jnp.linalg.norm(positions[:, None, :] - goal_pos[None, :, :], axis=-1)  # (2 agents, 2 goals)
        at_goal = dists_to_goal < self.detection_radius  # (2, 2)
        
        # Opaque wall at x=0.5: line of sight cannot penetrate the geographic boundary
        is_agent_left = positions[:, 0] < 0.5
        is_goal_left = goal_pos[:, 0] < 0.5
        same_side = is_agent_left[:, None] == is_goal_left[None, :]
        at_goal = at_goal & same_side

        # Initialize rewards with step penalty (-0.01 to match PyTorch)
        reward_values = jnp.full(self.num_agents, -0.01, dtype=jnp.float32)

        # Split keys for each agent
        key_water_0, key_water_1 = jax.random.split(key_water)
        key_life_0, key_life_1 = jax.random.split(key_life)
        water_keys = jnp.array([key_water_0, key_water_1])
        life_keys = jnp.array([key_life_0, key_life_1])

        # Vectorized detection checks
        w_action = (actions == 5)[:, None] # (2 agents, 1)
        l_action = (actions == 6)[:, None] # (2 agents, 1)
        p_action = (actions == 7)[:, None] # (2 agents, 1)
        
        is_detect_water = w_action & at_goal & ~detected_water[None, :]
        is_detect_life  = l_action & at_goal & detected_water[None, :] & ~detected_life[None, :]
        is_picture      = p_action & at_goal & detected_life[None, :]  & ~picture_taken[None, :]

        # Stochastic success for water and life detection (vectorized over agents)
        water_success = jax.vmap(lambda k: jax.random.bernoulli(k, 0.8))(water_keys)[:, None]  # (2, 1)
        life_success = jax.vmap(lambda k: jax.random.bernoulli(k, 0.8))(life_keys)[:, None]    # (2, 1)

        # Determine which detections succeeded
        water_detected = is_detect_water & water_success  # (2 agents, 2 goals)
        life_detected = is_detect_life & life_success     # (2 agents, 2 goals)
        picture_succeeded = is_picture                    # (2 agents, 2 goals)

        # Update global state (any agent can trigger these on any goal)
        new_detected_water = detected_water | jnp.any(water_detected, axis=0) # (2 goals,)
        new_detected_life = detected_life | jnp.any(life_detected, axis=0)    # (2 goals,)
        new_picture_taken = picture_taken | jnp.any(picture_succeeded, axis=0)# (2 goals,)

        # Update rewards - Give agent i reward if they triggered the detection on any goal
        reward_values = reward_values + jnp.sum(jnp.where(water_detected, 1.0, 0.0), axis=1)
        reward_values = reward_values + jnp.sum(jnp.where(life_detected, 1.0, 0.0), axis=1)
        reward_values = reward_values + jnp.sum(jnp.where(picture_succeeded, 10.0, 0.0), axis=1)
        
        # Collision penalty removed to match PyTorch implementation

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
            obs_full: Dictionary of full observations for each agent
        """
        obs_dict = {}
        obs_full_dict = {}
        for agent_idx, agent in enumerate(self.agents):
            if self._ego_centric_obs:
                # Ego-centric observation for agent 0: relative pos to agent 1, own vel, other vel, goal pos, task states
                # Ego-centric observation for agent 1: relative pos to agent 0, own vel, other vel, goal pos, task states
                other_agent_idx = 1 - agent_idx
                obs_dict[agent] = jnp.concatenate([
                    env_state.positions[agent_idx] - env_state.positions[other_agent_idx], # Relative position
                    env_state.velocities[agent_idx],                                       # Own velocity
                    env_state.velocities[other_agent_idx],                                 # Other agent's velocity
                    env_state.goal_pos.flatten(),                                          # [g1x, g1y, g2x, g2y]
                    env_state.detected_water.astype(jnp.float32),                          # [w1, w2]
                    env_state.detected_life.astype(jnp.float32),                           # [l1, l2]
                    env_state.picture_taken.astype(jnp.float32)                            # [p1, p2]
                ], dtype=jnp.float32)
            else:
                # Observation mapped to the agent's absolute identity: [OWN_POS, OTHER_POS]
                # To prevent the shared MLP from having to learn asymmetric routing logic, 
                # we also map goals and task states to an ego-centric [OWN, OTHER] frame of reference.
                other_agent_idx = 1 - agent_idx
                obs_dict[agent] = jnp.concatenate([
                    env_state.positions[agent_idx],                                        # Own absolute position
                    env_state.positions[other_agent_idx],                                  # Other agent's absolute position
                    env_state.velocities[agent_idx],                                       # Own velocity
                    env_state.velocities[other_agent_idx],                                 # Other agent's velocity
                    env_state.goal_pos[agent_idx],                                         # Own goal
                    env_state.goal_pos[other_agent_idx],                                   # Other goal
                    jnp.array([env_state.detected_water[agent_idx], env_state.detected_water[other_agent_idx]], dtype=jnp.float32),
                    jnp.array([env_state.detected_life[agent_idx], env_state.detected_life[other_agent_idx]], dtype=jnp.float32),
                    jnp.array([env_state.picture_taken[agent_idx], env_state.picture_taken[other_agent_idx]], dtype=jnp.float32)
                ], dtype=jnp.float32)

            # Full observation (same for both agents, always includes all global info)
            obs_full_dict[agent] = jnp.concatenate([
                env_state.positions.flatten(),      # [x1, y1, x2, y2]
                env_state.velocities.flatten(),     # [vx1, vy1, vx2, vy2]
                env_state.goal_pos.flatten(),       # [g1x, g1y, g2x, g2y]
                env_state.detected_water.astype(jnp.float32),
                env_state.detected_life.astype(jnp.float32),
                env_state.picture_taken.astype(jnp.float32)
            ], dtype=jnp.float32)
        return obs_dict, obs_full_dict

    def _get_dones(self, env_state: CoopReconEnvState) -> Dict[str, bool]:
        """Compute done flags for each agent.

        Args:
            env_state: Current environment state

        Returns:
            dones: Dictionary of done flags
        """
        if self.done_condition == 'any':
            done = jnp.any(env_state.picture_taken) | (env_state.timestep >= self.horizon)
        else:  # 'all'
            done = jnp.all(env_state.picture_taken) | (env_state.timestep >= self.horizon)
        dones = {agent: done for agent in self.agents}
        dones["__all__"] = done
        return dones

    def _get_avail_actions(self, env_state: CoopReconEnvState) -> Dict[str, jnp.ndarray]:
        """Get available actions for each agent.
        Actions 0-4 (movement/noop) are always available.
        Actions 5-7 (tasks) are dynamically masked based on distance and sequential state.

        Args:
            env_state: Current environment state

        Returns:
            avail_actions: Dictionary of available action masks
        """
        dists_to_goal = jnp.linalg.norm(env_state.positions[:, None, :] - env_state.goal_pos[None, :, :], axis=-1)
        at_goal = dists_to_goal < self.detection_radius
        
        # Opaque wall at x=0.5: line of sight cannot penetrate the geographic boundary
        is_agent_left = env_state.positions[:, 0] < 0.5
        is_goal_left = env_state.goal_pos[:, 0] < 0.5
        same_side = is_agent_left[:, None] == is_goal_left[None, :]
        at_goal = at_goal & same_side
        
        avail_actions = {}
        for i, agent in enumerate(self.agents):
            # Once this agent's own goal's picture is taken, force noop (stop moving)
            agent_done = env_state.picture_taken[i]

            can_water = jnp.any(at_goal[i] & ~env_state.detected_water)
            can_life = jnp.any(at_goal[i] & env_state.detected_water & ~env_state.detected_life)
            can_pic = jnp.any(at_goal[i] & env_state.detected_life & ~env_state.picture_taken)

            full_mask = jnp.array([
                True, True, True, True, True,  # 0-4: noop, directions
                can_water, can_life, can_pic   # 5-7: conditional tasks
            ], dtype=jnp.float32)
            noop_mask = jnp.array([True, False, False, False, False, False, False, False], dtype=jnp.float32)

            # If agent's own goal is done, force noop; otherwise use full mask
            mask = jnp.where(agent_done, noop_mask, full_mask)
            avail_actions[agent] = mask

        return avail_actions

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
        if debug:
            import logging
            from envs.coop_recon_continuous.CoopReconContinuousViz import CoopReconContinuousViz
            log = logging.getLogger(__name__)
            try:
                init_state, env_states = states
                
                base_dir = self._render_dir
                if extra_dir is not None:
                    base_dir = os.path.join(base_dir, extra_dir)
                os.makedirs(base_dir, exist_ok=True)
                
                for ep in range(num_episodes):
                    positions = env_states.positions[0, ep] # (max_steps, 2, 2)
                    goals = env_states.goal_pos[0, ep] # (max_steps, 2, 2)
                    water = env_states.detected_water[0, ep] # (max_steps, 2)
                    life = env_states.detected_life[0, ep] # (max_steps, 2)
                    picture = env_states.picture_taken[0, ep] # (max_steps, 2)
                    
                    viz = CoopReconContinuousViz()
                    frames = []
                    max_steps = positions.shape[0]
                    
                    for t in range(max_steps):
                        pos_0 = positions[t, 0]
                        pos_1 = positions[t, 1]
                        g_0 = goals[t, 0]
                        g_1 = goals[t, 1]
                        
                        w = [bool(water[t, 0]), bool(water[t, 1])]
                        l = [bool(life[t, 0]), bool(life[t, 1])]
                        p = [bool(picture[t, 0]), bool(picture[t, 1])]
                        
                        state_layout = {
                            'agent_positions': [(pos_0[0], pos_0[1]), (pos_1[0], pos_1[1])],
                            'goal_positions': [(g_0[0], g_0[1]), (g_1[0], g_1[1])],
                            'detected_water': w,
                            'detected_life': l,
                            'picture_taken': p
                        }
                        
                        frame = viz.render(state_layout)
                        frames.append(frame)
                        
                        if bool(dones[0, ep, t]):
                            break

                    frames[0].save(
                        os.path.join(base_dir, f"{self._render_name}_ep_{ep}.gif"),
                        save_all=True,
                        append_images=frames[1:],
                        duration=1000 // fps,
                        loop=loop_count
                    )
                log.info(f"Successfully saved {num_episodes} evaluation GIFs to {base_dir}")
                
            except Exception as e:
                log.error(f"Diagnostic evaluation printing/rendering failed: {e}")
