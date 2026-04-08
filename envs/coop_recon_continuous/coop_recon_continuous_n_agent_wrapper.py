"""N-Agent Cooperative Reconnaissance Environment (Continuous).

Generalization of coop_recon_continuous_wrapper.py to N agents.
Original 2-agent file is kept untouched for Phase A reproducibility.

Key differences from 2-agent wrapper:
- `num_agents` kwarg (default 2, tested for N=3 and N=4)
- `grid_size` kwarg (default 1.0; set to 1.5 for N=4 to reduce crowding)
- All (2, 2) / (2,) shaped arrays generalized to (N, 2) / (N,)
- Collision detection: all C(N,2) pairs via vectorized all-pairs distance
- Goal sampling: sequential rejection for all N goals
- Observation: ego-centric lists others as (i+1)%N, (i+2)%N, ...
- SAP domain randomization: all non-focal agents are randomized
- done_condition 'agent_i': parsed from string, works for any i
"""

from functools import partial
from typing import Dict, Any, Tuple, Optional, List
import os

import chex
import jax
import jax.numpy as jnp
from flax.struct import dataclass
from jaxmarl.environments import spaces as jaxmarl_spaces

from envs.base_env import BaseEnv, WrappedEnvState


@dataclass
class CoopReconNAgentEnvState:
    """Internal state for the N-agent cooperative reconnaissance environment."""
    positions: jnp.ndarray        # (N, 2) - agent positions [x, y]
    velocities: jnp.ndarray       # (N, 2) - agent velocities [vx, vy]
    goal_pos: jnp.ndarray         # (N, 2) - goal positions per agent (agent i → goal i)
    detected_water: jnp.ndarray   # (N,) bool
    detected_life: jnp.ndarray    # (N,) bool
    picture_taken: jnp.ndarray    # (N,) bool
    timestep: jnp.ndarray         # scalar int
    collision_happened: jnp.ndarray  # scalar bool — True once any two agents collide

@dataclass
class CoopReconWrappedEnvState:
    """WrappedEnvState extended with per-agent collision and social-law counters."""
    env_state: Any
    base_return_so_far: jnp.ndarray
    avail_actions: jnp.ndarray
    step: jnp.array
    collisions_so_far: jnp.ndarray       # (N,) int32 — cumulative per-agent collision count
    law_activations_so_far: jnp.ndarray  # (N,) int32 — steps agent was within social_min_dist of another


class CoopReconContinuousNAgentWrapper(BaseEnv):
    """N-agent JAX wrapper for cooperative reconnaissance continuous environment.

    Args:
        **kwargs:
            num_agents (int): Number of agents. Default 2. Use 3 or 4 for Phase B.
            grid_size (float): Side length of the square grid. Default 1.0.
                               Use 1.5 for N=4 to reduce crowding.
            share_rewards (bool): Share rewards between agents. Default False.
            dt (float): Physics time step. Default 0.05.
            max_speed (float): Maximum agent speed. Default 1.0.
            detection_radius (float): Radius for task actions. Default 0.2.
            horizon (int): Episode length. Default 80 for N=3, 100 for N=4.
            done_condition (str): 'all', 'any', or 'agent_i'. Default 'all'.
            movement_noise_std (float): Gaussian noise fraction. Default 0.0.
            collision_radius (float): Dead-end collision distance. Default 0.05.
            step_penalty (float): Per-step reward penalty. Default -0.01.
            min_sep_goal (float): Minimum goal separation. Default 0.30.
            ego_centric_obs (bool): Use relative observations. Default False.
            sap_domain_randomize_partner (bool): Randomize partner states. Default False.
            sap_focal_agent_idx (int): Focal agent for SAP randomization. Default 0.
            social_min_dist (float): Minimum distance social law threshold. Default 0.0
                (disabled). When > 0, any locomotion action (1-4) that would bring an
                agent within this distance of another is masked unavailable. Gaussian
                movement noise applied in step() can still cause violations — this is
                intentional so the law governs deliberate actions only.
    """

    def __init__(self, **kwargs):
        self.num_agents = kwargs.get('num_agents', 2)
        self.grid_size = kwargs.get('grid_size', 1.0)

        self.share_rewards = kwargs.get('share_rewards', False)
        self.dt = kwargs.get('dt', 0.05)
        self.max_speed = kwargs.get('max_speed', 1.0)
        self.detection_radius = kwargs.get('detection_radius', 0.2)
        self.horizon = kwargs.get('horizon', 80)
        self.done_condition = kwargs.get('done_condition', 'all')
        self.movement_noise_std = kwargs.get('movement_noise_std', 0.0)

        self.sap_domain_randomize_partner = kwargs.get('sap_domain_randomize_partner', False)
        self.sap_focal_agent_idx = kwargs.get('sap_focal_agent_idx', 0)

        self.collision_radius = kwargs.get('collision_radius', 0.05)
        self.step_penalty = kwargs.get('step_penalty', -0.01)
        self.min_sep_goal = kwargs.get('min_sep_goal', 0.30)
        # Social minimum distance law. Default 0.0 = disabled (backward compatible).
        # Must be > collision_radius to provide any protection before termination.
        self.social_min_dist = kwargs.get('social_min_dist', 0.0)

        self._render = kwargs.get('render', False)
        self._render_name = kwargs.get('render_name', "coop_recon_n_agent")
        self._render_dir = kwargs.get('render_dir', "render")

        self._ego_centric_obs = kwargs.get('ego_centric_obs', False)

        N = self.num_agents
        self.agents = [f"agent_{i}" for i in range(N)]
        self.name = f"CoopReconContinuous{N}Agent"

        # Obs dims:
        # Ego: 2*(N-1) rel_pos + 2*N vel + 2*N goal_vec + 3*N task = 7N + 2(N-1) = 9N - 2
        # Non-ego: 2*N pos + 2*N vel + 2*N goal + 3*N task = 9N
        if self._ego_centric_obs:
            obs_size = 9 * N - 2
        else:
            obs_size = 9 * N
        # Full obs always uses global absolute info: same as non-ego
        obs_full_size = 9 * N

        self.observation_spaces = {}
        self.observation_full_spaces = {}
        for agent in self.agents:
            self.observation_spaces[agent] = jaxmarl_spaces.Box(
                low=-self.grid_size * 2, high=self.grid_size * 2,
                shape=(obs_size,),
                dtype=jnp.float32
            )
            self.observation_full_spaces[agent] = jaxmarl_spaces.Box(
                low=-self.grid_size * 2, high=self.grid_size * 2,
                shape=(obs_full_size,),
                dtype=jnp.float32
            )

        # Actions: 8 discrete
        # 0=noop, 1=north, 2=south, 3=east, 4=west, 5=detect_water, 6=detect_life, 7=picture
        self.action_spaces = {
            agent: jaxmarl_spaces.Discrete(num_categories=8, dtype=jnp.int32)
            for agent in self.agents
        }

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

    # -------------------------------------------------------------------------
    # HELPERS
    # -------------------------------------------------------------------------

    def _spawn_margin(self):
        """Small margin from grid boundary for spawning."""
        return 0.01 * self.grid_size

    def _sample_position(self, key):
        """Sample a single 2D position within the grid."""
        margin = self._spawn_margin()
        return jax.random.uniform(key, shape=(2,), minval=margin, maxval=self.grid_size - margin)

    # -------------------------------------------------------------------------
    # RESET
    # -------------------------------------------------------------------------

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, jnp.ndarray], WrappedEnvState]:
        N = self.num_agents
        key, pos_key, goal_key, partner_state_key = jax.random.split(key, 4)

        # ------------------------------------------------------------------
        # Sample N agent positions independently (full grid)
        # ------------------------------------------------------------------
        pos_keys = jax.random.split(pos_key, N)
        positions = jnp.stack([self._sample_position(pos_keys[i]) for i in range(N)])  # (N, 2)
        velocities = jnp.zeros((N, 2), dtype=jnp.float32)

        # ------------------------------------------------------------------
        # Sample N goals with sequential pairwise rejection sampling.
        # Each new goal is resampled until it is >= min_sep_goal from all previous.
        # ------------------------------------------------------------------
        goal_keys = jax.random.split(goal_key, N * 10)  # generous key budget
        goals = []

        # Goal 0: always free
        goals.append(self._sample_position(goal_keys[0]))

        for gi in range(1, N):
            # Build a fixed array of previously sampled goals for use inside while_loop.
            # We stack them and freeze them as a Python-level constant per iteration
            # (safe because gi is a Python int, not a JAX traced value).
            prev_goals_fixed = jnp.stack(goals)  # (gi, 2)

            def _too_close(carry, prev=prev_goals_fixed):
                k, g = carry
                dists = jnp.linalg.norm(g[None, :] - prev, axis=-1)  # (gi,)
                return jnp.any(dists < self.min_sep_goal)

            def _resample(carry):
                k, _ = carry
                k, subk = jax.random.split(k)
                g = self._sample_position(subk)
                return k, g

            init_g = self._sample_position(goal_keys[gi])
            init_carry = (goal_keys[gi + N], init_g)
            _, goal_i = jax.lax.while_loop(_too_close, _resample, init_carry)
            goals.append(goal_i)

        goal_pos = jnp.stack(goals)  # (N, 2)

        # ------------------------------------------------------------------
        # Task state arrays
        # ------------------------------------------------------------------
        detected_water = jnp.zeros(N, dtype=bool)
        detected_life = jnp.zeros(N, dtype=bool)
        picture_taken = jnp.zeros(N, dtype=bool)
        collision_happened = jnp.array(False)

        # SAP Domain Randomization: randomize ALL non-focal agents' task states
        if self.sap_domain_randomize_partner:
            focal_idx = self.sap_focal_agent_idx
            partner_indices = [i for i in range(N) if i != focal_idx]
            partner_keys = jax.random.split(partner_state_key, len(partner_indices) * 3)
            ki = 0
            for pidx in partner_indices:
                pw = jax.random.bernoulli(partner_keys[ki], p=0.5); ki += 1
                pl = pw & jax.random.bernoulli(partner_keys[ki], p=0.5); ki += 1
                pp = pl & jax.random.bernoulli(partner_keys[ki], p=0.5); ki += 1
                detected_water = detected_water.at[pidx].set(pw)
                detected_life = detected_life.at[pidx].set(pl)
                picture_taken = picture_taken.at[pidx].set(pp)

        timestep = jnp.array(0, dtype=jnp.int32)

        env_state = CoopReconNAgentEnvState(
            positions=positions,
            velocities=velocities,
            goal_pos=goal_pos,
            detected_water=detected_water,
            detected_life=detected_life,
            picture_taken=picture_taken,
            timestep=timestep,
            collision_happened=collision_happened,
        )

        obs = self._get_obs(env_state)
        avail_actions = self._get_avail_actions(env_state)

        state = CoopReconWrappedEnvState(
            env_state=env_state,
            base_return_so_far=jnp.zeros(N, dtype=jnp.float32),
            avail_actions=avail_actions,
            step=timestep,
            collisions_so_far=jnp.zeros(N, dtype=jnp.int32),
            law_activations_so_far=jnp.zeros(N, dtype=jnp.int32),
        )

        return obs, state

    # -------------------------------------------------------------------------
    # STEP
    # -------------------------------------------------------------------------

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: WrappedEnvState,
        actions: Dict[str, chex.Array],
        reset_state: Optional[WrappedEnvState] = None,
    ) -> Tuple[Dict[str, jnp.ndarray], WrappedEnvState, Dict[str, float], Dict[str, bool], Dict]:
        key, key_reset = jax.random.split(key)
        env_state = state.env_state

        actions_array = jnp.stack([actions[agent] for agent in self.agents], axis=0).astype(jnp.int32).squeeze()

        new_velocities = self._update_velocities(env_state.velocities, actions_array, env_state.picture_taken)
        new_positions = env_state.positions + new_velocities * self.dt

        # Gaussian movement noise
        key, key_noise = jax.random.split(key)
        noise = jax.random.normal(key_noise, shape=new_positions.shape) * self.movement_noise_std * self.dt
        new_positions = new_positions + noise

        # Boundary clip (full grid)
        new_positions = jnp.clip(new_positions, 0.0, self.grid_size)

        # Collision detection: all C(N,2) pairs
        diffs = new_positions[:, None, :] - new_positions[None, :, :]  # (N, N, 2)
        dists = jnp.linalg.norm(diffs, axis=-1)                        # (N, N)
        off_diag = ~jnp.eye(self.num_agents, dtype=bool)
        
        # Existing global collision flag for termination logging
        collision = jnp.any((dists < self.collision_radius) & off_diag)
        new_collision_happened = env_state.collision_happened | collision

        # Per-agent collision accumulation exactly mimicking teammate's code
        per_agent_collision = jnp.any((dists < self.collision_radius) & off_diag, axis=1) # (N,)
        new_episode_collisions = state.collisions_so_far + per_agent_collision.astype(jnp.int32)

        # Social law proximity tracking: count steps where agent i ended up within
        # social_min_dist of another agent (AFTER noise is applied). This captures
        # both noise-driven violations and any collision-zone entries.
        # When social_min_dist == 0.0 this is always False (no overhead).
        per_agent_in_law_zone = jnp.any(
            (dists < self.social_min_dist) & off_diag, axis=1
        )  # (N,) bool
        new_law_activations = state.law_activations_so_far + per_agent_in_law_zone.astype(jnp.int32)

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

        env_state_next = CoopReconNAgentEnvState(
            positions=new_positions,
            velocities=new_velocities,
            goal_pos=env_state.goal_pos,
            detected_water=new_detected_water,
            detected_life=new_detected_life,
            picture_taken=new_picture_taken,
            timestep=new_timestep,
            collision_happened=new_collision_happened,
        )

        obs_st = self._get_obs(env_state_next)
        avail_actions = self._get_avail_actions(env_state_next)
        dones = self._get_dones(env_state_next)

        # Build next state before reset triggers, appropriately masking collisions on done
        state_st = CoopReconWrappedEnvState(
            env_state=env_state_next,
            base_return_so_far=jnp.zeros(self.num_agents, dtype=jnp.float32),
            avail_actions=avail_actions,
            step=new_timestep,
            collisions_so_far=(
                state.collisions_so_far * (1 - dones["__all__"])
                + new_episode_collisions * dones["__all__"]
            ),
            law_activations_so_far=(
                state.law_activations_so_far * (1 - dones["__all__"])
                + new_law_activations * dones["__all__"]
            ),
        )

        info = {
            'pre_reset_state': state_st,
            'pre_reset_obs': obs_st,
            'returned_episode_collisions': state_st.collisions_so_far,
            'returned_episode_law_activations': state_st.law_activations_so_far,
        }

        obs, state = jax.tree.map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y),
            self.reset(key_reset),
            (obs_st, state_st)
        )

        return obs, state, rewards, dones, info

    # -------------------------------------------------------------------------
    # VELOCITY UPDATE
    # -------------------------------------------------------------------------

    @partial(jax.jit, static_argnums=(0,))
    def _update_velocities(self, velocities: jnp.ndarray, actions: jnp.ndarray, picture_taken: jnp.ndarray) -> jnp.ndarray:
        target_vels = self.action_to_vel[actions]  # (N, 2)
        new_velocities = 0.7 * velocities + 0.3 * target_vels
        speeds = jnp.linalg.norm(new_velocities, axis=1, keepdims=True)
        scale = jnp.minimum(1.0, self.max_speed / (speeds + 1e-8))
        
        # Zero out velocity for agents who have completed their task
        completed_mask = picture_taken[:, None]  # (N, 1)
        new_velocities = jnp.where(completed_mask, jnp.zeros_like(new_velocities), new_velocities * scale)
        
        return new_velocities

    # -------------------------------------------------------------------------
    # PROCESS ACTIONS & REWARDS
    # -------------------------------------------------------------------------

    @partial(jax.jit, static_argnums=(0,))
    def _process_actions(
        self,
        key_water: chex.PRNGKey,
        key_life: chex.PRNGKey,
        positions: jnp.ndarray,     # (N, 2)
        goal_pos: jnp.ndarray,      # (N, 2)
        actions: jnp.ndarray,       # (N,)
        detected_water: jnp.ndarray,
        detected_life: jnp.ndarray,
        picture_taken: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, float]]:
        N = self.num_agents

        # Distances from all agents to all goals: (N agents, N goals)
        dists_to_goal = jnp.linalg.norm(positions[:, None, :] - goal_pos[None, :, :], axis=-1)
        at_goal = dists_to_goal < self.detection_radius  # (N, N)

        # Goal-allocation social law: agent i → goal i only
        agent_goal_mask = jnp.eye(N, dtype=bool)  # (N, N)
        at_goal = at_goal & agent_goal_mask

        # Rewards: start with step penalty
        reward_values = jnp.full(N, self.step_penalty, dtype=jnp.float32)

        # Split keys for N agents
        water_keys = jax.random.split(key_water, N)
        life_keys = jax.random.split(key_life, N)

        # Action masks (N, 1) for broadcasting
        w_action = (actions == 5)[:, None]
        l_action = (actions == 6)[:, None]
        p_action = (actions == 7)[:, None]

        is_detect_water = w_action & at_goal & ~detected_water[None, :]
        is_detect_life  = l_action & at_goal & detected_water[None, :] & ~detected_life[None, :]
        is_picture      = p_action & at_goal & detected_life[None, :]  & ~picture_taken[None, :]

        # Stochastic success (N agents, vectorized)
        water_success = jax.vmap(lambda k: jax.random.bernoulli(k, 0.8))(water_keys)[:, None]
        life_success  = jax.vmap(lambda k: jax.random.bernoulli(k, 0.8))(life_keys)[:, None]

        water_detected    = is_detect_water & water_success  # (N, N)
        life_detected     = is_detect_life  & life_success   # (N, N)
        picture_succeeded = is_picture                        # (N, N)

        # Update global task state (OR across agents acting on each goal)
        new_detected_water = detected_water | jnp.any(water_detected, axis=0)
        new_detected_life  = detected_life  | jnp.any(life_detected,  axis=0)
        new_picture_taken  = picture_taken  | jnp.any(picture_succeeded, axis=0)

        # Rewards: agent i gets reward for any detection it triggers
        reward_values = reward_values + jnp.sum(jnp.where(water_detected,    1.0, 0.0), axis=1)
        reward_values = reward_values + jnp.sum(jnp.where(life_detected,     1.0, 0.0), axis=1)
        reward_values = reward_values + jnp.sum(jnp.where(picture_succeeded, 10.0, 0.0), axis=1)

        rewards = {agent: reward_values[i] for i, agent in enumerate(self.agents)}

        if self.share_rewards:
            total = jnp.sum(reward_values)
            rewards = {agent: total for agent in self.agents}

        return new_detected_water, new_detected_life, new_picture_taken, rewards

    # -------------------------------------------------------------------------
    # OBSERVATIONS
    # -------------------------------------------------------------------------

    @partial(jax.jit, static_argnums=(0,))
    def _get_obs(self, env_state: CoopReconNAgentEnvState) -> Dict[str, jnp.ndarray]:
        N = self.num_agents
        obs_dict = {}
        obs_full_dict = {}

        for agent_idx, agent in enumerate(self.agents):
            # The "others" for agent i, in order: (i+1)%N, (i+2)%N, ..., (i+N-1)%N
            others = [(agent_idx + k) % N for k in range(1, N)]

            if self._ego_centric_obs:
                # Relative positions to each other agent
                rel_positions = jnp.concatenate([
                    env_state.positions[other_idx] - env_state.positions[agent_idx]
                    for other_idx in others
                ])  # 2*(N-1) dims

                # Velocities: own first, then others
                velocities = jnp.concatenate([
                    env_state.velocities[agent_idx],
                    *[env_state.velocities[other_idx] for other_idx in others]
                ])  # 2*N dims

                # Goal vectors: relative from self to each goal (own first, then others)
                goal_vectors = jnp.concatenate([
                    env_state.goal_pos[agent_idx] - env_state.positions[agent_idx],
                    *[env_state.goal_pos[other_idx] - env_state.positions[agent_idx]
                      for other_idx in others]
                ])  # 2*N dims

                # Task states: own first, then others
                water = jnp.array(
                    [env_state.detected_water[agent_idx]] + [env_state.detected_water[o] for o in others],
                    dtype=jnp.float32)
                life = jnp.array(
                    [env_state.detected_life[agent_idx]] + [env_state.detected_life[o] for o in others],
                    dtype=jnp.float32)
                pic = jnp.array(
                    [env_state.picture_taken[agent_idx]] + [env_state.picture_taken[o] for o in others],
                    dtype=jnp.float32)

                obs_dict[agent] = jnp.concatenate([
                    rel_positions, velocities, goal_vectors, water, life, pic
                ], dtype=jnp.float32)

            else:
                # Absolute positions: own first, then others
                positions = jnp.concatenate([
                    env_state.positions[agent_idx],
                    *[env_state.positions[other_idx] for other_idx in others]
                ])  # 2*N dims

                velocities = jnp.concatenate([
                    env_state.velocities[agent_idx],
                    *[env_state.velocities[other_idx] for other_idx in others]
                ])  # 2*N dims

                goals = jnp.concatenate([
                    env_state.goal_pos[agent_idx],
                    *[env_state.goal_pos[other_idx] for other_idx in others]
                ])  # 2*N dims

                water = jnp.array(
                    [env_state.detected_water[agent_idx]] + [env_state.detected_water[o] for o in others],
                    dtype=jnp.float32)
                life = jnp.array(
                    [env_state.detected_life[agent_idx]] + [env_state.detected_life[o] for o in others],
                    dtype=jnp.float32)
                pic = jnp.array(
                    [env_state.picture_taken[agent_idx]] + [env_state.picture_taken[o] for o in others],
                    dtype=jnp.float32)

                obs_dict[agent] = jnp.concatenate([
                    positions, velocities, goals, water, life, pic
                ], dtype=jnp.float32)

            # Full obs: always absolute global info (used by joint adversarial training)
            obs_full_dict[agent] = jnp.concatenate([
                env_state.positions.flatten(),       # 2*N
                env_state.velocities.flatten(),      # 2*N
                env_state.goal_pos.flatten(),        # 2*N
                env_state.detected_water.astype(jnp.float32),  # N
                env_state.detected_life.astype(jnp.float32),   # N
                env_state.picture_taken.astype(jnp.float32),   # N
            ], dtype=jnp.float32)

        return obs_dict, obs_full_dict

    # -------------------------------------------------------------------------
    # DONES
    # -------------------------------------------------------------------------

    @partial(jax.jit, static_argnums=(0,))
    def _get_dones(self, env_state: CoopReconNAgentEnvState) -> Dict[str, bool]:
        collision_done = env_state.collision_happened
        horizon_done = env_state.timestep >= self.horizon

        if self.done_condition == 'any':
            done = jnp.any(env_state.picture_taken) | horizon_done | collision_done
        elif self.done_condition == 'all':
            done = jnp.all(env_state.picture_taken) | horizon_done | collision_done
        elif self.done_condition.startswith('agent_'):
            idx = int(self.done_condition.split('_')[1])
            done = env_state.picture_taken[idx] | horizon_done | collision_done
        else:
            raise ValueError(f"Unknown done_condition: {self.done_condition}")

        dones = {agent: done for agent in self.agents}
        dones["__all__"] = done
        return dones

    # -------------------------------------------------------------------------
    # AVAILABLE ACTIONS
    # -------------------------------------------------------------------------

    @partial(jax.jit, static_argnums=(0,))
    def _get_avail_actions(self, env_state: CoopReconNAgentEnvState) -> Dict[str, jnp.ndarray]:
        N = self.num_agents

        dists_to_goal = jnp.linalg.norm(
            env_state.positions[:, None, :] - env_state.goal_pos[None, :, :], axis=-1)  # (N, N)
        at_goal = dists_to_goal < self.detection_radius  # (N, N)

        # Goal-allocation social law: agent i may only act at goal i.
        agent_goal_mask = jnp.eye(N, dtype=bool)
        at_goal = at_goal & agent_goal_mask  # (N, N)

        # -------------------------------------------------------------------
        # Social Minimum Distance Law
        # -------------------------------------------------------------------
        # For every locomotion action (1=N, 2=S, 3=E, 4=W) for every agent i,
        # predict the next position using the same velocity physics as step():
        #
        #   new_vel  = 0.7 * vel_i + 0.3 * target_vel   (momentum smoothing)
        #   new_vel  = new_vel * min(1, max_speed / ||new_vel||)  (speed clip)
        #   pred_pos = pos_i + new_vel * dt              (Euler integration)
        #
        # If pred_pos would bring agent i within social_min_dist of any other
        # agent j (using their CURRENT, pre-noise positions), that action is
        # masked unavailable.
        #
        # Key properties:
        #  - NOOP (0) and task actions (5/6/7) are NEVER restricted — agent
        #    can always stand still or perform its reconnaissance task.
        #  - Gaussian noise in step() is applied AFTER this check, so the law
        #    governs deliberate actions only; noise-driven violations are still
        #    possible (intentional, per PI specification).
        #  - Because `self` is static_argnums=(0,), the `if` branch below is
        #    evaluated at JIT compile time. When social_min_dist == 0.0 the
        #    entire block is compiled away — zero runtime cost for old configs.
        # -------------------------------------------------------------------
        if self.social_min_dist > 0.0:
            # Predicted velocity for each agent × each of the 8 actions.
            # Shapes: velocities (N, 2), action_to_vel (8, 2)
            # → new_vels (N, 8, 2) via broadcasting
            new_vels = (
                0.7 * env_state.velocities[:, None, :]   # (N, 1, 2)
                + 0.3 * self.action_to_vel[None, :, :]   # (1, 8, 2)
            )  # (N, 8, 2)

            # Speed-clip each (agent, action) velocity to max_speed.
            speeds = jnp.linalg.norm(new_vels, axis=-1, keepdims=True)  # (N, 8, 1)
            scale  = jnp.minimum(1.0, self.max_speed / (speeds + 1e-8))  # (N, 8, 1)
            new_vels = new_vels * scale  # (N, 8, 2)

            # Completed agents stop moving regardless of the chosen action.
            done_mask = env_state.picture_taken[:, None, None]  # (N, 1, 1) bool
            new_vels = jnp.where(done_mask, jnp.zeros_like(new_vels), new_vels)

            # Predicted positions (no noise — deliberate-action check only).
            pred_pos = (
                env_state.positions[:, None, :]   # (N, 1, 2)
                + new_vels * self.dt               # (N, 8, 2)
            )  # (N, 8, 2)
            pred_pos = jnp.clip(pred_pos, 0.0, self.grid_size)

            # Distance from each (agent_i, action_k) predicted position to every
            # agent_j's CURRENT position.
            # pred_pos[:, :, None, :] : (N, 8, 1, 2)
            # positions[None, None, :, :] : (1, 1, N, 2)
            # → pred_dists : (N, 8, N)
            pred_diffs = (
                pred_pos[:, :, None, :]                      # (N, 8, 1, 2)
                - env_state.positions[None, None, :, :]      # (1, 1, N, 2)
            )  # (N, 8, N, 2)
            pred_dists = jnp.linalg.norm(pred_diffs, axis=-1)  # (N, 8, N)

            # Exclude self-distance: off_diag[i, j] = (i != j).
            # Shape (N, N) -> (N, 1, N) broadcasts to (N, 8, N).
            off_diag_3d = (~jnp.eye(N, dtype=bool))[:, None, :]  # (N, 1, N)

            # For each (agent_i, action_k): would the move violate min_dist?
            would_violate = jnp.any(
                (pred_dists < self.social_min_dist) & off_diag_3d,
                axis=-1,
            )  # (N, 8)

            # Only restrict locomotion actions 1-4 (N/S/E/W).
            # Actions 0 (noop), 5 (water), 6 (life), 7 (picture) are never
            # restricted by the minimum-distance law.
            is_locomotion = jnp.array(
                [False, True, True, True, True, False, False, False], dtype=bool
            )  # (8,)

            # social_law_ok[i, k] = True means action k is ALLOWED for agent i.
            social_law_ok = ~(would_violate & is_locomotion[None, :])  # (N, 8)

        avail_actions = {}
        for i, agent in enumerate(self.agents):
            agent_done = env_state.picture_taken[i]
            at_own_goal = at_goal[i, i]

            can_water = at_own_goal & ~env_state.detected_water[i]
            can_life  = at_own_goal & env_state.detected_water[i] & ~env_state.detected_life[i]
            can_pic   = at_own_goal & env_state.detected_life[i]  & ~env_state.picture_taken[i]

            full_mask = jnp.array([
                True, True, True, True, True,
                can_water, can_life, can_pic
            ], dtype=jnp.float32)
            noop_mask = jnp.array(
                [True, False, False, False, False, False, False, False], dtype=jnp.float32
            )

            mask = jnp.where(agent_done, noop_mask, full_mask)

            # Apply social minimum distance masking (AND with social_law_ok[i]).
            # This is a compile-time branch — elided entirely when social_min_dist==0.
            if self.social_min_dist > 0.0:
                mask = mask * social_law_ok[i].astype(jnp.float32)

            avail_actions[agent] = mask

        return avail_actions

    # -------------------------------------------------------------------------
    # SPACES
    # -------------------------------------------------------------------------

    def observation_space(self, agent: str, observation_type: str = "agent"):
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
        return state.avail_actions

    @partial(jax.jit, static_argnums=(0,))
    def get_step_count(self, state: WrappedEnvState) -> jnp.ndarray:
        return state.step

    def render(self, env_state, save_frame: bool = True) -> Any:
        pass

    def reset_render(self):
        pass

    def animate(self, states, dones, num_episodes, extra_dir=None, fps=1, loop_count=0, debug=False):
        if debug:
            import logging
            from envs.coop_recon_continuous.CoopReconContinuousNAgentViz import CoopReconContinuousNAgentViz
            log = logging.getLogger(__name__)
            try:
                init_state, env_states = states
                
                base_dir = self._render_dir
                if extra_dir is not None:
                    base_dir = os.path.join(base_dir, extra_dir)
                os.makedirs(base_dir, exist_ok=True)
                
                N = self.num_agents

                for ep in range(num_episodes):
                    positions = env_states.positions[0, ep] # (max_steps, N, 2)
                    goals = env_states.goal_pos[0, ep] # (max_steps, N, 2)
                    water = env_states.detected_water[0, ep] # (max_steps, N)
                    life = env_states.detected_life[0, ep] # (max_steps, N)
                    picture = env_states.picture_taken[0, ep] # (max_steps, N)
                    
                    viz = CoopReconContinuousNAgentViz(grid_size=self.grid_size)
                    frames = []
                    max_steps = positions.shape[0]
                    
                    for t in range(max_steps):
                        state_layout = {
                            'agent_positions': [(float(positions[t, i, 0]), float(positions[t, i, 1])) for i in range(N)],
                            'goal_positions': [(float(goals[t, i, 0]), float(goals[t, i, 1])) for i in range(N)],
                            'detected_water': [bool(water[t, i]) for i in range(N)],
                            'detected_life': [bool(life[t, i]) for i in range(N)],
                            'picture_taken': [bool(picture[t, i]) for i in range(N)]
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
