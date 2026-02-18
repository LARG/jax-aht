from functools import partial
from itertools import product
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

from envs.rddl.grid_10x10_alternating.Grid10x10MultiAgentViz import Grid10x10MultiAgentVisualizer

class Grid10x10AlternatingWrapper(BaseEnv):
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
        self._render_name = kwargs.get('render_name', "grid_10x10")
        self._render_dir = kwargs.get('render_dir', "grid_10x10")

        self._ego_centric_obs = kwargs.get('ego_centric_obs', False)

        # Stochastic movement
        self.stochastic_movement_prob = kwargs.get('stochastic_movement_prob', 0.0)

        self.horizon = self.env.horizon
        self.name = self.env.__class__.__name__
        self.toroidal = self.env.model.non_fluents['TOROIDAL']
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
        self.limit_goals_to_inner_grid = kwargs.get("limit_goals_to_inner_grid", True)

        self._xpos_list = list(self.env.model.type_to_objects['xpos'])
        self._ypos_list = list(self.env.model.type_to_objects['ypos'])
        self._pos_list = list(product(self._xpos_list, self._ypos_list))
        self._restriction_type = self.env.model.non_fluents['RESTRICTION-MODE']


        if self._render:
            self.render_name = kwargs.get('render_name', "grid_10x10_alternating")
            self.render_dir = os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, kwargs.get('render_dir', "render"))
            if not os.path.exists(self.render_dir):
                os.makedirs(self.render_dir)
            # movie_gen = MovieGenerator(save_dir=self.render_dir, env_name=self.render_name, max_frames=1000, frame_duration=1000)
            viz=Grid10x10MultiAgentVisualizer(self.env.model)
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

    def _actions_to_rddl(self, actions: Dict[str, Any]):
        '''Convert dict of actions to array'''
        if self.vectorized:
            action_array = jnp.zeros((self.num_agents,), dtype=jnp.int32)
            for agent_idx, agent in enumerate(self.agents):
                # Ensure action is scalar by squeezing if needed
                # action_value = jnp.asarray(actions[agent]).squeeze()
                action_array = action_array.at[agent_idx].set(actions[agent].squeeze())
            vectorized_actions = {self.rddl_action_keys[0]: action_array}
            return vectorized_actions
        else:
            raise NotImplementedError("Non-vectorized actions not implemented yet.")

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
            dones["__all__"] = dones[self.agents[self.controlled_agents[0]]] | terminal | done

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

    def _get_valid_moves_mask(self, x_idx, y_idx):
        """
        Get valid move mask based on restriction mode.
        Returns a 4-element boolean array for [EAST, WEST, NORTH, SOUTH].

        Restrictions (based on domain preconditions):
        - EVEN columns (x0, x2): Can move SOUTH (forbidden: NORTH)
        - ODD columns (x1, x3): Can move NORTH (forbidden: SOUTH)
        - EVEN rows (y0, y2): Can move EAST (forbidden: WEST)
        - ODD rows (y1, y3): Can move WEST (forbidden: EAST)
        """
        x_even = (x_idx % 2 == 0)
        y_even = (y_idx % 2 == 0)

        # Mode -1: No restrictions (all moves allowed)
        all_moves = jnp.array([True, True, True, True])  # [EAST, WEST, NORTH, SOUTH]

        # Build mask based on restriction type
        # EAST/WEST based on row (y) parity
        east_allowed = jnp.where(
            (self._restriction_type == 0) | (self._restriction_type == 2),
            y_even,  # EVEN rows can go EAST
            True
        )
        west_allowed = jnp.where(
            (self._restriction_type == 0) | (self._restriction_type == 2),
            jnp.logical_not(y_even),  # ODD rows can go WEST
            True
        )

        # NORTH/SOUTH based on column (x) parity
        north_allowed = jnp.where(
            (self._restriction_type == 0) | (self._restriction_type == 1),
            jnp.logical_not(x_even),  # ODD columns can go NORTH
            True
        )
        south_allowed = jnp.where(
            (self._restriction_type == 0) | (self._restriction_type == 1),
            x_even,  # EVEN columns can go SOUTH
            True
        )

        moves_mask = jnp.array([east_allowed, west_allowed, north_allowed, south_allowed])

        # If mode is -1, return all moves
        return jnp.where(self._restriction_type == -1, all_moves, moves_mask)

    def _would_block_path_jax(self, agent_pos, goal_pos, blocking_pos, obstacle_mask):
        """
        Check if blocking_pos blocks the path from agent_pos to goal_pos using adaptive BFS.
        Uses queue-based for small grids (<= 5x5) and vectorized for larger grids.
        """
        agent_x, agent_y = agent_pos
        goal_x, goal_y = goal_pos
        block_x, block_y = blocking_pos

        # Quick checks
        is_blocking_agent = (block_x == agent_x) & (block_y == agent_y)
        is_blocking_goal = (block_x == goal_x) & (block_y == goal_y)

        num_x = len(self._xpos_list)
        num_y = len(self._ypos_list)
        max_cells = num_x * num_y

        # Use queue-based BFS for small grids (faster), vectorized for large grids
        use_queue_based = max_cells <= 25  # 5x5 or smaller

        def queue_based_bfs():
            """Queue-based BFS - efficient for small grids with sparse frontiers"""
            # Initialize visited
            visited = jnp.zeros((num_x, num_y), dtype=jnp.bool_)
            visited = visited.at[agent_x, agent_y].set(True)
            visited = visited.at[block_x, block_y].set(True)
            visited = jnp.where(obstacle_mask, True, visited)

            # Fixed-size queue
            queue = jnp.full((max_cells, 2), -1, dtype=jnp.int32)
            queue = queue.at[0].set(jnp.array([agent_x, agent_y]))
            queue_start = 0
            queue_end = 1

            move_deltas = jnp.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=jnp.int32)

            def process_one_cell(carry):
                visited, queue, queue_start, queue_end, found = carry

                curr_pos = queue[queue_start]
                curr_x, curr_y = curr_pos[0], curr_pos[1]
                queue_start = queue_start + 1

                def try_direction(dir_carry, direction):
                    visited, queue, queue_end, found = dir_carry

                    new_x = curr_x + move_deltas[direction, 0]
                    new_y = curr_y + move_deltas[direction, 1]

                    in_bounds = (new_x >= 0) & (new_x < num_x) & (new_y >= 0) & (new_y < num_y)

                    x_even = (curr_x % 2 == 0)
                    y_even = (curr_y % 2 == 0)

                    move_allowed = jnp.where(
                        self._restriction_type == -1,
                        True,
                        jnp.where(
                            direction == 0,  # EAST
                            jnp.where((self._restriction_type == 0) | (self._restriction_type == 2), y_even, True),
                            jnp.where(
                                direction == 1,  # WEST
                                jnp.where((self._restriction_type == 0) | (self._restriction_type == 2), jnp.logical_not(y_even), True),
                                jnp.where(
                                    direction == 2,  # NORTH
                                    jnp.where((self._restriction_type == 0) | (self._restriction_type == 1), jnp.logical_not(x_even), True),
                                    jnp.where((self._restriction_type == 0) | (self._restriction_type == 1), x_even, True)
                                )
                            )
                        )
                    )

                    safe_x = jnp.clip(new_x, 0, num_x - 1)
                    safe_y = jnp.clip(new_y, 0, num_y - 1)
                    already_visited = visited[safe_x, safe_y]

                    can_move = in_bounds & move_allowed & jnp.logical_not(already_visited)
                    reached_goal = can_move & (new_x == goal_x) & (new_y == goal_y)
                    found = found | reached_goal

                    visited = jnp.where(can_move, visited.at[new_x, new_y].set(True), visited)
                    queue = jnp.where(can_move, queue.at[queue_end].set(jnp.array([new_x, new_y])), queue)
                    queue_end = jnp.where(can_move, queue_end + 1, queue_end)

                    return (visited, queue, queue_end, found), None

                (visited, queue, queue_end, found), _ = jax.lax.scan(
                    try_direction,
                    (visited, queue, queue_end, found),
                    jnp.arange(4)
                )

                return visited, queue, queue_start, queue_end, found

            def bfs_cond(carry):
                visited, queue, queue_start, queue_end, found = carry
                has_more = queue_start < queue_end
                return has_more & jnp.logical_not(found) & (queue_start < max_cells)

            visited_final, _, _, _, found = jax.lax.while_loop(
                bfs_cond,
                process_one_cell,
                (visited, queue, queue_start, queue_end, False)
            )

            return found

        def vectorized_bfs():
            """Vectorized BFS - efficient for larger grids with parallel processing"""
            visited = jnp.zeros((num_x, num_y), dtype=jnp.bool_)
            visited = visited.at[agent_x, agent_y].set(True)
            visited = visited.at[block_x, block_y].set(True)
            visited = jnp.where(obstacle_mask, True, visited)

            frontier = jnp.zeros((num_x, num_y), dtype=jnp.bool_)
            frontier = frontier.at[agent_x, agent_y].set(True)

            move_deltas = jnp.array([[1, 0], [-1, 0], [0, 1], [0, -1]])

            # Precompute move masks for all positions
            x_coords = jnp.arange(num_x)
            y_coords = jnp.arange(num_y)
            x_grid, y_grid = jnp.meshgrid(x_coords, y_coords, indexing='ij')

            x_even = (x_grid % 2 == 0)
            y_even = (y_grid % 2 == 0)

            east_allowed = jnp.where(
                (self._restriction_type == 0) | (self._restriction_type == 2),
                y_even, jnp.ones_like(x_even)
            )
            west_allowed = jnp.where(
                (self._restriction_type == 0) | (self._restriction_type == 2),
                jnp.logical_not(y_even), jnp.ones_like(x_even)
            )
            north_allowed = jnp.where(
                (self._restriction_type == 0) | (self._restriction_type == 1),
                jnp.logical_not(x_even), jnp.ones_like(x_even)
            )
            south_allowed = jnp.where(
                (self._restriction_type == 0) | (self._restriction_type == 1),
                x_even, jnp.ones_like(x_even)
            )

            all_moves_mask = jnp.stack([east_allowed, west_allowed, north_allowed, south_allowed], axis=-1)
            all_moves_mask = jnp.where(
                self._restriction_type == -1,
                jnp.ones_like(all_moves_mask),
                all_moves_mask
            )

            def bfs_step(carry):
                visited, frontier, found, iteration = carry

                frontier_coords = jnp.stack([x_grid, y_grid], axis=-1)
                frontier_mask = frontier[..., None]

                new_positions = frontier_coords[:, :, None, :] + move_deltas[None, None, :, :]
                new_x = new_positions[..., 0]
                new_y = new_positions[..., 1]

                in_bounds = (new_x >= 0) & (new_x < num_x) & (new_y >= 0) & (new_y < num_y)
                is_valid = all_moves_mask & in_bounds & frontier_mask

                safe_new_x = jnp.clip(new_x, 0, num_x - 1).astype(jnp.int32)
                safe_new_y = jnp.clip(new_y, 0, num_y - 1).astype(jnp.int32)

                already_visited = visited[safe_new_x, safe_new_y]
                can_visit = is_valid & jnp.logical_not(already_visited)

                reaches_goal = can_visit & (safe_new_x == goal_x) & (safe_new_y == goal_y)
                found = found | jnp.any(reaches_goal)

                flat_can_visit = can_visit.reshape(-1)
                flat_dest_x = safe_new_x.reshape(-1)
                flat_dest_y = safe_new_y.reshape(-1)
                flat_dest_indices = flat_dest_x * num_y + flat_dest_y

                flat_visited = visited.reshape(-1)
                flat_frontier = jnp.zeros(num_x * num_y, dtype=jnp.bool_)

                new_visited_flat = flat_visited.at[flat_dest_indices].set(
                    jnp.where(flat_can_visit, True, flat_visited[flat_dest_indices])
                )
                new_frontier_flat = flat_frontier.at[flat_dest_indices].set(
                    jnp.where(flat_can_visit, True, flat_frontier[flat_dest_indices])
                )

                new_visited = new_visited_flat.reshape(num_x, num_y)
                new_frontier = new_frontier_flat.reshape(num_x, num_y)

                return new_visited, new_frontier, found, iteration + 1

            def bfs_cond(carry):
                visited, frontier, found, iteration = carry
                has_frontier = jnp.any(frontier)
                return has_frontier & jnp.logical_not(found) & (iteration < max_cells)

            _, _, found, _ = jax.lax.while_loop(
                bfs_cond,
                bfs_step,
                (visited, frontier, False, 0)
            )

            return found

        # Choose BFS strategy based on grid size
        found = jax.lax.cond(
            use_queue_based,
            queue_based_bfs,
            vectorized_bfs
        )

        return is_blocking_agent | is_blocking_goal | jnp.logical_not(found)

    def _randomize_initial_positions(self, key: chex.PRNGKey, env_state, timestep):
        '''Randomizes initial positions of agents and goals using full JAX operations'''

        num_x = len(self._xpos_list)
        num_y = len(self._ypos_list)
        obstacle_mask = jnp.array(self.env.init_values['OBSTACLE'])

        # Initialize arrays
        agent_at = jnp.zeros_like(env_state.obs['agent-at'])
        goal_at = jnp.zeros_like(env_state.obs['goal-at'])
        goal = jnp.zeros_like(env_state.subs['GOAL'])

        # Create position arrays to store (x, y) for each agent
        goal_positions = jnp.zeros((self.num_agents, 2), dtype=jnp.int32)
        agent_positions = jnp.zeros((self.num_agents, 2), dtype=jnp.int32)

        # Valid position masks (ensure JAX arrays)
        valid_pos_mask = jnp.logical_not(obstacle_mask)
        valid_goal_mask = jnp.where(
            (self._restriction_type != -1) & self.limit_goals_to_inner_grid,
            valid_pos_mask.at[0, :].set(False).at[-1, :].set(False).at[:, 0].set(False).at[:, -1].set(False),
            valid_pos_mask
        )

        occupied = jnp.array(obstacle_mask)

        # Sample goal positions using scan
        def sample_goal(carry, agent_idx):
            key, occupied, goal_at_arr, goal_arr, goal_pos_arr = carry
            key, subkey = jax.random.split(key)

            available = valid_goal_mask & jnp.logical_not(occupied)
            flat_available = available.flatten()
            flat_probs = flat_available.astype(jnp.float32) / jnp.sum(flat_available)

            pos_idx = jax.random.choice(subkey, num_x * num_y, p=flat_probs)
            goal_x = pos_idx // num_y
            goal_y = pos_idx % num_y

            goal_at_arr = goal_at_arr.at[agent_idx, goal_x, goal_y].set(True)
            goal_arr = goal_arr.at[agent_idx, goal_x, goal_y].set(True)
            goal_pos_arr = goal_pos_arr.at[agent_idx].set(jnp.array([goal_x, goal_y]))
            occupied = occupied.at[goal_x, goal_y].set(True)

            return (key, occupied, goal_at_arr, goal_arr, goal_pos_arr), None

        (key, occupied, goal_at, goal, goal_positions), _ = jax.lax.scan(
            sample_goal,
            (key, occupied, goal_at, goal, goal_positions),
            jnp.arange(self.num_agents)
        )

        # Sample agent positions using scan
        def sample_agent(carry, agent_idx):
            key, occupied, agent_at_arr, agent_pos_arr = carry
            key, subkey = jax.random.split(key)

            available = valid_pos_mask & jnp.logical_not(occupied)
            flat_available = available.flatten()
            flat_probs = flat_available.astype(jnp.float32) / jnp.sum(flat_available)

            pos_idx = jax.random.choice(subkey, num_x * num_y, p=flat_probs)
            agent_x = pos_idx // num_y
            agent_y = pos_idx % num_y

            agent_at_arr = agent_at_arr.at[agent_idx, agent_x, agent_y].set(True)
            agent_pos_arr = agent_pos_arr.at[agent_idx].set(jnp.array([agent_x, agent_y]))
            occupied = occupied.at[agent_x, agent_y].set(True)

            return (key, occupied, agent_at_arr, agent_pos_arr), None

        (key, occupied, agent_at, agent_positions), _ = jax.lax.scan(
            sample_agent,
            (key, occupied, agent_at, agent_positions),
            jnp.arange(self.num_agents)
        )

        # Handle blocking check and resampling if needed
        def resample_if_needed(carry_input):
            key, agent_at_arr, agent_pos_arr, occupied = carry_input

            # Only resample if conditions are met
            should_check = (~self.toroidal) & (self._restriction_type != -1) & self.uncontrolled_agents_exist

            def do_resampling(carry_input):
                key, agent_at_arr, agent_pos_arr, occupied = carry_input

                # For each controlled agent, check if uncontrolled agents block
                def check_controlled(carry, controlled_idx):
                    key, agent_at_arr, agent_pos_arr, occupied = carry

                    agent_pos = (agent_pos_arr[controlled_idx, 0], agent_pos_arr[controlled_idx, 1])
                    goal_pos = (goal_positions[controlled_idx, 0], goal_positions[controlled_idx, 1])

                    # Check each uncontrolled agent
                    def check_uncontrolled(carry, uncontrolled_idx):
                        key, agent_at_arr, agent_pos_arr, occupied = carry

                        blocking_pos = (agent_pos_arr[uncontrolled_idx, 0], agent_pos_arr[uncontrolled_idx, 1])
                        is_blocking = self._would_block_path_jax(agent_pos, goal_pos, blocking_pos, obstacle_mask)

                        # Resample with while loop if blocking
                        def resample_loop(loop_carry):
                            key, agent_at_arr, agent_pos_arr, occupied, iter_count, _ = loop_carry

                            old_x, old_y = agent_pos_arr[uncontrolled_idx, 0], agent_pos_arr[uncontrolled_idx, 1]
                            occupied = occupied.at[old_x, old_y].set(False)
                            agent_at_arr = agent_at_arr.at[uncontrolled_idx, old_x, old_y].set(False)

                            key, subkey = jax.random.split(key)
                            available = valid_pos_mask & jnp.logical_not(occupied)
                            flat_available = available.flatten()
                            flat_probs = flat_available.astype(jnp.float32) / jnp.sum(flat_available)

                            pos_idx = jax.random.choice(subkey, num_x * num_y, p=flat_probs)
                            new_x = pos_idx // num_y
                            new_y = pos_idx % num_y

                            agent_at_arr = agent_at_arr.at[uncontrolled_idx, new_x, new_y].set(True)
                            agent_pos_arr = agent_pos_arr.at[uncontrolled_idx].set(jnp.array([new_x, new_y]))
                            occupied = occupied.at[new_x, new_y].set(True)

                            new_blocking_pos = (new_x, new_y)
                            still_blocking = self._would_block_path_jax(agent_pos, goal_pos, new_blocking_pos, obstacle_mask)

                            return (key, agent_at_arr, agent_pos_arr, occupied, iter_count + 1, still_blocking)

                        def resample_cond(loop_carry):
                            _, _, _, _, iter_count, still_blocking = loop_carry
                            return still_blocking & (iter_count < 100)

                        # Only resample if blocking
                        def do_resample_loop():
                            result = jax.lax.while_loop(
                                resample_cond,
                                resample_loop,
                                (key, agent_at_arr, agent_pos_arr, occupied, 0, True)
                            )
                            return (result[0], result[1], result[2], result[3])

                        key, agent_at_arr, agent_pos_arr, occupied = jax.lax.cond(
                            is_blocking,
                            do_resample_loop,
                            lambda: (key, agent_at_arr, agent_pos_arr, occupied)
                        )

                        return (key, agent_at_arr, agent_pos_arr, occupied), None

                    # Iterate over uncontrolled agents
                    uncontrolled_agents_arr = jnp.array(self.uncontrolled_agents, dtype=jnp.int32)
                    carry, _ = jax.lax.scan(check_uncontrolled, carry, uncontrolled_agents_arr)
                    return carry, None

                # Iterate over controlled agents
                controlled_agents_arr = jnp.array(self.controlled_agents, dtype=jnp.int32)
                carry, _ = jax.lax.scan(check_controlled, (key, agent_at_arr, agent_pos_arr, occupied), controlled_agents_arr)
                return carry

            return jax.lax.cond(
                should_check,
                do_resampling,
                lambda c: c,
                carry_input
            )

        key, agent_at, agent_positions, occupied = resample_if_needed((key, agent_at, agent_positions, occupied))

        # Update environment state
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
        image = self.env._visualizer.render(state, env_state.actions, env_state.subs)

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
