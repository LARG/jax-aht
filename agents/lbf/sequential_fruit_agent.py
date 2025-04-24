from collections import deque
from functools import partial
from typing import Tuple, Dict, Any, Set, List

import jax
import jax.numpy as jnp
from flax import struct
from jumanji.environments.routing.lbf.types import State as LBFState

from agents.lbf.base_agent import BaseAgent, AgentState


class SequentialFruitAgent(BaseAgent):
    """Goes fruit-by-fruit in lexicographic order of their initial positions."""

    @struct.dataclass
    class SeqAgentState:
        rng_key: jnp.ndarray               # inherited randomness
        sequence: jnp.ndarray              # shape (num_fruits, 2)
        idx: jnp.ndarray                   # scalar index into sequence
        last_pos: jnp.ndarray              # last position of the agent

    def __init__(self, agent_id: int):
        super().__init__(agent_id)

    def reset(self, env_state: LBFState) -> SeqAgentState:
        # 1) Extract all fruit positions at the start of the episode
        positions = jnp.stack([food for food in env_state.food_items.position], axis=0)  # (F,2)
        # 2) Sort them lexicographically by (row, col)
        order = jnp.lexsort((positions[:,1], positions[:,0]))
        seq = positions[order]
        # 3) Start at the first fruit
        return SequentialFruitAgent.SeqAgentState(
            rng_key=self.initial_state.rng_key,
            sequence=seq,
            idx=jnp.array(0, dtype=jnp.int32),
            last_pos=None,
        )

    def init_hstate(self, batch_size):
        return None

    def plan_path(
        self,
        start: Tuple[int,int],
        goal:  Tuple[int,int],
        obstacles: Set[Tuple[int,int]],
        grid_shape: Tuple[int,int],
    ) -> Tuple[Tuple[int,int], ...]:
        """BFS from start to goal avoiding obstacles. Returns full path."""
        rows, cols = grid_shape
        frontier   = deque([start])
        came_from  = {start: None}

        while frontier:
            cur = frontier.popleft()
            if cur == goal:
                break
            for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                nb = (cur[0]+dr, cur[1]+dc)
                if (0 <= nb[0] < rows
                    and 0 <= nb[1] < cols
                    and nb not in obstacles
                    and nb not in came_from):
                    came_from[nb] = cur
                    frontier.append(nb)

        # reconstruct path
        if goal not in came_from:
            return (start,)
        path = []
        node = goal
        while node is not None:
            path.append(node)
            node = came_from[node]
        return tuple(reversed(path))

    def get_action(
        self,
        obs: jnp.ndarray,
        env_state: LBFState,
        agent_state: SeqAgentState,
    ) -> Tuple[int, SeqAgentState]:

        if agent_state is None:
            agent_state = self.reset(env_state)

        seq = agent_state.sequence
        i = int(agent_state.idx)
        target = tuple(map(int, seq[i]))      
        agent_pos = tuple(map(int, env_state.agents.position[self.agent_id]))

        # 1) Has the fruit disappeared? Advance index if so.
        food_pos = [tuple(map(int, p)) for p in env_state.food_items.position]
        eaten = list(env_state.food_items.eaten)
        still_there = any((p == target and not e) for p,e in zip(food_pos, eaten))
        if not still_there and i < seq.shape[0] - 1:
            i += 1
            target = tuple(map(int, seq[i]))

        # 2) If we’re on it, load:
        if jnp.abs(agent_pos[0] - target[0]) + jnp.abs(agent_pos[1] - target[1]) <= 1:
            action = 5  # LOAD
            rng_key, subkey = jax.random.split(agent_state.rng_key)
        else:
            # build obstacle set = other agents
            obstacles = {
                tuple(map(int, pos))
                for idx, pos in enumerate(env_state.agents.position)
                if idx != self.agent_id
            }
            # infer grid dims from max coords + 1
            all_rows = list(env_state.agents.position[:,0]) + list(env_state.food_items.position[:,0])
            all_cols = list(env_state.agents.position[:,1]) + list(env_state.food_items.position[:,1])
            grid_shape = (int(max(all_rows)) + 1, int(max(all_cols)) + 1)
            rows, cols = grid_shape

            planning_target = target
            if agent_state.last_pos == agent_pos:
                # randomize the target if we are stuck
                raw_neighbours = [
                    (target[0] - 1, target[1]),  # up
                    (target[0] + 1, target[1]),  # down
                    (target[0], target[1] - 1),  # left
                    (target[0], target[1] + 1),  # right
                ]
                valid_neighbours = [
                    (r, c)
                    for (r, c) in raw_neighbours
                    if 0 <= r < rows
                    and 0 <= c < cols
                    and (r, c) not in obstacles
                ]
                rng_key, subkey = jax.random.split(agent_state.rng_key)
                planning_target = jax.random.choice(
                    subkey,
                    jnp.array(valid_neighbours, dtype=jnp.int32),
                )
                planning_target = tuple(map(int,planning_target))
                print("new planning target is ", planning_target)
            else:
                rng_key, _ = jax.random.split(agent_state.rng_key)
            path = self.plan_path(agent_pos, planning_target, obstacles, grid_shape)
            next_step = path[1] if len(path) > 1 else agent_pos
            dr, dc = next_step[0] - agent_pos[0], next_step[1] - agent_pos[1]

            # map to move
            if   dr == -1: action = 1  # NORTH
            elif dr ==  1: action = 2  # SOUTH
            elif dc == -1: action = 3  # WEST
            elif dc ==  1: action = 4  # EAST
            else: action = 0           # NOOP (shouldn’t happen)

        new_state = SequentialFruitAgent.SeqAgentState(
            rng_key = rng_key,
            sequence = seq,
            idx = jnp.array(i, jnp.int32),
            last_pos = agent_pos,
        )
        return action, new_state