import jax
import jax.numpy as jnp
from envs.jumanji_jaxmarl_wrapper import JumanjiToJaxMARL, WrappedEnvState
from jumanji.environments.routing.lbf.constants import LOAD
from typing import Dict, Any, List
from flax.struct import dataclass
from functools import partial

class RewardShapingJumanjiToJaxMARL(JumanjiToJaxMARL):
    """
    A wrapper for Jumanji environments that implements reward shaping.
    This wrapper modifies the reward structure of the environment to encourage
    certain behaviors or strategies.

    Agent ideas: 
    - H1. Agents under H1 will move towards the closest item from its current location and collect it.
          Process is repeated until no item is left.
    - H2. At the beginning of an episode, agents will move towards the furthest object from its location and collect it.
          Every time its targeted iteme is collected, the agent will move to collect the remaining item whose location is furthest
          from the agent's current location. Process is repeated until no item is left.
    - H3-H8. H3-H8 corresponds to a heuristic that collects items following one of six possible permuations
             of collecting the three items available in the environment.
    """

    REWARD_SHAPING_PARAMS = {
        "DISTANCE_TO_NEAREST_FOOD_REW": 1.0, # Reward for moving closer to food (H1)
        "DISTANCE_TO_FARTHEST_FOOD_REW": 1.0, # Reward for moving further from food (H2)
        "SEQUENCE_REW": 1.0, # Reward for completing a sequence of actions (H3-H8)
        "FOLLOWING_TEAMMATE_REW": 1.0, # Reward for following another agent
        "CENTERED_FOOD_DISTANCE_REW": 1.0, # Reward for moving towards towards the food that is closest to the midpoint of the two agents
        "PROXIMITY_TO_TEAMMATE_REW": 1.0, # Reward for Proimity to teammate
        "COLLECT_FOOD_REW": 1.0,
    }    

    def __init__(self, env, reward_shaping_params=None, agent_heuristics=None):
        super().__init__(env)

        self.reward_shaping_params = (
            {
                agent_id: dict(self.REWARD_SHAPING_PARAMS)
                for agent_id in self.agents
            }
            if reward_shaping_params is None
            else reward_shaping_params
        )

        self.agent_heuristics = (
            agent_heuristics if agent_heuristics is not None
            else {agent_id: None for agent_id in self.agents}
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state: WrappedEnvState, actions, params=None):
        obs, next_state, dense_reward_dict, dones, info = super().step(key, state, actions, params)

        shaped_rewards_dict = self._extract_shaped_rewards(obs, state, next_state, actions)
        total_reward_dict = {
            agent: dense_reward_dict[agent] + shaped_rewards_dict[agent]
            for agent in self.agents
        }
        return obs, next_state, total_reward_dict, dones, info

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        obs, state = super().reset(key, params)
        return obs, state

    # def _extract_shaped_rewards(self, obs, state, actions):
    #     """Calculate the shaped reward based on the original reward and the shaping parameters."""

    #     shaped_reward = {}

    #     for i, agent_id in enumerate(self.agents):
    #         agent_heuristic = self.agent_heuristics.get(agent_id, None)

    #         if agent_heuristic is None:
    #             shaped_reward[agent_id] = 0
    #             continue

    #         if agent_heuristic == "H1":
    #             distance_to_nearest_food_rew = self._calculate_distance_to_nearest_food_reward(obs[agent_id], state.env_state, agent_id, i)
    #             shaped_reward[agent_id] = distance_to_nearest_food_rew

    #         elif agent_heuristic == "H2":
    #             sequence_rew = self._calculate_distance_to_furthest_food_reward(obs[agent_id], state.env_state, agent_id, i)
    #             shaped_reward[agent_id] = sequence_rew

    #         elif agent_heuristic == "H3":
    #             sequence_H3 = [0, 1, 2]
    #             H3_sequence_rew = self._calculate_sequence_reward(obs[agent_id], state.env_state, agent_id, i, sequence = sequence_H3)
    #             shaped_reward[agent_id] = H3_sequence_rew

    #         elif agent_heuristic == "H4":
    #             sequence_H4 = [1, 0, 2]
    #             H4_sequence_rew = self._calculate_sequence_reward(obs[agent_id], state.env_state, agent_id, i, sequence = sequence_H4)
    #             shaped_reward[agent_id] = H4_sequence_rew

    #         elif agent_heuristic == "H5":
    #             sequence_H5 = [2, 0, 1]
    #             H5_sequence_rew = self._calculate_sequence_reward(obs[agent_id], state.env_state, agent_id, i, sequence = sequence_H5)
    #             shaped_reward[agent_id] = H5_sequence_rew

    #         elif agent_heuristic == "H6":
    #             sequence_H6 = [0, 2, 1]
    #             H6_sequence_rew = self._calculate_sequence_reward(obs[agent_id], state.env_state, agent_id, i, sequence = sequence_H6)
    #             shaped_reward[agent_id] = H6_sequence_rew

    #         elif agent_heuristic == "H7":
    #             sequence_H7 = [1, 2, 0]
    #             H7_sequence_rew = self._calculate_sequence_reward(obs[agent_id], state.env_state, agent_id, i, sequence = sequence_H7)
    #             shaped_reward[agent_id] = H7_sequence_rew

    #         elif agent_heuristic == "H8":
    #             sequence_H8 = [2, 1, 0]
    #             H8_sequence_rew = self._calculate_sequence_reward(obs[agent_id], state.env_state, agent_id, i, sequence = sequence_H8)
    #             shaped_reward[agent_id] = H8_sequence_rew

    #         else:
    #             shaped_reward[agent_id] = 0.0

    #         collect_food_rew = self._calculate_collect_food_reward(obs[agent_id], state.env_state, agent_id, i)
    #         shaped_reward[agent_id] += collect_food_rew

    #     return shaped_reward

    def _extract_shaped_rewards(self, obs, state: WrappedEnvState, new_state: WrappedEnvState, actions):
        shaped_rewards = {}

        for agent_index, agent_id in enumerate(self.agents):
            agent_reward = 0.0

            agent_reward += self._calculate_distance_to_nearest_food_reward(
                state, new_state, agent_id, agent_index
            )

            agent_reward += self._calculate_distance_to_farthest_food_reward(
                state, new_state, agent_id, agent_index
            )

            agent_reward += self._calculate_following_teammate_reward(
                state, new_state, agent_id, agent_index
            )

            agent_reward += self._calculate_centered_food_reward(
                state, new_state, agent_id, agent_index
            )

            agent_reward += self._calculate_proximity_to_teammate_reward(
                state, new_state, agent_id, agent_index
            )

            agent_reward += self._calculate_collect_food_reward(state, agent_id, agent_index, actions)

            shaped_rewards[agent_id] = agent_reward

        return shaped_rewards

    def _calculate_distance_to_nearest_food_reward(self, prev_state, new_state, agent_id: str, i: int):
        old_pos = prev_state.env_state.agents.position[i]
        new_pos = new_state.env_state.agents.position[i]

        food_pos = prev_state.env_state.food_items.position
        food_eaten = prev_state.env_state.food_items.eaten
        uneaten_mask = ~food_eaten

        masked_food_pos = jnp.where(uneaten_mask[:, None], food_pos, jnp.array([jnp.inf, jnp.inf]))

        old_dists = jnp.sum(jnp.abs(masked_food_pos - old_pos), axis=1)
        new_dists = jnp.sum(jnp.abs(masked_food_pos - new_pos), axis=1)

        target_idx = jnp.argmin(old_dists)
        old_dist = old_dists[target_idx]
        new_dist = new_dists[target_idx]

        reward_val = self.reward_shaping_params[agent_id]["DISTANCE_TO_NEAREST_FOOD_REW"]
        reward = reward_val * jnp.maximum((old_dist - new_dist), 0.0) / (old_dist + 1.0)

        return jnp.where(jnp.isfinite(old_dist), reward, 0.0)

    def _calculate_distance_to_farthest_food_reward(self, prev_state, new_state, agent_id: str, i: int):
        old_pos = prev_state.env_state.agents.position[i]
        new_pos = new_state.env_state.agents.position[i]

        food_pos = prev_state.env_state.food_items.position
        food_eaten = prev_state.env_state.food_items.eaten
        uneaten_mask = ~food_eaten

        masked_food_pos = jnp.where(uneaten_mask[:, None], food_pos, jnp.array([-jnp.inf, -jnp.inf]))

        old_dists = jnp.sum(jnp.abs(masked_food_pos - old_pos), axis=1)
        new_dists = jnp.sum(jnp.abs(masked_food_pos - new_pos), axis=1)

        target_idx = jnp.argmax(old_dists)
        old_dist = old_dists[target_idx]
        new_dist = new_dists[target_idx]

        reward_val = self.reward_shaping_params[agent_id]["DISTANCE_TO_FARTHEST_FOOD_REW"]
        reward = reward_val * jnp.maximum((old_dist - new_dist), 0.0) / (old_dist + 1.0)

        return jnp.where(jnp.isfinite(old_dist), reward, 0.0)
    
    # def _calculate_sequence_reward(self, agent_id: str, env_state, sequence: List[int], sequence_progress: Dict[str, int]):
    #     food_eaten = env_state.food_items.eaten
    #     current_progress = sequence_progress[agent_id]
    #     target_food_id = sequence[current_progress]
    #     target_food_eaten = food_eaten[target_food_id]

    #     reward = jax.lax.select(
    #         target_food_eaten,
    #         self.reward_shaping_params[agent_id]["SEQUENCE_REW"],
    #         0.0
    #     )

    #     new_progress = jax.lax.select(target_food_eaten, current_progress + 1, current_progress)
    #     sequence_progress = sequence_progress.copy()
    #     sequence_progress[agent_id] = new_progress

    #     return reward, sequence_progress
    
    def _calculate_following_teammate_reward(self, prev_state, new_state, agent_id: str, i: int):
        old_pos = prev_state.env_state.agents.position[i]
        new_pos = new_state.env_state.agents.position[i]

        old_teammate_pos = prev_state.env_state.agents.position[1 - i]
        new_teammate_pos = new_state.env_state.agents.position[1 - i]

        old_dist = jnp.sum(jnp.abs(old_pos - old_teammate_pos))
        new_dist = jnp.sum(jnp.abs(new_pos - new_teammate_pos))

        reward_val = self.reward_shaping_params[agent_id]["FOLLOWING_TEAMMATE_REW"]
        reward = reward_val * (jnp.maximum(old_dist - new_dist, 0.0) / (old_dist + 1.0) + 1.0 / (new_dist + 1.0))
        return jnp.where(jnp.isfinite(old_dist), reward, 0.0)
    
    def _calculate_centered_food_reward(self, prev_state, new_state, agent_id: str, i: int):
        old_pos = prev_state.env_state.agents.position[i]
        new_pos = new_state.env_state.agents.position[i]

        teammate_pos = prev_state.env_state.agents.position[1 - i]
        midpoint = (old_pos + teammate_pos) / 2.0

        food_pos = prev_state.env_state.food_items.position
        food_eaten = prev_state.env_state.food_items.eaten
        uneaten_mask = ~food_eaten
        masked_food_pos = jnp.where(uneaten_mask[:, None], food_pos, jnp.array([jnp.inf, jnp.inf]))

        midpoint_dists = jnp.sum(jnp.abs(masked_food_pos - midpoint), axis=1)
        target_idx = jnp.argmin(midpoint_dists)
        target_food_pos = masked_food_pos[target_idx]

        old_dist = jnp.sum(jnp.abs(old_pos - target_food_pos))
        new_dist = jnp.sum(jnp.abs(new_pos - target_food_pos))

        reward_val = self.reward_shaping_params[agent_id]["CENTERED_FOOD_DISTANCE_REW"]
        reward = reward_val * jnp.maximum((old_dist - new_dist), 0.0) / (old_dist + 1.0)
        return jnp.where(jnp.isfinite(old_dist), reward, 0.0)

    def _calculate_proximity_to_teammate_reward(self, prev_state, new_state, agent_id: str, i: int):
        new_pos = new_state.env_state.agents.position[i]
        new_teammate_pos = new_state.env_state.agents.position[1 - i]
        new_dist = jnp.sum(jnp.abs(new_pos - new_teammate_pos))
        reward_val = self.reward_shaping_params[agent_id]["PROXIMITY_TO_TEAMMATE_REW"]
        reward = reward_val * (1.0 / (new_dist + 1.0))
        return reward

    def _calculate_collect_food_reward(self, state, agent_id: str, i: int, actions: Dict[str, int]) -> float:
        agent_pos = state.env_state.agents.position[i]
        teammate_idx = 1 - i
        teammate_id = f'agent_{teammate_idx}'
        teammate_pos = state.env_state.agents.position[teammate_idx]

        food_pos = state.env_state.food_items.position
        food_eaten = state.env_state.food_items.eaten
        uneaten_mask = ~food_eaten
        masked_food_pos = jnp.where(uneaten_mask[:, None], food_pos, jnp.array([jnp.inf, jnp.inf]))

        agent_dists = jnp.sum(jnp.abs(masked_food_pos - agent_pos), axis=1)
        teammate_dists = jnp.sum(jnp.abs(masked_food_pos - teammate_pos), axis=1)
        both_adjacent = jnp.logical_and(agent_dists == 1, teammate_dists == 1)
        agent_loading = actions[agent_id] == LOAD
        teammate_loading = actions[teammate_id] == LOAD
        both_loading = jnp.logical_and(agent_loading, teammate_loading)
        collecting = jnp.logical_and(both_adjacent, both_loading)
        successful = jnp.any(collecting)

        reward_val = self.reward_shaping_params[agent_id]["COLLECT_FOOD_REW"]
        return jnp.where(successful, reward_val, 0.0)