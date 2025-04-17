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
        "DISTANCE_TO_FURTHEST_FOOD_REW": 1.0, # Reward for moving further from food (H2)
        "SEQUENCE_REW": 1.0, # Reward for completing a sequence of actions (H3-H8)
        # "FOLLOWING_TEAMMATE_REW": 1.0, # Reward for following another agent (H9)
        "CENTERED_FOOD_DISTANCE_REW": 1.0, # Reward for moving towards towards the food that is closest to the midpoint of the two agents
        "PROXIMITY_TO_TEAMMATE_REW": 1.0, # Reward for Proimity to teammate
        "COLLECT_FOOD_REW": 1.0,
    }    

    def __init__(self, env, agent_heuristics=None, reward_shaping_params=None, reward_shaping=False):
        super().__init__(env)

        self.reward_shaping = reward_shaping

        if self.reward_shaping:
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
        else:
            self.reward_shaping_params = {}
            self.agent_heuristics = {}

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state: WrappedEnvState, actions, params=None):
        obs, next_state, dense_reward, done, info = super().step(key, state, actions, params)

        if self.reward_shaping:
            shaped_rewards = self._extract_shaped_rewards(obs, next_state.env_state)
            total_reward = {
                agent: dense_reward[agent] + shaped_rewards[agent]
                for agent in self.agents
            }
            return obs, next_state, total_reward, done, info

        return obs, next_state, dense_reward, done, info

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

    #         # Collect Food Reward
    #         collect_food_rew = self._calculate_collect_food_reward(obs[agent_id], state.env_state, agent_id, i)
    #         shaped_reward[agent_id] += collect_food_rew

    #     return shaped_reward

    def _extract_shaped_rewards(self, obs, env_state):
        shaped_rewards = {}

        for agent_index, agent_id in enumerate(self.agents):
            agent_reward = 0.0

            agent_reward += self._calculate_distance_to_nearest_food_reward(
                obs[agent_id], env_state, agent_id, agent_index
            )

            agent_reward += self._calculate_distance_to_furthest_food_reward(
                obs[agent_id], env_state, agent_id, agent_index
            )

            agent_reward += self._calculate_following_teammate_reward(
                obs[agent_id], env_state, agent_id, agent_index
            )

            agent_reward += self._calculate_centered_food_reward(
                obs[agent_id], env_state, agent_id, agent_index
            )

            agent_reward += self._calculate_proximity_to_teammate_reward(
                obs[agent_id], env_state, agent_id, agent_index
            )

            agent_reward += self._calculate_collect_food_reward(env_state, agent_id, agent_index)

            shaped_rewards[agent_id] = agent_reward

        return shaped_rewards

    def _calculate_distance_to_nearest_food_reward(self, agent_obs, env_state, agent_id: str, i: int):
        agent_pos = env_state.agents.position[i]
        food_positions = env_state.food_items.position
        food_eaten = env_state.food_items.eaten

        uneaten_food_positions = jnp.where(~food_eaten[:, None], food_positions, jnp.array([jnp.inf, jnp.inf]))
        distances = jnp.sum(jnp.abs(uneaten_food_positions - agent_pos), axis=1)
        min_dist = jnp.min(distances)

        reward_weight = self.reward_shaping_params[agent_id]["DISTANCE_TO_NEAREST_FOOD_REW"]
        reward = reward_weight / (min_dist + 1.0)
        return reward

    def _calculate_distance_to_furthest_food_reward(self, agent_obs, env_state, agent_id: str, i: int):
        agent_pos = env_state.agents.position[i]
        food_positions = env_state.food_items.position
        food_eaten = env_state.food_items.eaten

        uneaten_food_positions = jnp.where(~food_eaten[:, None], food_positions, jnp.array([-jnp.inf, -jnp.inf]))
        distances = jnp.sum(jnp.abs(uneaten_food_positions - agent_pos), axis=1)
        max_dist = jnp.max(distances)

        reward_weight = self.reward_shaping_params[agent_id]["DISTANCE_TO_FURTHEST_FOOD_REW"]
        reward = reward_weight / (max_dist + 1.0)

        return reward
    
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

    
    def _calculate_following_teammate_reward(self, agent_obs, env_state, agent_id: str, i: int):
        agent_pos = env_state.agents.position[i]
        teammate_pos = env_state.agents.position[1 - i]
        dist = jnp.sum(jnp.abs(agent_pos - teammate_pos))
        reward_weight = self.reward_shaping_params[agent_id]["FOLLOWING_TEAMMATE_REW"]
        reward = reward_weight / (dist + 1.0)
        return reward

    def _calculate_centered_food_reward(self, agent_obs, env_state, agent_id: str, i: int):
        agent_pos = env_state.agents.position[i]
        teammate_pos = env_state.agents.position[1 - i]
        midpoint = (agent_pos + teammate_pos) / 2.0

        food_positions = env_state.food_items.position
        food_eaten = env_state.food_items.eaten

        uneaten_food_positions = jnp.where(~food_eaten[:, None], food_positions, jnp.array([jnp.inf, jnp.inf]))
        dists_to_midpoint = jnp.sum(jnp.abs(uneaten_food_positions - midpoint), axis=1)
        min_idx = jnp.argmin(dists_to_midpoint)
        target_food_pos = uneaten_food_positions[min_idx]
        dist_to_target = jnp.sum(jnp.abs(agent_pos - target_food_pos))

        reward_weight = self.reward_shaping_params[agent_id]["CENTERED_FOOD_DISTANCE_REW"]
        reward = reward_weight / (dist_to_target + 1.0)
        return reward

    def _calculate_proximity_to_teammate_reward(self, agent_obs, env_state, agent_id: str, i: int):
        agent_pos = env_state.agents.position[i]
        teammate_pos = env_state.agents.position[1 - i]
        dist = jnp.sum(jnp.abs(agent_pos - teammate_pos))
        reward_weight = self.reward_shaping_params[agent_id]["PROXIMITY_TO_TEAMMATE_REW"]
        reward = reward_weight / (dist + 1)
        return reward

    def _calculate_collect_food_reward(self, env_state, agent_id: str, i: int) -> float:
        agent_pos = env_state.agents.position[i]
        food_pos = env_state.food_items.position
        food_eaten = env_state.food_items.eaten

        uneaten_food = ~food_eaten
        masked_food_pos = jnp.where(uneaten_food[:, None], food_pos, jnp.array([jnp.inf, jnp.inf]))

        dists = jnp.sum(jnp.abs(masked_food_pos - agent_pos), axis=1)

        adjacent = jnp.any(dists == 1)

        is_loading = env_state.agents.loading[i]
        collected = jnp.logical_and(is_loading, adjacent)

        reward = self.reward_shaping_params[agent_id]["COLLECT_FOOD_REW"]
        return jax.lax.select(collected, reward, 0.0)