import jax
import jax.numpy as jnp
from envs.jumanji_jaxmarl_wrapper import JumanjiToJaxMARL
from jumanji.environments.routing.lbf.constants import LOAD
from typing import Dict, Any
from flax.struct import dataclass
from functools import partial

@dataclass
class RewardShapingEnvState:
    env_state: Any # a jumanji state
    prev_env_state: Any # a jumanji state
    avail_actions: jnp.ndarray
    step: jnp.array

REWARD_SHAPING_PARAMS = {
    "agent_0": {
    "DISTANCE_TO_NEAREST_FOOD_REW": 0.0, # Reward for moving closer to food (H1)
    "DISTANCE_TO_FARTHEST_FOOD_REW": 0.0, # Reward for moving further from food (H2)
    "SEQUENCE_REW": 0.0, # Reward for completing a sequence of actions (H3-H8)
    "FOLLOWING_TEAMMATE_REW": 0.0, # Reward for following another agent
    "CENTERED_FOOD_DISTANCE_REW": 0.5, # Reward for moving towards towards the food that is closest to the midpoint of the two agents
    "PROXIMITY_TO_TEAMMATE_REW": 0.2, # Reward for proximity to teammate
    "COLLECT_FOOD_REW": 3.0,
    },
    "agent_1": {
    "DISTANCE_TO_NEAREST_FOOD_REW": 0.0, # Reward for moving closer to food (H1)
    "DISTANCE_TO_FARTHEST_FOOD_REW": 0.0, # Reward for moving further from food (H2)
    "SEQUENCE_REW": 0.0, # Reward for completing a sequence of actions (H3-H8)
    "FOLLOWING_TEAMMATE_REW": 1.0, # Reward for following another agent
    "CENTERED_FOOD_DISTANCE_REW": 0.0, # Reward for moving towards towards the food that is closest to the midpoint of the two agents
    "PROXIMITY_TO_TEAMMATE_REW": 1.0, # Reward for proximity to teammate
    "COLLECT_FOOD_REW": 3.0,
    },
    "REWARD_SHAPING_COEF": 0.01,
}    

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

    def __init__(self, env, share_rewards: bool = False):
        super().__init__(env, share_rewards)

        self.reward_shaping_params = REWARD_SHAPING_PARAMS

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        env_state, timestep = self.env.reset(key)
        obs = self._extract_observations(timestep.observation)
        state = RewardShapingEnvState(env_state, 
                                      env_state,
                                      self._extract_avail_actions(timestep),
                                      timestep.observation.step_count)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state: RewardShapingEnvState, actions, params=None):
        key, key_reset = jax.random.split(key)

        prev_env_state = state.env_state

        actions_array = self._actions_to_array(actions)
        next_env_state, timestep = self.env.step(state.env_state, actions_array)
        avail_actions = self._extract_avail_actions(timestep)

        next_state = RewardShapingEnvState(
            env_state=next_env_state,
            prev_env_state=prev_env_state,
            avail_actions=avail_actions,
            step=timestep.observation.step_count,
        )

        next_obs = self._extract_observations(timestep.observation)
        reward = self._extract_rewards(timestep.reward)
        done = self._extract_dones(timestep)
        info = self._extract_infos(timestep)

        shaped_rewards_dict = self._extract_shaped_rewards(next_obs, next_state.prev_env_state, next_state.env_state, actions)
        total_reward_dict = {
            agent: reward[agent] + (REWARD_SHAPING_PARAMS["REWARD_SHAPING_COEF"] * shaped_rewards_dict[agent])
            for agent in self.agents
        }

        reset_obs, reset_state = self.reset(key_reset)
        reset_state = reset_state.replace(prev_env_state=reset_state.env_state)

        (obs, state) = jax.tree_util.tree_map(
            lambda reset_val, next_val: jax.lax.select(done["__all__"], reset_val, next_val),
            (reset_obs, reset_state),
            (next_obs, next_state),
        )
        
        # add the original reward to the info dict
        original_reward = jnp.array([reward[agent] for agent in self.agents])
        # create a new info dictionary with all the keys of the original info dictionary plus the inew original_reward key
        new_info = {**info, "original_reward": original_reward}
        return obs, state, total_reward_dict, done, new_info

    def _extract_shaped_rewards(self, obs, prev_env_state, env_state, actions):
        shaped_rewards = {}

        for agent_index, agent_id in enumerate(self.agents):
            agent_reward = 0.0

            agent_reward += self._calculate_distance_to_nearest_food_reward(
                prev_env_state, env_state, agent_id, agent_index
            )

            agent_reward += self._calculate_distance_to_farthest_food_reward(
                prev_env_state, env_state, agent_id, agent_index
            )

            agent_reward += self._calculate_following_teammate_reward(
                prev_env_state, env_state, agent_id, agent_index
            )

            agent_reward += self._calculate_centered_food_reward(
                prev_env_state, env_state, agent_id, agent_index
            )

            agent_reward += self._calculate_proximity_to_teammate_reward(
                prev_env_state, env_state, agent_id, agent_index
            )

            agent_reward += self._calculate_collect_food_reward(env_state, agent_id, agent_index, actions)

            shaped_rewards[agent_id] = agent_reward

        return shaped_rewards

    def _calculate_distance_to_nearest_food_reward(self, prev_state, new_state, agent_id: str, i: int):
        old_pos = prev_state.agents.position[i]
        new_pos = new_state.agents.position[i]

        food_pos = prev_state.food_items.position
        food_eaten = prev_state.food_items.eaten
        uneaten_mask = ~food_eaten

        masked_food_pos = jnp.where(uneaten_mask[:, None], food_pos, jnp.array([jnp.inf, jnp.inf]))
        old_dists = jnp.sum(jnp.abs(masked_food_pos - old_pos), axis=1)
        new_dists = jnp.sum(jnp.abs(masked_food_pos - new_pos), axis=1)

        target_idx = jnp.argmin(old_dists)
        old_dist = old_dists[target_idx]
        new_dist = new_dists[target_idx]
        dist_change = new_dist - old_dist
        reward_val = self.reward_shaping_params[agent_id]["DISTANCE_TO_NEAREST_FOOD_REW"]
        reward = reward_val * jnp.tanh(-dist_change.astype(jnp.float32))

        valid_mask = jnp.any(uneaten_mask) & jnp.isfinite(old_dist) & jnp.isfinite(new_dist)
        reward = jnp.where(valid_mask, reward, 0.0)
        return reward
    
    def _calculate_distance_to_farthest_food_reward(self, prev_state, new_state, agent_id: str, i: int):
        old_pos = prev_state.agents.position[i]
        new_pos = new_state.agents.position[i]

        food_pos = prev_state.food_items.position
        food_eaten = prev_state.food_items.eaten
        uneaten_mask = ~food_eaten

        masked_food_pos = jnp.where(uneaten_mask[:, None], food_pos, jnp.array([-jnp.inf, -jnp.inf]))

        old_dists = jnp.sum(jnp.abs(masked_food_pos - old_pos), axis=1)
        new_dists = jnp.sum(jnp.abs(masked_food_pos - new_pos), axis=1)

        target_idx = jnp.argmax(old_dists)
        old_dist = old_dists[target_idx]
        new_dist = new_dists[target_idx]

        dist_change = new_dist - old_dist
        reward_val = self.reward_shaping_params[agent_id]["DISTANCE_TO_FARTHEST_FOOD_REW"]
        reward = reward_val * jnp.tanh(-dist_change.astype(jnp.float32))

        valid_mask = jnp.any(uneaten_mask) & jnp.isfinite(old_dist) & jnp.isfinite(new_dist)
        reward = jnp.where(valid_mask, reward, 0.0)

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
    
    def _calculate_following_teammate_reward(self, prev_state, new_state, agent_id: str, i: int):
        old_pos = prev_state.agents.position[i]
        new_pos = new_state.agents.position[i]

        old_teammate_pos = prev_state.agents.position[1 - i]
        new_teammate_pos = new_state.agents.position[1 - i]

        old_dist = jnp.sum(jnp.abs(old_pos - old_teammate_pos))
        new_dist = jnp.sum(jnp.abs(new_pos - new_teammate_pos))
        dist_change = new_dist - old_dist 

        reward_val = self.reward_shaping_params[agent_id]["FOLLOWING_TEAMMATE_REW"]
        reward = reward_val * jnp.tanh(-dist_change.astype(jnp.float32))

        valid_mask = (
            jnp.all(jnp.isfinite(old_pos)) &
            jnp.all(jnp.isfinite(new_pos)) &
            jnp.all(jnp.isfinite(old_teammate_pos)) &
            jnp.all(jnp.isfinite(new_teammate_pos))
        )
        reward = jnp.where(valid_mask, reward, 0.0)
        return reward

    
    def _calculate_centered_food_reward(self, prev_state, new_state, agent_id: str, i: int):
        old_pos = prev_state.agents.position[i]
        new_pos = new_state.agents.position[i]

        teammate_pos = prev_state.agents.position[1 - i]
        midpoint = (old_pos + teammate_pos) / 2.0

        food_pos = prev_state.food_items.position
        food_eaten = prev_state.food_items.eaten
        uneaten_mask = ~food_eaten

        has_food = jnp.any(uneaten_mask)

        masked_food_pos = jnp.where(uneaten_mask[:, None], food_pos, jnp.array([jnp.inf, jnp.inf]))

        midpoint_dists = jnp.sum(jnp.abs(masked_food_pos - midpoint), axis=1)
        target_idx = jnp.argmin(midpoint_dists)
        target_food_pos = masked_food_pos[target_idx]

        old_dist = jnp.sum(jnp.abs(old_pos - target_food_pos))
        new_dist = jnp.sum(jnp.abs(new_pos - target_food_pos))
        dist_change = new_dist - old_dist

        reward_val = self.reward_shaping_params[agent_id]["CENTERED_FOOD_DISTANCE_REW"]
        reward = reward_val * jnp.tanh(-dist_change.astype(jnp.float32))

        valid_mask = (
            jnp.all(jnp.isfinite(old_pos)) &
            jnp.all(jnp.isfinite(new_pos)) &
            jnp.all(jnp.isfinite(teammate_pos)) &
            jnp.all(jnp.isfinite(target_food_pos)) &
            has_food
        )

        reward = jnp.where(valid_mask, reward, 0.0)
        return reward


    def _calculate_proximity_to_teammate_reward(self, prev_state, new_state, agent_id: str, i: int):
        new_pos = new_state.agents.position[i]
        teammate_pos = new_state.agents.position[1 - i]

        new_dist = jnp.sum(jnp.abs(new_pos - teammate_pos))

        reward_val = self.reward_shaping_params[agent_id]["PROXIMITY_TO_TEAMMATE_REW"]
        reward = reward_val * jnp.tanh(-new_dist.astype(jnp.float32))

        valid_mask = jnp.all(jnp.isfinite(new_pos)) & jnp.all(jnp.isfinite(teammate_pos))
        reward = jnp.where(valid_mask, reward, 0.0)
        return reward


    def _calculate_collect_food_reward(self, state, agent_id: str, i: int, actions: Dict[str, int]) -> float:
        agent_pos = state.agents.position[i]
        teammate_idx = 1 - i
        teammate_id = f'agent_{teammate_idx}'
        teammate_pos = state.agents.position[teammate_idx]

        food_pos = state.food_items.position
        food_eaten = state.food_items.eaten
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
        reward = jnp.where(jnp.any(uneaten_mask), reward_val, 0.0)
        return jnp.where(successful, reward, 0.0)