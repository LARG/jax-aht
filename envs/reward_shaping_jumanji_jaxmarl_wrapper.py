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
    #  [0] = nearest, [1] = farthest, [2] = centered
    target_food_idx: jnp.ndarray      # shape (num_agents, 3)
    avail_actions: jnp.ndarray
    step: jnp.array

REWARD_SHAPING_PARAMS = {
    "agent_0": {
    "DISTANCE_TO_NEAREST_FOOD_REW": 0.0, # Reward for moving closer to food (H1)
    "DISTANCE_TO_FARTHEST_FOOD_REW": 1.0, # Reward for moving further from food (H2)
    "SEQUENCE_REW": 0.0, # Reward for completing a sequence of actions (H3-H8)
    "FOLLOWING_TEAMMATE_REW": 0.0, # Reward for following another agent
    "CENTERED_FOOD_DISTANCE_REW": 0.0, # Reward for moving towards towards the food that is closest to the midpoint of the two agents
    "PROXIMITY_TO_TEAMMATE_REW": 0.0, # Reward for proximity to teammate
    "COLLECT_FOOD_REW": 0.0,
    },
    "agent_1": {
    "DISTANCE_TO_NEAREST_FOOD_REW": 0.0, # Reward for moving closer to food (H1)
    "DISTANCE_TO_FARTHEST_FOOD_REW": 0.0, # Reward for moving further from food (H2)
    "SEQUENCE_REW": 0.0, # Reward for completing a sequence of actions (H3-H8)
    "FOLLOWING_TEAMMATE_REW": 1.0, # Reward for following another agent
    "CENTERED_FOOD_DISTANCE_REW": 0.0, # Reward for moving towards towards the food that is closest to the midpoint of the two agents
    "PROXIMITY_TO_TEAMMATE_REW": 0.0, # Reward for proximity to teammate
    "COLLECT_FOOD_REW": 0.0,
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


    def _compute_initial_targets(self, env_state):
        """
        Returns an array of shape (num_agents, 3):
         - col 0: index of the nearest uneaten food from each agent
         - col 1: index of the farthest uneaten food from each agent
         - col 2: index of the food closest to the midpoint between each agent & its teammate
        """
        food_pos = env_state.food_items.position  
        eaten = env_state.food_items.eaten      
        uneaten_mask = ~eaten                          
        agent_pos = env_state.agents.position       
        n_agents = agent_pos.shape[0]

        dists = jnp.sum(jnp.abs(food_pos[None, :, :] - agent_pos[:, None, :]), axis=-1)

        nearest_idxs  = jnp.argmin(jnp.where(uneaten_mask,  dists, jnp.inf), axis=1)
        farthest_idxs = jnp.argmax(jnp.where(uneaten_mask, dists, -jnp.inf), axis=1)

        teammate_idx  = (jnp.arange(n_agents) + 1) % n_agents
        teammate_pos  = agent_pos[teammate_idx]               
        midpoint      = (agent_pos + teammate_pos) / 2.0

        dists_mid = jnp.sum(jnp.abs(food_pos[None, :, :] - midpoint[:, None, :]), axis=-1)
        centered_idxs = jnp.argmin(jnp.where(uneaten_mask, dists_mid, jnp.inf), axis=1)

        init_targets = jnp.stack([
            nearest_idxs,    # col 0
            farthest_idxs,   # col 1
            centered_idxs    # col 2
        ], axis=1).astype(jnp.int32)

        return init_targets
         

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        env_state, timestep = self.env.reset(key)
        #target_food_idx = jnp.full((self.num_agents,), -1, dtype=jnp.int32)
        init_targets = self._compute_initial_targets(env_state)
        obs = self._extract_observations(timestep.observation)
        state = RewardShapingEnvState(env_state, 
                                      env_state,
                                      target_food_idx=init_targets,
                                      avail_actions=self._extract_avail_actions(timestep),
                                      step=timestep.observation.step_count)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state: RewardShapingEnvState, actions, params=None):
        key, key_reset = jax.random.split(key)

        prev_env_state = state.env_state
        target_food_idx = state.target_food_idx
        actions_array = self._actions_to_array(actions)
        next_env_state, timestep = self.env.step(state.env_state, actions_array)
        avail_actions = self._extract_avail_actions(timestep)
        
        next_obs = self._extract_observations(timestep.observation)
        reward = self._extract_rewards(timestep.reward)
        done = self._extract_dones(timestep)
        info = self._extract_infos(timestep)

        shaped_rewards_dict, updated_target_food_idx = self._extract_shaped_rewards(prev_env_state, next_env_state, actions, target_food_idx)
        total_reward_dict = {
            agent: reward[agent] + (REWARD_SHAPING_PARAMS["REWARD_SHAPING_COEF"] * shaped_rewards_dict[agent])
            for agent in self.agents
        }

        next_state = RewardShapingEnvState(
            env_state=next_env_state,
            prev_env_state=prev_env_state,
            target_food_idx=updated_target_food_idx,
            avail_actions=avail_actions,
            step=timestep.observation.step_count,
        )

        reset_obs, reset_state = self.reset(key_reset)
        # reset_state = reset_state.replace(prev_env_state=reset_state.env_state)
        # reset_state = reset_state.replace(target_food_idx=jnp.full((self.num_agents, 3), -1))

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

    # def _extract_shaped_rewards(self, obs, prev_env_state, env_state, actions):
    #     shaped_rewards = {}

    #     for agent_index, agent_id in enumerate(self.agents):
    #         agent_reward = 0.0

    #         agent_reward += self._calculate_distance_to_nearest_food_reward(
    #             prev_env_state, env_state, agent_id, agent_index
    #         )

    #         agent_reward += self._calculate_distance_to_farthest_food_reward(
    #             prev_env_state, env_state, agent_id, agent_index
    #         )

    #         agent_reward += self._calculate_following_teammate_reward(
    #             prev_env_state, env_state, agent_id, agent_index
    #         )

    #         agent_reward += self._calculate_centered_food_reward(
    #             prev_env_state, env_state, agent_id, agent_index
    #         )

    #         agent_reward += self._calculate_proximity_to_teammate_reward(
    #             prev_env_state, env_state, agent_id, agent_index
    #         )

    #         agent_reward += self._calculate_collect_food_reward(env_state, agent_id, agent_index, actions)

    #         shaped_rewards[agent_id] = agent_reward

    #     return shaped_rewards

    def _extract_shaped_rewards(self, prev_env_state, env_state, actions, target_food_idx):
        #target_food_idx = jnp.broadcast_to(target_food_idx, (self.num_agents, 3)) # [0] = nearest, [1] = farthest, [2] = centered
        shaped_rewards = {}
        updated_target_indices = []

        for agent_index, agent_id in enumerate(self.agents):
            total_shaped_reward = 0.0
            current_target_nearest = target_food_idx[agent_index, 0]
            current_target_farthest = target_food_idx[agent_index, 1]
            current_target_centered = target_food_idx[agent_index, 2]

            nearest_food_reward, nearest_target_index = self._calculate_distance_to_nearest_food_reward(
                prev_env_state, env_state, agent_id, agent_index, current_target_nearest
            )
            total_shaped_reward += nearest_food_reward

            farthest_food_reward, farthest_target_index = self._calculate_distance_to_farthest_food_reward(
                prev_env_state, env_state, agent_id, agent_index, current_target_farthest
            )
            total_shaped_reward += farthest_food_reward

            centered_food_reward, centered_target_index = self._calculate_centered_food_reward(
                prev_env_state, env_state, agent_id, agent_index, current_target_centered
            )
            total_shaped_reward += centered_food_reward

            following_reward = self._calculate_following_teammate_reward(
                prev_env_state, env_state, agent_id, agent_index
            )
            total_shaped_reward += following_reward

            proximity_reward = self._calculate_proximity_to_teammate_reward(
                prev_env_state, env_state, agent_id, agent_index
            )
            total_shaped_reward += proximity_reward

            collect_food_reward = self._calculate_collect_food_reward(
                env_state, agent_id, agent_index, actions
            )
            total_shaped_reward += collect_food_reward

            shaped_rewards[agent_id] = total_shaped_reward
            
            # new_nearest_target   = jax.lax.select(nearest_food_reward != 0.0, nearest_target_index,  current_target_nearest)
            # new_farthest_target    = jax.lax.select(farthest_food_reward != 0.0, farthest_target_index, current_target_farthest)
            # new_center_target = jax.lax.select(centered_food_reward != 0.0, centered_target_index, current_target_centered)

            updated_index = jnp.stack([nearest_target_index, farthest_target_index, centered_target_index])
            updated_target_indices.append(updated_index)

        updated_target_food_idx = jnp.stack(updated_target_indices, axis=0)
        return shaped_rewards, updated_target_food_idx

    # def _calculate_distance_to_nearest_food_reward(self, prev_state, new_state, agent_id: str, i: int):
    #     old_pos = prev_state.agents.position[i]
    #     new_pos = new_state.agents.position[i]

    #     food_pos = prev_state.food_items.position
    #     food_eaten = prev_state.food_items.eaten
    #     uneaten_mask = ~food_eaten

    #     masked_food_pos = jnp.where(uneaten_mask[:, None], food_pos, jnp.array([jnp.inf, jnp.inf]))
    #     old_dists = jnp.sum(jnp.abs(masked_food_pos - old_pos), axis=1)
    #     new_dists = jnp.sum(jnp.abs(masked_food_pos - new_pos), axis=1)

    #     target_idx = jnp.argmin(old_dists)
    #     old_dist = old_dists[target_idx]
    #     new_dist = new_dists[target_idx]
    #     dist_change = new_dist - old_dist
    #     reward_val = self.reward_shaping_params[agent_id]["DISTANCE_TO_NEAREST_FOOD_REW"]
    #     reward = reward_val * jnp.tanh(-dist_change.astype(jnp.float32))

    #     valid_mask = jnp.any(uneaten_mask) & jnp.isfinite(old_dist) & jnp.isfinite(new_dist)
    #     reward = jnp.where(valid_mask, reward, 0.0)
    #     return reward

    def _calculate_distance_to_nearest_food_reward(self, prev_state, new_state, agent_id: str, i: int, target_food_idx):
        old_pos = prev_state.agents.position[i]
        new_pos = new_state.agents.position[i]

        food_pos = prev_state.food_items.position
        food_eaten = new_state.food_items.eaten
        uneaten_mask = ~food_eaten
        has_food = jnp.any(uneaten_mask)

        current_target_eaten = food_eaten[target_food_idx]
        needs_update = current_target_eaten

        dists = jnp.sum(jnp.abs(food_pos - old_pos), axis=1)
        new_target_idx = jnp.argmin(jnp.where(uneaten_mask, dists, jnp.inf))

        final_target_idx = jnp.where(needs_update, new_target_idx, target_food_idx)

        target_food_pos = food_pos[final_target_idx]
        old_dist = jnp.sum(jnp.abs(old_pos - target_food_pos))
        new_dist = jnp.sum(jnp.abs(new_pos - target_food_pos))
        dist_change = new_dist - old_dist

        reward_val = self.reward_shaping_params[agent_id]["DISTANCE_TO_NEAREST_FOOD_REW"]
        reward = reward_val * jnp.tanh(-dist_change.astype(jnp.float32))

        valid_mask = has_food & jnp.isfinite(old_dist) & jnp.isfinite(new_dist)
        reward = jnp.where(valid_mask, reward, 0.0)
        return reward, final_target_idx

    # def _calculate_distance_to_farthest_food_reward(self, prev_state, new_state, agent_id: str, i: int):
    #     old_pos = prev_state.agents.position[i]
    #     new_pos = new_state.agents.position[i]

    #     food_pos = prev_state.food_items.position
    #     food_eaten = prev_state.food_items.eaten
    #     uneaten_mask = ~food_eaten

    #     masked_food_pos = jnp.where(uneaten_mask[:, None], food_pos, jnp.array([-jnp.inf, -jnp.inf]))

    #     old_dists = jnp.sum(jnp.abs(masked_food_pos - old_pos), axis=1)
    #     new_dists = jnp.sum(jnp.abs(masked_food_pos - new_pos), axis=1)

    #     target_idx = jnp.argmax(old_dists)
    #     old_dist = old_dists[target_idx]
    #     new_dist = new_dists[target_idx]

    #     dist_change = new_dist - old_dist
    #     reward_val = self.reward_shaping_params[agent_id]["DISTANCE_TO_FARTHEST_FOOD_REW"]
    #     reward = reward_val * jnp.tanh(-dist_change.astype(jnp.float32))

    #     valid_mask = jnp.any(uneaten_mask) & jnp.isfinite(old_dist) & jnp.isfinite(new_dist)
    #     reward = jnp.where(valid_mask, reward, 0.0)

    #     return reward

    def _calculate_distance_to_farthest_food_reward(self, prev_state, new_state, agent_id: str, i: int, target_food_idx: jnp.ndarray):
        old_pos = prev_state.agents.position[i]
        new_pos = new_state.agents.position[i]

        old_target_idx = target_food_idx

        food_pos = prev_state.food_items.position
        food_eaten = new_state.food_items.eaten
        uneaten_mask = ~food_eaten
        has_food = jnp.any(uneaten_mask)

        current_target_eaten = food_eaten[old_target_idx]
        needs_update = current_target_eaten

        dists = jnp.sum(jnp.abs(food_pos - old_pos), axis=1)
        new_target_idx = jnp.argmax(jnp.where(uneaten_mask, dists, -jnp.inf))

        final_target_idx = jnp.where(needs_update, new_target_idx, old_target_idx)
        target_food_pos = food_pos[final_target_idx]

        old_dist = jnp.sum(jnp.abs(old_pos - target_food_pos))
        new_dist = jnp.sum(jnp.abs(new_pos - target_food_pos))

        dist_change = new_dist - old_dist

        reward_val = self.reward_shaping_params[agent_id]["DISTANCE_TO_FARTHEST_FOOD_REW"]
        reward = reward_val * jnp.tanh(-dist_change.astype(jnp.float32))
        valid_mask = has_food & jnp.isfinite(old_dist) & jnp.isfinite(new_dist)
        reward = jnp.where(valid_mask, reward, 0.0)
        return reward, final_target_idx

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

    
    # def _calculate_centered_food_reward(self, prev_state, new_state, agent_id: str, i: int):
    #     old_pos = prev_state.agents.position[i]
    #     new_pos = new_state.agents.position[i]

    #     teammate_pos = prev_state.agents.position[1 - i]
    #     midpoint = (old_pos + teammate_pos) / 2.0

    #     food_pos = prev_state.food_items.position
    #     food_eaten = prev_state.food_items.eaten
    #     uneaten_mask = ~food_eaten

    #     has_food = jnp.any(uneaten_mask)

    #     masked_food_pos = jnp.where(uneaten_mask[:, None], food_pos, jnp.array([jnp.inf, jnp.inf]))

    #     midpoint_dists = jnp.sum(jnp.abs(masked_food_pos - midpoint), axis=1)
    #     target_idx = jnp.argmin(midpoint_dists)
    #     target_food_pos = masked_food_pos[target_idx]

    #     old_dist = jnp.sum(jnp.abs(old_pos - target_food_pos))
    #     new_dist = jnp.sum(jnp.abs(new_pos - target_food_pos))
    #     dist_change = new_dist - old_dist

    #     reward_val = self.reward_shaping_params[agent_id]["CENTERED_FOOD_DISTANCE_REW"]
    #     reward = reward_val * jnp.tanh(-dist_change.astype(jnp.float32))

    #     valid_mask = (
    #         jnp.all(jnp.isfinite(old_pos)) &
    #         jnp.all(jnp.isfinite(new_pos)) &
    #         jnp.all(jnp.isfinite(teammate_pos)) &
    #         jnp.all(jnp.isfinite(target_food_pos)) &
    #         has_food
    #     )

    #     reward = jnp.where(valid_mask, reward, 0.0)
    #     return reward

    def _calculate_centered_food_reward(self, prev_state, new_state, agent_id: str, i: int, target_food_idx):
        old_pos = prev_state.agents.position[i]
        new_pos = new_state.agents.position[i]
        teammate_pos = prev_state.agents.position[1 - i]
        midpoint = (old_pos + teammate_pos) / 2.0

        food_pos = prev_state.food_items.position
        food_eaten = prev_state.food_items.eaten
        uneaten_mask = ~food_eaten
        has_food = jnp.any(uneaten_mask)

        current_eaten = jnp.where(target_food_idx >= 0, food_eaten[target_food_idx], True)
        needs_update  = ((target_food_idx == -1) & has_food) | current_eaten

        midpoint_dists = jnp.sum(jnp.abs(food_pos - midpoint), axis=1)
        new_target_idx = jnp.argmin(jnp.where(uneaten_mask, midpoint_dists, jnp.inf))

        final_target_idx = jnp.where(needs_update, new_target_idx, target_food_idx)

        target_food_pos = food_pos[final_target_idx]
        old_dist = jnp.sum(jnp.abs(old_pos - target_food_pos))
        new_dist = jnp.sum(jnp.abs(new_pos - target_food_pos))
        dist_change = new_dist - old_dist

        reward_val = self.reward_shaping_params[agent_id]["CENTERED_FOOD_DISTANCE_REW"]
        reward = reward_val * jnp.tanh(-dist_change.astype(jnp.float32))

        valid_mask = (
            has_food
            & jnp.isfinite(old_dist)
            & jnp.isfinite(new_dist)
            & jnp.all(jnp.isfinite(teammate_pos))
        )

        reward = jnp.where(valid_mask, reward, 0.0)
        return reward, final_target_idx


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