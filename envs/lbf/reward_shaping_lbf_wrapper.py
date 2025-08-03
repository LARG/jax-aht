import jax
import jax.numpy as jnp
from envs.lbf.lbf_wrapper import LBFWrapper
from jumanji.environments.routing.lbf.constants import LOAD
from typing import Dict, Any
from flax.struct import dataclass
from functools import partial

@dataclass
class RewardShapingEnvState:
    env_state: Any # a jumanji state
    prev_env_state: Any # a jumanji state
    target_food_idx: jnp.ndarray #  [0] = nearest, [1] = farthest, [2] = centered
    avail_actions: jnp.ndarray
    step: jnp.array

REWARD_SHAPING_PARAMS = {
    "agent_0": {
    "DISTANCE_TO_NEAREST_FOOD_REW": -1.0, # Reward for moving closer to food (H1)
    "DISTANCE_TO_FARTHEST_FOOD_REW": 1.0, # Reward for moving further from food (H2)
    "FOLLOWING_TEAMMATE_REW": 0.0, # Reward for following another agent (H9)
    "CENTERED_FOOD_DISTANCE_REW": 0.0, # Reward for moving towards towards the food that is closest to the midpoint of the two agents
    "PROXIMITY_TO_TEAMMATE_REW": 0.0, # Reward for proximity to teammate
    "COLLECT_FOOD_REW": 0.5, # Reward for collecting food
    },
    "agent_1": {
    "DISTANCE_TO_NEAREST_FOOD_REW": 0.0, # Reward for moving closer to food (H1)
    "DISTANCE_TO_FARTHEST_FOOD_REW": 0.0, # Reward for moving further from food (H2)
    "FOLLOWING_TEAMMATE_REW": 1.0, # Reward for following another agent (H9)
    "CENTERED_FOOD_DISTANCE_REW": 0.0, # Reward for moving towards towards the food that is closest to the midpoint of the two agents
    "PROXIMITY_TO_TEAMMATE_REW": 0.0, # Reward for proximity to teammate
    "COLLECT_FOOD_REW": 0.5, # Reward for collecting food
    },
    "REWARD_SHAPING_COEF": 0.1,
}    

class RewardShapingLBFWrapper(LBFWrapper):
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
    - H9. Agents under H9 will follow their teammate.
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

        (obs, state) = jax.tree_util.tree_map(
            lambda reset_val, next_val: jax.lax.select(done["__all__"], reset_val, next_val),
            (reset_obs, reset_state),
            (next_obs, next_state),
        )

        # add the original reward to the info dict
        original_reward = jnp.array([reward[agent] for agent in self.agents])
        # create a new info dictionary with all the keys of the original info dictionary plus the new original_reward key
        new_info = {**info, "original_reward": original_reward}
        return obs, state, total_reward_dict, done, new_info

    def _extract_shaped_rewards(self, prev_env_state, env_state, actions, target_food_idx):
        shaped_rewards = {}

        # Compute shaped rewards for each agent
        for agent_index, agent_id in enumerate(self.agents):
            total_shaped_reward = 0.0
            current_targets = target_food_idx[agent_index]
            nearest = current_targets[0]
            farthest = current_targets[1]
            centered = current_targets[2]

            nearest_rew = self._calculate_distance_to_nearest_food_reward(
                prev_env_state, env_state, agent_id, agent_index, nearest, farthest, centered
            )
            total_shaped_reward += nearest_rew

            farthest_rew = self._calculate_distance_to_farthest_food_reward(
                prev_env_state, env_state, agent_id, agent_index, farthest, nearest, centered
            )
            total_shaped_reward += farthest_rew

            centered_rew = self._calculate_centered_food_reward(
                prev_env_state, env_state, agent_id, agent_index, centered
            )
            total_shaped_reward += centered_rew

            total_shaped_reward += self._calculate_following_teammate_reward(
                prev_env_state, env_state, agent_id, agent_index
            )

            total_shaped_reward += self._calculate_proximity_to_teammate_reward(
                prev_env_state, env_state, agent_id, agent_index
            )

            total_shaped_reward += self._calculate_collect_food_reward(
                prev_env_state, env_state, agent_id
            )

            shaped_rewards[agent_id] = total_shaped_reward

        prev_eaten = prev_env_state.food_items.eaten
        curr_eaten = env_state.food_items.eaten
        newly_eaten = jnp.logical_and(curr_eaten, jnp.logical_not(prev_eaten))
        any_newly_eaten = jnp.any(newly_eaten)

        # Compute new targets if any food was eaten
        new_initial_targets = self._compute_initial_targets(env_state)

        updated_target_food_idx = jax.lax.select(
            any_newly_eaten,
            new_initial_targets,
            target_food_idx
        )

        return shaped_rewards, updated_target_food_idx

    def _calculate_distance_to_nearest_food_reward(self, prev_state, new_state, agent_id: str, i: int, target_food_idx, farthest_food_idx, centered_food_idx):
        """
        Rewards agents for moving towards the food that is closest to their current position.
        If the target food is eaten, the target is updated.
        The reward is based on the change in distance to the target food.
        """
        # Agent and teammate positions
        old_pos = prev_state.agents.position[i]
        new_pos = new_state.agents.position[i]
        food_pos = prev_state.food_items.position
        food_eaten = new_state.food_items.eaten
        uneaten_mask = ~food_eaten

        # Compute Manhattan distance to current target food
        target_food_pos = food_pos[target_food_idx]
        old_dist = jnp.sum(jnp.abs(old_pos - target_food_pos))
        new_dist = jnp.sum(jnp.abs(new_pos - target_food_pos))
        dist_change = new_dist - old_dist

        reward_val = self.reward_shaping_params[agent_id]["DISTANCE_TO_NEAREST_FOOD_REW"]
        raw_reward = reward_val * jnp.tanh(-dist_change.astype(jnp.float32))

        # Skip reward if nearest == farthest or only one food left or nearest == centered
        skip_reward = (
            jnp.equal(target_food_idx, farthest_food_idx) |
            jnp.equal(target_food_idx, centered_food_idx) |
            (jnp.sum(uneaten_mask) == 1)
        )
        reward = jnp.where(skip_reward, 0.0, raw_reward)

        valid_mask = jnp.isfinite(old_dist) & jnp.isfinite(new_dist)
        reward = jnp.where(valid_mask, reward, 0.0)
        return reward

    def _calculate_distance_to_farthest_food_reward(self, prev_state, new_state, agent_id: str, i: int, target_food_idx, nearest_food_idx, centered_food_idx):
        """
        Rewards agents for moving towards the food that is furthest from their current position.
        If the target food is eaten, the target is updated.
        The reward is based on the change in distance to the target food.
        """
        # Agent and teammate positions
        old_pos = prev_state.agents.position[i]
        new_pos = new_state.agents.position[i]
        food_pos = prev_state.food_items.position
        food_eaten = new_state.food_items.eaten
        uneaten_mask = ~food_eaten

        # Compute Manhattan distance to current target food
        target_food_pos = food_pos[target_food_idx]
        old_dist = jnp.sum(jnp.abs(old_pos - target_food_pos))
        new_dist = jnp.sum(jnp.abs(new_pos - target_food_pos))
        dist_change = new_dist - old_dist

        reward_val = self.reward_shaping_params[agent_id]["DISTANCE_TO_FARTHEST_FOOD_REW"]
        raw_reward = reward_val * jnp.tanh(-dist_change.astype(jnp.float32))

        # Skip reward if farthest == nearest or only one food left or farthest == centered
        skip_reward = (
            jnp.equal(target_food_idx, nearest_food_idx) |
            jnp.equal(target_food_idx, centered_food_idx) |
            (jnp.sum(uneaten_mask) == 1)
        )
        reward = jnp.where(skip_reward, 0.0, raw_reward)

        valid_mask = jnp.isfinite(old_dist) & jnp.isfinite(new_dist)
        reward = jnp.where(valid_mask, reward, 0.0)
        return reward
    
    def _calculate_following_teammate_reward(self, prev_state, new_state, agent_id: str, i: int):
        """
        Rewards agents for following their teammate.
        The reward is based on the change in distance between the agent and its teammate.
        If the agent is closer to its teammate than before, it receives a positive reward.
        """
        # Agent and teammate positions
        old_pos = prev_state.agents.position[i]
        new_pos = new_state.agents.position[i]

        old_teammate_pos = prev_state.agents.position[1 - i]
        new_teammate_pos = new_state.agents.position[1 - i]

        # Calculate Manhattan distance between agent and teammate for both old and new positions
        old_dist = jnp.sum(jnp.abs(old_pos - old_teammate_pos))
        new_dist = jnp.sum(jnp.abs(new_pos - new_teammate_pos))
        dist_change = new_dist - old_dist 

        reward_val = self.reward_shaping_params[agent_id]["FOLLOWING_TEAMMATE_REW"]

        reward = reward_val * jnp.tanh(-dist_change.astype(jnp.float32))
        reward = jnp.where(jnp.isfinite(reward), reward, 0.0)

        # Ensure all positions are finite
        valid_mask = (
            jnp.all(jnp.isfinite(old_pos)) &
            jnp.all(jnp.isfinite(new_pos)) &
            jnp.all(jnp.isfinite(old_teammate_pos)) &
            jnp.all(jnp.isfinite(new_teammate_pos))
        )
        reward = jnp.where(valid_mask, reward, 0.0)
        return reward

    def _calculate_centered_food_reward(self, prev_state, new_state, agent_id: str, i: int, target_food_idx):
        """
        Rewards the agent for being close to the centered food target (based on midpoint of both agents).
        """
        # Agent and teammate positions
        agent_pos = new_state.agents.position[i]
        teammate_pos = new_state.agents.position[1 - i]
        food_pos = prev_state.food_items.position 
        target_food_pos = food_pos[target_food_idx]

        # Manhattan distance from agent to target food
        dist = jnp.sum(jnp.abs(agent_pos - target_food_pos))

        # Reward is based on the distance to the centered food target which is closest to the midpoint of both agents (the closer the agent is the higher the reward)
        reward_val = self.reward_shaping_params[agent_id]["CENTERED_FOOD_DISTANCE_REW"]
        reward = reward_val * jnp.exp(-dist.astype(jnp.float32)) 

        # Only skip reward if inputs are invalid
        valid_mask = (
            jnp.isfinite(dist)
            & jnp.all(jnp.isfinite(agent_pos))
            & jnp.all(jnp.isfinite(teammate_pos))
        )
        reward = jnp.where(valid_mask, reward, 0.0)
        return reward

    def _calculate_proximity_to_teammate_reward(self, prev_state, new_state, agent_id: str, i: int):
        """
        Rewards agents for being close to their teammate.
        The reward is based on the distance between the agent and its teammate.
        The closer they are, the higher the reward.
        If the agent is not close to its teammate, the reward is 0.
        """
        # Agent and teammate positions
        new_pos = new_state.agents.position[i]
        teammate_pos = new_state.agents.position[1 - i]

        new_dist = jnp.sum(jnp.abs(new_pos - teammate_pos))
         
        # Calculate the reward based on the distance between the agent and its teammate (the closer the agent is the higher the reward)
        reward_val = self.reward_shaping_params[agent_id]["PROXIMITY_TO_TEAMMATE_REW"]
        reward = reward_val * jnp.tanh(-new_dist.astype(jnp.float32))

        valid_mask = jnp.all(jnp.isfinite(new_pos)) & jnp.all(jnp.isfinite(teammate_pos))
        reward = jnp.where(valid_mask, reward, 0.0)
        return reward

    def _calculate_collect_food_reward(self, prev_state, new_state, agent_id: str) -> float:
        """
        Rewards agents for collecting food.
        The reward is given if the agent has eaten a food item that was not eaten in the previous state.
        """
        prev_eaten = prev_state.food_items.eaten
        curr_eaten = new_state.food_items.eaten
        newly_eaten = jnp.logical_and(curr_eaten, jnp.logical_not(prev_eaten))
        any_newly_eaten = jnp.any(newly_eaten)

        reward = self.reward_shaping_params[agent_id]["COLLECT_FOOD_REW"]
        return jnp.where(any_newly_eaten, reward, 0.0)