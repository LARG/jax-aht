import numpy as np
from typing import Tuple, Dict
from functools import partial

import chex
from flax.core.frozen_dict import FrozenDict
import jax
from jax import lax
import jax.numpy as jnp
from jaxmarl.environments.overcooked.overcooked import Overcooked, State as OvercookedState
from jaxmarl.environments.overcooked.common import (
    OBJECT_TO_INDEX,
    DIR_TO_VEC,
    OBJECT_INDEX_TO_VEC,
    make_overcooked_map)

from envs.overcooked.augmented_layouts import augmented_layouts as layouts

BASE_REW_SHAPING_PARAMS = {
    "PLACEMENT_IN_POT_REW": 3, # reward for putting ingredients 
    "PLATE_PICKUP_REWARD": 3, # reward for picking up a plate
    "SOUP_PICKUP_REWARD": 5, # reward for picking up a ready soup
    "ONION_PICKUP_REWARD": 0,
    "COUNTER_PICKUP_REWARD": 0,
    "COUNTER_DROP_REWARD": 0,
    "DISH_DISP_DISTANCE_REW": 0,
    "POT_DISTANCE_REW": 0,
    "SOUP_DISTANCE_REW": 0,
}

# Pot status indicated by an integer, which ranges from 23 to 0
POT_EMPTY_STATUS = 23 # 22 = 1 onion in pot; 21 = 2 onions in pot; 20 = 3 onions in pot
POT_FULL_STATUS = 20 # 3 onions. Below this status, pot is cooking, and status acts like a countdown timer.
POT_READY_STATUS = 0
MAX_ONIONS_IN_POT = 3 # A pot has at most 3 onions. A soup contains exactly 3 onions.

URGENCY_CUTOFF = 40 # When this many time steps remain, the urgency layer is flipped on
DELIVERY_REWARD = 20

class OvercookedV1(Overcooked):
    '''This environment is a modified version of the JaxMARL Overcooked environment 
    that ensures environments are solvable. In addition, this environment adds additional
    reward shaping parameters.
    
    The main modifications are: 
    - Random resets: Previously, setting `random_reset` would lead to 
        random initial agent positions, and randomized initial object states (e.g. pot might be initialized with onions already in it, agents might be initialized holding plates, etc.). We separate the functionality of the argument `random_reset` into two arguments: `random_reset` and `random_obj_state`, 
        where `random_reset` only controls the initial positions of the two agents. 
    - Initial agent positions: Previously, agent positions were initialized by choosing randomly from any free space on 
        the map, which could lead to the two agents being on the same side of a disconnected map. 
        Now, we ensure that the two agents are always initialized in separate components of the map
        (if there are at least two components). 
    - More reward shaping: Previously, reward could only be shaped with plate pickup, pot placement, and soup pickup.
        Now, we add dense rewards for onion pickups from piles, picking up from counter, and dropping at a counter.
    '''
    def __init__(self, 
            layout = FrozenDict(layouts["cramped_room"]),
            random_reset: bool = False,
            random_obj_state: bool = False, 
            max_steps: int = 400,
            do_reward_shaping: bool = False,
            reward_shaping_params = FrozenDict({}),
    ):
        super().__init__(layout=layout, 
                         random_reset=random_reset, 
                         max_steps=max_steps)
        self.do_reward_shaping = do_reward_shaping
        def merge_params(base_dict, override_dict):
            merged = {k: override_dict.get(k, v) for k, v in base_dict.items()}
            return FrozenDict(merged)
        self.reward_shaping_params = merge_params(BASE_REW_SHAPING_PARAMS, reward_shaping_params)
        self.random_obj_state = random_obj_state # controls whether pot state and inventory are randomized
    
    def _initialize_agent_positions(self, key: chex.PRNGKey, all_pos: jnp.ndarray, num_agents: int) -> Tuple[chex.PRNGKey, jnp.ndarray]:
        """Initialize agent positions ensuring they are on separate halves of the map if possible.
        Function assumes there are two agents.

        Args:
            key: JAX PRNG key
            all_pos: Array of all possible positions
            num_agents: Number of agents to initialize
        Returns:
            Tuple of (new_key, agent_idx) where agent_idx contains the initialized agent positions
        """
        free_space_map = self.layout["free_space_map"]
        wall_map = self.layout["wall_map"]
        num_components = self.layout["num_components"]

        if self.random_reset and num_components >= 2:
            # If we have at least 2 components, ensure agents are in different components
            key, subkey = jax.random.split(key)
            # Randomly choose num_agents different components
            component_indices = jax.random.choice(subkey, jnp.arange(1, num_components + 1), 
                shape=(num_agents,), replace=False)
            
            # Randomly sample each agent's position from each component
            # Note that the free_space_map is an h x w array where each connected components 
            # is labelled by a unique integer counting up from 1. 
            # Example: 
            # free_space_map = [[0 0 0 0 0 0 0 0 0]
            #                   [0 1 0 0 0 0 0 2 0]
            #                   [0 1 1 1 0 2 2 2 0]
            #                   [0 1 1 1 0 2 2 2 0]
            #                   [0 0 0 0 0 0 0 0 0]]
            # Here, there are two components: one with label 1 and one with label 2.
            # Each component has 6 positions.
            
            # For each agent, find positions in their assigned component and sample one
            agent_idx = jnp.zeros(num_agents, dtype=jnp.uint32)
            for i in range(num_agents):
                component_idx = component_indices[i]
                # Create a mask where 1 indicates positions in the desired component
                component_mask = (free_space_map.reshape(-1) == component_idx).astype(jnp.float32)
                # Randomly sample one position from this component
                key, subkey = jax.random.split(key)
                agent_idx = agent_idx.at[i].set(jax.random.choice(subkey, all_pos, p=component_mask))
        else:
            # Use default layout positions or random positions if only a single component
            key, subkey = jax.random.split(key)
            agent_idx = jax.random.choice(subkey, all_pos, shape=(num_agents,),
                                      p=(~wall_map.reshape(-1).astype(jnp.bool_)).astype(jnp.float32), 
                                      replace=False)
            agent_idx = self.random_reset*agent_idx + (1-self.random_reset)*self.layout.get("agent_idx", agent_idx)
            
        return key, agent_idx

    @partial(jax.jit, static_argnums=(0,))
    def step_env(self, key: jax.random.PRNGKey, state: OvercookedState, actions: Dict[str, jnp.ndarray]) -> tuple:
        """Override step_env to perform reward shaping and reshape the info dictionary."""
        obs, state, rewards, dones, info = super().step_env(key, state, actions)
        
        rewards_shaped = {"agent_0": rewards["agent_0"] + self.do_reward_shaping * info['shaped_reward']["agent_0"], 
                         "agent_1": rewards["agent_1"] + self.do_reward_shaping * info['shaped_reward']["agent_1"]}

        # Add shaped rewards to info dictionary
        shaped_rewards = jnp.array([info['shaped_reward'][agent] for agent in self.agents])
        base_reward = jnp.array([rewards[agent] for agent in self.agents])
        info = {'shaped_reward': shaped_rewards, "base_reward": base_reward}
        
        return obs, state, rewards_shaped, dones, info

    def process_interact(
            self,
            maze_map: chex.Array,
            wall_map: chex.Array,
            fwd_pos_all: chex.Array,
            inventory_all: chex.Array,
            player_idx: int):
        """Assume agent took interact actions. Result depends on what agent is facing and what it is holding."""
        
        fwd_pos = fwd_pos_all[player_idx]
        inventory = inventory_all[player_idx]

        shaped_reward = 0.

        height = self.obs_shape[1]
        padding = (maze_map.shape[0] - height) // 2

        # Get object in front of agent (on the "table")
        maze_object_on_table = maze_map.at[padding + fwd_pos[1], padding + fwd_pos[0]].get()
        object_on_table = maze_object_on_table[0]  # Simple index

        # Booleans depending on what the object is
        object_is_pile = jnp.logical_or(object_on_table == OBJECT_TO_INDEX["plate_pile"], object_on_table == OBJECT_TO_INDEX["onion_pile"])
        object_is_pot = jnp.array(object_on_table == OBJECT_TO_INDEX["pot"])
        object_is_goal = jnp.array(object_on_table == OBJECT_TO_INDEX["goal"])
        object_is_agent = jnp.array(object_on_table == OBJECT_TO_INDEX["agent"])
        object_is_pickable = jnp.logical_or(
            jnp.logical_or(object_on_table == OBJECT_TO_INDEX["plate"], object_on_table == OBJECT_TO_INDEX["onion"]),
            object_on_table == OBJECT_TO_INDEX["dish"]
        )
        # Whether the object in front is counter space that the agent can drop on.
        is_table = jnp.logical_and(wall_map.at[fwd_pos[1], fwd_pos[0]].get(), ~object_is_pot)

        table_is_empty = jnp.logical_or(object_on_table == OBJECT_TO_INDEX["wall"], object_on_table == OBJECT_TO_INDEX["empty"])

        # Pot status (used if the object is a pot)
        pot_status = maze_object_on_table[-1]

        # Get inventory object, and related booleans
        inv_is_empty = jnp.array(inventory == OBJECT_TO_INDEX["empty"])
        object_in_inv = inventory
        holding_onion = jnp.array(object_in_inv == OBJECT_TO_INDEX["onion"])
        holding_plate = jnp.array(object_in_inv == OBJECT_TO_INDEX["plate"])
        holding_dish = jnp.array(object_in_inv == OBJECT_TO_INDEX["dish"])

        # Interactions with pot. 3 cases: add onion if missing, collect soup if ready, do nothing otherwise
        case_1 = (pot_status > POT_FULL_STATUS) * holding_onion * object_is_pot
        case_2 = (pot_status == POT_READY_STATUS) * holding_plate * object_is_pot
        case_3 = (pot_status > POT_READY_STATUS) * (pot_status <= POT_FULL_STATUS) * object_is_pot
        else_case = ~case_1 * ~case_2 * ~case_3

        # give reward for placing onion in pot, and for picking up soup
        shaped_reward += case_1 * self.reward_shaping_params["PLACEMENT_IN_POT_REW"]
        shaped_reward += case_2 * self.reward_shaping_params["SOUP_PICKUP_REWARD"]

        # Update pot status and object in inventory
        new_pot_status = \
            case_1 * (pot_status - 1) \
            + case_2 * POT_EMPTY_STATUS \
            + case_3 * pot_status \
            + else_case * pot_status
        new_object_in_inv = \
            case_1 * OBJECT_TO_INDEX["empty"] \
            + case_2 * OBJECT_TO_INDEX["dish"] \
            + case_3 * object_in_inv \
            + else_case * object_in_inv

        # Interactions with onion/plate piles and objects on counter
        # Pickup if: table, not empty, room in inv & object is not something unpickable (e.g. pot or goal)
        successful_pickup = is_table * ~table_is_empty * inv_is_empty * jnp.logical_or(object_is_pile, object_is_pickable)
        successful_drop = is_table * table_is_empty * ~inv_is_empty
        successful_delivery = is_table * object_is_goal * holding_dish
        no_effect = jnp.logical_and(jnp.logical_and(~successful_pickup, ~successful_drop), ~successful_delivery)

        # give reward for picking up onion, placing on a counter, and removing from a counter
        shaped_reward += jnp.logical_and(successful_pickup, object_is_pile) * self.reward_shaping_params["ONION_PICKUP_REWARD"]
        shaped_reward += jnp.logical_and(successful_pickup, object_is_pickable) * self.reward_shaping_params["COUNTER_PICKUP_REWARD"]
        shaped_reward += successful_drop * self.reward_shaping_params["COUNTER_DROP_REWARD"]

        # Update object on table
        new_object_on_table = \
            no_effect * object_on_table \
            + successful_delivery * object_on_table \
            + successful_pickup * object_is_pile * object_on_table \
            + successful_pickup * object_is_pickable * OBJECT_TO_INDEX["wall"] \
            + successful_drop * object_in_inv

        # Update object in inventory
        new_object_in_inv = \
            no_effect * new_object_in_inv \
            + successful_delivery * OBJECT_TO_INDEX["empty"] \
            + successful_pickup * object_is_pickable * object_on_table \
            + successful_pickup * (object_on_table == OBJECT_TO_INDEX["plate_pile"]) * OBJECT_TO_INDEX["plate"] \
            + successful_pickup * (object_on_table == OBJECT_TO_INDEX["onion_pile"]) * OBJECT_TO_INDEX["onion"] \
            + successful_drop * OBJECT_TO_INDEX["empty"]

        # Apply inventory update
        has_picked_up_plate = successful_pickup*(new_object_in_inv == OBJECT_TO_INDEX["plate"])
        
        # number of plates in player hands < number ready/cooking/partially full pot
        num_plates_in_inv = jnp.sum(inventory == OBJECT_TO_INDEX["plate"])
        pot_loc_layer = jnp.array(maze_map[padding:-padding, padding:-padding, 0] == OBJECT_TO_INDEX["pot"], dtype=jnp.uint8)
        padded_map = maze_map[padding:-padding, padding:-padding, 2] 
        num_notempty_pots = jnp.sum((padded_map!=POT_EMPTY_STATUS)* pot_loc_layer)
        is_dish_picku_useful = num_plates_in_inv < num_notempty_pots

        plate_loc_layer = jnp.array(maze_map == OBJECT_TO_INDEX["plate"], dtype=jnp.uint8)
        no_plates_on_counters = jnp.sum(plate_loc_layer) == 0
        
        shaped_reward += no_plates_on_counters*has_picked_up_plate*is_dish_picku_useful*self.reward_shaping_params["PLATE_PICKUP_REWARD"]

        inventory = new_object_in_inv
        
        # Apply changes to maze
        new_maze_object_on_table = \
            object_is_pot * OBJECT_INDEX_TO_VEC[new_object_on_table].at[-1].set(new_pot_status) \
            + ~object_is_pot * ~object_is_agent * OBJECT_INDEX_TO_VEC[new_object_on_table] \
            + object_is_agent * maze_object_on_table

        maze_map = maze_map.at[padding + fwd_pos[1], padding + fwd_pos[0], :].set(new_maze_object_on_table)

        # Reward of 20 for a soup delivery
        reward = jnp.array(successful_delivery, dtype=float)*DELIVERY_REWARD
        return maze_map, inventory, reward, shaped_reward

    def reset(
            self,
            key: chex.PRNGKey,
    ) -> Tuple[Dict[str, chex.Array], OvercookedState]:
        """Reset environment state based on `self.random_reset` and `self.random_obj_state`

        If random_reset, agent initial positions are randomized.
        If random_obj_state, pot states and inventory are randomized.

        Environment layout is determined by `self.layout`
        """

        layout = self.layout
        h = self.height
        w = self.width
        num_agents = self.num_agents
        all_pos = np.arange(np.prod([h, w]), dtype=jnp.uint32)
        
        wall_map = layout.get("wall_map")
        wall_idx = layout.get("wall_idx")

        # Initialize agent positions
        key, agent_idx = self._initialize_agent_positions(key, all_pos, num_agents)
        agent_pos = jnp.array([agent_idx % w, agent_idx // w], dtype=jnp.uint32).transpose() # dim = n_agents x 2

        key, subkey = jax.random.split(key)
        agent_dir_idx = jax.random.choice(subkey, jnp.arange(len(DIR_TO_VEC), dtype=jnp.int32), shape=(num_agents,))
        agent_dir = DIR_TO_VEC.at[agent_dir_idx].get() # dim = n_agents x 2

        # Keep track of empty counter space (table)
        empty_table_mask = jnp.zeros_like(all_pos)
        empty_table_mask = empty_table_mask.at[wall_idx].set(1)

        goal_idx = layout.get("goal_idx")
        goal_pos = jnp.array([goal_idx % w, goal_idx // w], dtype=jnp.uint32).transpose()
        empty_table_mask = empty_table_mask.at[goal_idx].set(0)

        onion_pile_idx = layout.get("onion_pile_idx")
        onion_pile_pos = jnp.array([onion_pile_idx % w, onion_pile_idx // w], dtype=jnp.uint32).transpose()
        empty_table_mask = empty_table_mask.at[onion_pile_idx].set(0)

        plate_pile_idx = layout.get("plate_pile_idx")
        plate_pile_pos = jnp.array([plate_pile_idx % w, plate_pile_idx // w], dtype=jnp.uint32).transpose()
        empty_table_mask = empty_table_mask.at[plate_pile_idx].set(0)

        pot_idx = layout.get("pot_idx")
        pot_pos = jnp.array([pot_idx % w, pot_idx // w], dtype=jnp.uint32).transpose()
        empty_table_mask = empty_table_mask.at[pot_idx].set(0)

        key, subkey = jax.random.split(key)
        # Pot status is determined by a number between 0 (inclusive) and 24 (exclusive)
        # 23 corresponds to an empty pot (default)
        pot_status = jax.random.randint(subkey, (pot_idx.shape[0],), 0, 24)
        pot_status = pot_status * self.random_obj_state + (1-self.random_obj_state) * jnp.ones((pot_idx.shape[0])) * 23

        onion_pos = jnp.array([])
        plate_pos = jnp.array([])
        dish_pos = jnp.array([])

        maze_map = make_overcooked_map(
            wall_map,
            goal_pos,
            agent_pos,
            agent_dir_idx,
            plate_pile_pos,
            onion_pile_pos,
            pot_pos,
            pot_status,
            onion_pos,
            plate_pos,
            dish_pos,
            pad_obs=True,
            num_agents=self.num_agents,
            agent_view_size=self.agent_view_size
        )

        # agent inventory (empty by default, can be randomized)
        key, subkey = jax.random.split(key)
        possible_items = jnp.array([OBJECT_TO_INDEX['empty'], OBJECT_TO_INDEX['onion'],
                          OBJECT_TO_INDEX['plate'], OBJECT_TO_INDEX['dish']])
        random_agent_inv = jax.random.choice(subkey, possible_items, shape=(num_agents,), replace=True)
        agent_inv = self.random_obj_state * random_agent_inv + \
                    (1-self.random_obj_state) * jnp.array([OBJECT_TO_INDEX['empty'], OBJECT_TO_INDEX['empty']])

        state = OvercookedState(
            agent_pos=agent_pos,
            agent_dir=agent_dir,
            agent_dir_idx=agent_dir_idx,
            agent_inv=agent_inv,
            goal_pos=goal_pos,
            pot_pos=pot_pos,
            wall_map=wall_map.astype(jnp.bool_),
            maze_map=maze_map,
            time=0,
            terminal=False,
        )

        obs = self.get_obs(state)

        return lax.stop_gradient(obs), lax.stop_gradient(state)
