import numpy as np
from typing import Tuple, Dict

import chex
from flax.core.frozen_dict import FrozenDict
import jax
from jax import lax
import jax.numpy as jnp
from jaxmarl.environments.overcooked.overcooked import Overcooked, State
from jaxmarl.environments.overcooked.common import (
    OBJECT_TO_INDEX,
    DIR_TO_VEC,
    make_overcooked_map)

from envs.overcooked.augmented_layouts import augmented_layouts as layouts


class OvercookedV2(Overcooked):
    '''This environment is a modified version of the JaxMARL Overcooked environment 
    that ensures environments are solvable. 
    
    The main modifications are: 
    - Random resets: Previously, setting `random_reset` would lead to 
        random initial agent positions, and randomized initial object states (e.g. pot might be initialized with onions already in it, agents might be initialized holding plates, etc.). We separate the functionality of the argument `random_reset` into two arguments: `random_reset` and `random_obj_state`, 
        where `random_reset` only controls the initial positions of the two agents. 
    - Initial agent positions: Previously, agent positions were initialized by choosing randomly from any free space on 
        the map, which could lead to the two agents being on the same side of a disconnected map. 
        Now, we ensure that the two agents are always initialized in separate components of the map
        (if there are at least two components). 
    '''
    def __init__(self, 
            layout = FrozenDict(layouts["cramped_room"]),
            random_reset: bool = False,
            random_obj_state: bool = False, 
            max_steps: int = 400,
    ):
        super().__init__(layout=layout, 
                         random_reset=random_reset, 
                         max_steps=max_steps)
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

    def reset(
            self,
            key: chex.PRNGKey,
    ) -> Tuple[Dict[str, chex.Array], State]:
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

        state = State(
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
