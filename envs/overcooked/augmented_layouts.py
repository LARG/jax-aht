import numpy as np
from scipy.ndimage import label

from flax.core.frozen_dict import FrozenDict
import jax.numpy as jnp
from jaxmarl.environments.overcooked.layouts import overcooked_layouts

def get_augmented_layouts():
    '''Computes the number of connected components for each layout.
    This information is precomputed and stored in the layouts dictionary to
    allow placing agents in separate components at initialization.
    '''
    augmented_layouts = {}

    for layout_name, layout in overcooked_layouts.items():
        h = layout["height"]
        w = layout["width"]
        all_pos = np.arange(np.prod([h, w]), dtype=jnp.uint32)

        wall_idx = layout.get("wall_idx")
        occupied_mask = jnp.zeros_like(all_pos)
        occupied_mask = occupied_mask.at[wall_idx].set(1)
        wall_map = occupied_mask.reshape(h, w).astype(jnp.bool_)
        free_space_map = ~wall_map

        labelled_free_space, num_components = label(free_space_map)

        # construct augmented layout
        aug_layout = {}
        for k, v in layout.items():
            aug_layout[k] = v

        aug_layout["wall_map"] = wall_map
        aug_layout["free_space_map"] = jnp.array(labelled_free_space)
        aug_layout["num_components"] = num_components

        augmented_layouts[layout_name] = FrozenDict(aug_layout)
    return augmented_layouts

augmented_layouts = get_augmented_layouts()
