from scipy.ndimage import label
import jax
import jax.numpy as jnp
import numpy as np
from jaxmarl.environments.overcooked.layouts import overcooked_layouts as layouts

def test_conn_comps():
    # layout = layouts["coord_ring"]
    layout = layouts["asymm_advantages"]

    h = layout["height"]
    w = layout["width"]
    all_pos = np.arange(np.prod([h, w]), dtype=jnp.uint32)

    wall_idx = layout.get("wall_idx")
    occupied_mask = jnp.zeros_like(all_pos)
    occupied_mask = occupied_mask.at[wall_idx].set(1)
    wall_map = occupied_mask.reshape(h, w).astype(jnp.bool_)
    free_space = ~wall_map

    print("Free space:", free_space)

    labels, num_features = label(free_space)
    print("Labels:", jnp.array(labels))
    print("Number of features:", jnp.array(num_features))

if __name__ == "__main__":
    test_conn_comps()
