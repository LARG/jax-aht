from typing import NamedTuple

import jax.numpy as jnp

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray
    partner_obs: jnp.ndarray
    partner_action: jnp.ndarray
