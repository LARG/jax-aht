from typing import NamedTuple

import jax.numpy as jnp


class Transition(NamedTuple):
    done: jnp.ndarray  # done at t+1 (post-step); used for GAE bootstrapping
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray
    prev_action_onehot: jnp.ndarray
    partner_obs: jnp.ndarray
    partner_action_onehot: jnp.ndarray
    prev_done: jnp.ndarray = None  # done at t (pre-step); RNN reset signal for replay
