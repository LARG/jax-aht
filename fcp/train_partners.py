import time
import logging

import jax

from ppo.ippo import make_train

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def train_partners_in_parallel(config, base_seed):
    '''
    Train a pool of partners for FCP. Return checkpoints for all partners.
    Returns out, a dictionary of the final train_state, metrics, and checkpoints.
    '''
    start_time = time.time()
    rng = jax.random.PRNGKey(base_seed)
    rngs = jax.random.split(rng, config["NUM_SEEDS"])

    debug_mode = False
    with jax.disable_jit(debug_mode):
        if debug_mode: 
            out = make_train(config)(rngs)
        else:
            train_jit = jax.jit(jax.vmap(make_train(config)))
            out = train_jit(rngs)
    end_time = time.time()
    log.info(f"Training partners took {end_time - start_time:.2f} seconds.")
    return out
