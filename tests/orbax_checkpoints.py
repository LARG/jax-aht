import os
import shutil

import numpy as np
from flax import linen as nn
from flax.training import train_state
import jax
from jax import random, numpy as jnp
import optax
import orbax.checkpoint
from flax.training import orbax_utils


print("Starting the example...")


ckpt_dir = '/scratch/cluster/clw4542/explore_marl/continual-aht/results/orbax_example/'
print(f"Checkpoint directory: {ckpt_dir}")

if os.path.exists(ckpt_dir):
    shutil.rmtree(ckpt_dir)  # Remove any existing checkpoints from the last notebook run.
os.makedirs(ckpt_dir, exist_ok=True)

#################### CREATE STATE
# A simple model with one linear layer.
key1, key2 = random.split(random.key(0))
x1 = random.normal(key1, (5,))      # A simple JAX array.
model = nn.Dense(features=3)
variables = model.init(key2, x1)

# Flax's TrainState is a pytree dataclass and is supported in checkpointing.
# Define your class with `@flax.struct.dataclass` decorator to make it compatible.
tx = optax.sgd(learning_rate=0.001)      # An Optax SGD optimizer.
state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=variables['params'],
    tx=tx)
# Perform a simple gradient update similar to the one during a normal training workflow.
state = state.apply_gradients(grads=jax.tree_util.tree_map(jnp.ones_like, state.params))

# Some arbitrary nested pytree with a dictionary and a NumPy array.
config = {'dimensions': np.array([5, 3])}

# Bundle everything together.
ckpt = {'model': state, 'config': config, 'data': [x1]}

########### SAVE CHECKPOINTS
orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
save_args = orbax_utils.save_args_from_target(ckpt)
orbax_checkpointer.save(ckpt_dir + "/1/", ckpt, save_args=save_args)