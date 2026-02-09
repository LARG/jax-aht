import flax.linen as nn
import jax.numpy as jnp

from flax.core import FrozenDict
from flax.training.train_state import TrainState


class DQNTrainState(TrainState):
    target_network_params: FrozenDict
    timesteps: int
    n_updates: int

class QNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x
