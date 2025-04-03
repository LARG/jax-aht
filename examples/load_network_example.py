import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import distrax
from typing import Sequence
import numpy as np

import jumanji
from envs.jumanji_jaxmarl_wrapper import JumanjiToJaxMARL

import pickle

class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)

def select_checkpoint_params(full_checkpoints, seed_idx, ckpt_idx):
    """
    Slices the pytree so each leaf contains only the parameters for
    the given seed and checkpoint.
    """
    # 'full_checkpoints' is the entire dict with top-level keys like 'params', etc.
    # We'll map over the 'params' subtree (or whichever subtree you want).
    return jax.tree_map(lambda x: x[seed_idx, ckpt_idx], full_checkpoints['params'])

if __name__ == "__main__":
    ckpt_path = "results/lbf/2025-01-23_20-35-18/checkpoint.pkl"
    with open(ckpt_path, "rb") as f:
        checkpoints = pickle.load(f)

    # Checkpoints is a pytree where each leaf contains N seeds and M checkpoints 
    # worth  of parameters for a particular part of the network, 
    # e.g. checkpoints['params']['Dense_0']['kernel'] has 
    # shape (n_seeds, m_checkpoints, *weight_matrix_dims). 

    # Given seed n and checkpoint m, we want to extract the parameters 
    # and apply them to a new model instance.
    n, m = 0, 0  # which seed and which checkpoint you want
    params_for_n_m = select_checkpoint_params(checkpoints, n, m)
    
    # Initialize env
    env = jumanji.make("LevelBasedForaging-v0")
    env = JumanjiToJaxMARL(env)

    # Create a new model instance and apply the sliced parameters.
    model = ActorCritic(env.action_space(env.agents[0]).n)
    dummy_input = jnp.zeros((1, env.observation_space(env.agents[0]).shape[0]))

    # You can directly call 'apply' on the new model with your selected parameters.
    pi, value = model.apply({'params': params_for_n_m}, dummy_input)
    print("Action distribution:", pi)
    print("Critic value:", value)