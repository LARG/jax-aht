"""Env wrapper that applies a sampled symmetry to one agent's view.

Other-Play-style augmentation (Hu et al. 2020). One symmetry per episode,
persisted across all steps, so the policy sees a consistent permuted view.
On reset we sample a fresh one; on step we carry it forward, resampling
only on episode end (the env has already auto-reset by then).
"""
from functools import partial
from typing import Tuple, Union

import chex
import jax
import jax.numpy as jnp
from flax import struct
from jaxmarl.environments.multi_agent_env import MultiAgentEnv, State
from jaxmarl.wrappers.baselines import JaxMARLWrapper

from envs.common.symmetry import EnvSymmetry


@struct.dataclass
class SymmetryWrappedState:
    env_state: State
    sym: jnp.ndarray  # the currently-applied symmetry (permutation, reflection, etc.)


class SymmetryAugmentationWrapper(JaxMARLWrapper):
    """Wraps an env so one agent sees a permuted view.

    Applies `symmetry` to `env.agents[agent_idx_to_augment]`'s obs and
    avail_actions, and inverse-applies it to that agent's actions before
    stepping. Other agents see the env's native obs.
    """

    def __init__(self, env: MultiAgentEnv, symmetry: EnvSymmetry,
                 agent_idx_to_augment: int = 1):
        super().__init__(env)
        self.symmetry = symmetry
        self.agent_idx_to_augment = agent_idx_to_augment
        self.augmented_agent_name = env.agents[agent_idx_to_augment]

    def get_avail_actions(self, state):
        """Return avail_actions with the augmented agent's mask permuted."""
        if isinstance(state, SymmetryWrappedState):
            avail = self._env.get_avail_actions(state.env_state)
            sym = state.sym
        else:
            avail = self._env.get_avail_actions(state)
            return avail

        permuted = {**avail}
        permuted[self.augmented_agent_name] = self.symmetry.apply_to_avail_actions(
            avail[self.augmented_agent_name], sym
        )
        return permuted

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey):
        key, sym_key = jax.random.split(key)
        sym = self.symmetry.sample(sym_key)
        obs, env_state = self._env.reset(key)
        obs = {**obs}
        obs[self.augmented_agent_name] = self.symmetry.apply_to_obs(
            obs[self.augmented_agent_name], sym
        )
        return obs, SymmetryWrappedState(env_state=env_state, sym=sym)

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: SymmetryWrappedState,
        action: dict,
    ):
        # augmented agent picked its action in the permuted view, so
        # undo the permutation before stepping
        native_action = {**action}
        native_action[self.augmented_agent_name] = self.symmetry.apply_to_action(
            action[self.augmented_agent_name], state.sym
        )

        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, native_action
        )

        # one symmetry per episode. resample only on episode end since
        # the env already auto-reset for us
        key, sym_key = jax.random.split(key)
        resampled = self.symmetry.sample(sym_key)
        new_sym = jax.tree.map(
            lambda old, new: jnp.where(done["__all__"], new, old),
            state.sym, resampled,
        )

        obs = {**obs}
        obs[self.augmented_agent_name] = self.symmetry.apply_to_obs(
            obs[self.augmented_agent_name], new_sym
        )
        return obs, SymmetryWrappedState(env_state=env_state, sym=new_sym), \
               reward, done, info
