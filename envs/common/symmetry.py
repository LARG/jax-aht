"""Abstract base for env-level symmetries used by augmentation training.

A symmetry is an equivariance the policy can be augmented over: a permutation
of the env's color/agent/spatial coordinates that leaves task semantics
unchanged. SymmetryAugmentationWrapper samples one and applies it to one
agent's view + actions, so the policy learns to be robust to it.

Examples:
  HanabiColorSymmetry: Hu et al. 2020 Other-Play (color permutation)
  OvercookedReflection: left-right reflection of layout coordinates
  LBFAgentPermutation: agent-id permutation in the obs
"""
from abc import ABC, abstractmethod
import chex
import jax.numpy as jnp


class EnvSymmetry(ABC):
    """An equivariance the env policy can be augmented over."""

    @abstractmethod
    def sample(self, rng: chex.PRNGKey):
        """Sample a single group element from this symmetry."""

    @abstractmethod
    def apply_to_obs(self, obs: jnp.ndarray, sym) -> jnp.ndarray:
        """Apply the symmetry to a single agent's flat observation."""

    @abstractmethod
    def apply_to_action(self, action: jnp.ndarray, sym) -> jnp.ndarray:
        """Inverse-apply the symmetry: convert an action chosen in the
        permuted view back into the env's native action space."""

    @abstractmethod
    def apply_to_avail_actions(self, avail: jnp.ndarray, sym) -> jnp.ndarray:
        """Apply the symmetry to a single agent's avail-actions mask."""
