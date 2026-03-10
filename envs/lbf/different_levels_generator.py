"""Generator that produces LBF grids with varied food levels.

When using the default RandomGenerator with force_coop=True, all food items
get the same level (sum of agent levels), so every fruit requires cooperation.
This generator instead ensures variety:
  - At least one fruit is soloable by each agent (level = that agent's level)
  - Remaining fruits require cooperation (level = combined agent levels)
  - The ordering is shuffled randomly each reset
"""

import jax
import jax.numpy as jnp
import chex

from jumanji.environments.routing.lbf.generator import RandomGenerator
from jumanji.environments.routing.lbf.types import State


class DifferentLevelsGenerator(RandomGenerator):
    """Wraps RandomGenerator and overrides food levels to guarantee variety."""

    def __call__(self, key: chex.PRNGKey) -> State:
        key, shuffle_key = jax.random.split(key)
        state = super().__call__(key)

        agent_levels = state.agents.level          # (num_agents,)
        combined = jnp.sum(agent_levels)

        # Build new food levels:
        #   - first num_agents entries: each soloable by one agent
        #   - remaining entries: require cooperation (level = combined)
        solo_levels = agent_levels[: self.num_food]  # handles num_food < num_agents edge case
        coop_levels = jnp.full(
            (max(0, self.num_food - self.num_agents),), combined
        )
        new_levels = jnp.concatenate([solo_levels, coop_levels])

        # Shuffle so the varied levels aren't in a predictable order
        perm = jax.random.permutation(shuffle_key, self.num_food)
        new_levels = new_levels[perm]

        # Replace in state (chex dataclass supports .replace)
        new_food = state.food_items.replace(level=new_levels)
        return state.replace(food_items=new_food)