from jaxmarl.environments.coin_game.coin_game import CoinGame
import jax
import jax.numpy as jnp
from functools import partial

class CoinGameWrapper(CoinGame):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: WrappedEnvState) -> Dict[str, jnp.ndarray]:
        """Returns the available actions for each agent."""
        num_actions = len(self.env.action_set)
        return {agent: jnp.ones(num_actions) for agent in self.agents}

    def observation_space(self, agent: str):
        """
        Returns the observation space for the given agent.
        If the observation space has an attribute 'n', it will be converted to a shape of (n,).
        """
        space = super().observation_space(agent)
        if hasattr(space, "n"):
            space.shape = (space.n,)
        return space

    def action_space(self, agent: str):
        """
        Returns the action space for the given agent.
        If the action space has an attribute 'n', it will be converted to a shape of (n,).
        """
        space = super().action_space(agent)
        if hasattr(space, "n"):
            space.shape = (space.n,)
        return space

    def __getattr__(self, name):
        return getattr(super(), name)
