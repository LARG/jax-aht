from jaxmarl.environments.coin_game.coin_game import CoinGame
import jax
import jax.numpy as jnp
from functools import partial
from flax.struct import dataclass
from typing import Dict, Any

@dataclass
class WrappedEnvState:
    env_state: Any
    base_return_so_far: jnp.ndarray

class CoinGameWrapper:
    def __init__(self, **kwargs):
        self.env = CoinGame(**kwargs)
        self.agents = self.env.agents

    def observation_space(self, agent: str):
        """
        Returns the observation space for the given agent.
        If the observation space has an attribute 'n', it will be converted to a shape of (n,).
        """
        space = self.env.observation_space(agent)
        if hasattr(space, "n"):
            space.shape = (space.n,)
        return space

    def action_space(self, agent: str):
        """
        Returns the action space for the given agent.
        If the action space has an attribute 'n', it will be converted to a shape of (n,).
        """
        space = self.env.action_space(agent)
        if hasattr(space, "n"):
            space.shape = (space.n,)
        return space

    def get_avail_actions(self, state: WrappedEnvState) -> Dict[str, jnp.ndarray]:
        """Returns the available actions for each agent."""
        num_actions = self.env.num_actions
        return {agent: jnp.ones(num_actions) for agent in self.agents}

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key):
        obs, env_state = self.env.reset(key)
        # obs is already flattened if not CNN, otherwise flatten here if needed
        flat_obs = {agent: obs[agent].flatten() for agent in self.agents}
        return flat_obs, WrappedEnvState(env_state, jnp.zeros(len(self.agents)))

    @jax.jit
    def step(self, key, state: WrappedEnvState, actions: Dict[str, jnp.ndarray]):
        obs, env_state, rewards, dones, infos = self.env.step(key, state.env_state, actions)
        flat_obs = {agent: obs[agent].flatten() for agent in self.agents}
        base_return_so_far = state.base_return_so_far + jnp.array([rewards[a] for a in self.agents])
        new_state = WrappedEnvState(env_state, base_return_so_far)
        return flat_obs, new_state, rewards, dones, infos

    def __getattr__(self, name):
        # Forward any other attribute access to the underlying env
        return getattr(self.env, name)
