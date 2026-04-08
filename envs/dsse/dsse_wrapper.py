"""BaseEnv wrapper around the JAX DSSE backend. Mirrors the pattern
used by envs/lbf/lbf_wrapper.py: dict-keyed agent interface, auto
reset on done, jit-safe.
"""

from functools import partial
from typing import Dict, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from flax.struct import dataclass as flax_dataclass
from jaxmarl.environments import spaces

from ..base_env import BaseEnv, WrappedEnvState
from .dsse_jax import DSSEJax, DSSEState


class DSSEWrapper(BaseEnv):
    """jax-aht wrapper for DSSE (Drone Swarm Search Environment).

    Args:
        grid_size: Size of the search grid (grid_size x grid_size).
        n_drones: Number of drone agents.
        n_targets: Number of search targets.
        timestep_limit: Maximum steps per episode.
        probability_of_detection: Probability of detecting target when searching.
        drift_x: Probability center x drift per step.
        drift_y: Probability center y drift per step.
        dispersion_start: Initial Gaussian sigma.
        dispersion_inc: Sigma increase per step.
        share_rewards: Whether to share rewards across all agents.
    """

    def __init__(
        self,
        grid_size: int = 15,
        n_drones: int = 4,
        n_targets: int = 2,
        timestep_limit: int = 100,
        probability_of_detection: float = 0.9,
        share_rewards: bool = True,
        **kwargs,
    ):
        # Pass all kwargs through to DSSEJax (vector_x/y, dispersion, etc.)
        self.env = DSSEJax(
            grid_size=grid_size,
            n_drones=n_drones,
            n_targets=n_targets,
            timestep_limit=timestep_limit,
            probability_of_detection=probability_of_detection,
            share_rewards=share_rewards,
            **kwargs,
        )
        self.share_rewards = share_rewards
        self.num_agents = n_drones
        self.grid_size = grid_size
        self.timestep_limit = timestep_limit
        self.name = "DSSE"
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]

        obs_size = self.env.obs_size
        self.observation_spaces = {
            agent: spaces.Box(
                low=jnp.zeros(obs_size),
                high=jnp.ones(obs_size),
                shape=(obs_size,),
                dtype=jnp.float32,
            )
            for agent in self.agents
        }
        self.action_spaces = {
            agent: spaces.Discrete(num_categories=9)
            for agent in self.agents
        }

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], WrappedEnvState]:
        env_state = self.env.reset(key)
        obs_array = self.env.get_obs(env_state)
        obs = {self.agents[i]: obs_array[i] for i in range(self.num_agents)}

        avail_actions = {
            agent: jnp.ones(9, dtype=jnp.bool_) for agent in self.agents
        }

        state = WrappedEnvState(
            env_state=env_state,
            base_return_so_far=jnp.zeros(self.num_agents),
            avail_actions=avail_actions,
            step=jnp.int32(0),
        )
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: WrappedEnvState,
        actions: Dict[str, chex.Array],
        reset_state: Optional[WrappedEnvState] = None,
    ) -> Tuple[Dict[str, chex.Array], WrappedEnvState, Dict[str, float], Dict[str, bool], Dict]:
        key, key_step, key_reset = jax.random.split(key, 3)

        # Convert dict actions to array
        actions_array = jnp.array(
            [actions[agent] for agent in self.agents], dtype=jnp.int32,
        )

        # Step the underlying environment
        new_env_state, rewards_array, done, raw_info = self.env.step(
            key_step, state.env_state, actions_array,
        )

        # Get observations
        obs_array = self.env.get_obs(new_env_state)
        obs_st = {self.agents[i]: obs_array[i] for i in range(self.num_agents)}

        # Rewards. Convention note: when share_rewards=True the backend
        # (dsse_jax.py) already divides the team reward by n_drones before
        # adding the boundary penalty, so rewards_array[i] ~= team_reward /
        # n_drones + boundary_penalty_i. Summing here reconstructs
        # (team_reward + sum(boundary_penalty)) as the per-agent reward, i.e.
        # every agent receives the full team reward. This differs from the
        # LBF wrapper which uses jnp.mean on an already-per-agent reward
        # vector; the two wrappers produce different absolute scales (DSSE
        # x n_drones vs LBF x 1). All DSSE Phase A/B/D reported numbers are
        # on the DSSE scale (per-agent reward == team_reward), so any future
        # change here will invalidate the saved aggregate JSONs.
        if self.share_rewards:
            total_reward = rewards_array.sum()
            rewards = {agent: total_reward for agent in self.agents}
        else:
            rewards = {self.agents[i]: rewards_array[i] for i in range(self.num_agents)}

        # Dones
        dones = {agent: done for agent in self.agents}
        dones["__all__"] = done

        # Broadcast info per-agent (required by jax-aht training loop)
        info = {}
        for k, v in raw_info.items():
            info[k] = jnp.array([v for _ in range(self.num_agents)])

        # Available actions (all actions always available)
        avail_actions = {
            agent: jnp.ones(9, dtype=jnp.bool_) for agent in self.agents
        }

        # Update wrapped state. base_return_so_far accumulates the per-agent
        # return across the current episode; it is reset to zero on the
        # auto-reset path below (state_reset.base_return_so_far == 0), so the
        # invariant is: on the first step of every episode base_return_so_far
        # starts from zero.
        new_base_return = state.base_return_so_far + rewards_array
        state_st = WrappedEnvState(
            env_state=new_env_state,
            base_return_so_far=new_base_return,
            avail_actions=avail_actions,
            step=state.step + 1,
        )

        # Auto-reset on done. When done=True, jax.lax.select picks state_reset,
        # which has base_return_so_far=zeros, so the accumulator resets
        # correctly for the next episode.
        obs_reset, state_reset = self.reset(key_reset)
        obs, new_state = jax.tree.map(
            lambda x, y: jax.lax.select(done, x, y),
            (obs_reset, state_reset),
            (obs_st, state_st),
        )

        return obs, new_state, rewards, dones, info

    def observation_space(self, agent: str) -> spaces.Box:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Discrete:
        return self.action_spaces[agent]

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: WrappedEnvState) -> Dict[str, jnp.ndarray]:
        return state.avail_actions
