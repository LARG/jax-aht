Environment API and `BaseEnv` specification
==========================================

This document describes the expected interface for environment wrappers in this codebase and the contract exposed by `BaseEnv` and `WrappedEnvState` (see `envs/base_env.py`). The goal is to make it straightforward to add new environments while keeping a small and well-defined integration surface for agents and other systems.

Location
--------
- Base class: `envs/base_env.py`
- Examples of wrappers in this folder:
  - `envs/lbf/lbf_wrapper.py` (Jumanji LBF wrapper)
  - `envs/overcooked/overcooked_wrapper.py` (Overcooked wrapper)

Core dataclass: WrappedEnvState
------------------------------
WrappedEnvState is a small container used by the wrappers to hold the environment state and a few auxiliary values that are useful across environments.

Fields (as implemented in `envs/base_env.py`):

- `env_state: Any` — The underlying environment internal state object. This is the concrete state returned by the specific environment implementation (e.g. Jumanji timestep state, Overcooked State, etc.).
- `base_return_so_far: jnp.ndarray` — Per-agent accumulated return (used by some wrappers to track the original/"base" return when shaping is applied).
- `avail_actions: jnp.ndarray` — Per-agent action mask / available-actions array (shape depends on environment/action spec).
- `step: jnp.array` — Current step count (or a scalar holding step information).

Note: this dataclass is intentionally simple. Wrappers may pack extra information inside `env_state` or use `info` dictionaries returned by `step()` to transmit additional metadata (see below).

Base class: BaseEnv (contract)
------------------------------
`BaseEnv` is an abstract class that wrappers should implement. The file `envs/base_env.py` contains the abstract method signatures. The wrappers in this repository follow the functional/JAX-friendly pattern used by JAX/Jumanji/JaxMARL:

Recommended method signatures (used by the existing wrappers):

- `reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], WrappedEnvState]`
  - Reset the environment and return a dictionary of per-agent observations and the initial `WrappedEnvState`.
  - `key` is a PRNG key used for environment randomness.

- `step(self, key: chex.PRNGKey, state: WrappedEnvState, actions: Dict[str, chex.Array], reset_state: Optional[WrappedEnvState] = None) -> Tuple[Dict[str, chex.Array], WrappedEnvState, Dict[str, float], Dict[str, bool], Dict]`
  - Step the environment forward using `actions`.
  - Returns: `(obs, new_state, rewards, dones, infos)`.
  - `obs` is a dict mapping agent id -> observation ndarray (usually flattened or converted to jnp arrays).
  - `new_state` is a `WrappedEnvState` containing the new underlying env state and bookkeeping fields.
  - `rewards` is a dict mapping agent id -> scalar reward (float or jnp scalar).
  - `dones` is a dict mapping agent id -> boolean, and must include the special key `"__all__"` whose value is True when the episode has terminated for all agents.
  - `infos` is a dictionary with any environment-specific debug/auxiliary fields (e.g., `base_reward`).

- `get_avail_actions(self, state: WrappedEnvState) -> Dict[str, jnp.ndarray]`
  - Return per-agent available action masks (1/0 or boolean masks) in an easily consumable format.

- `observation_space(self, agent: str)` and `action_space(self, agent: str)`
  - Return the environment-specific observation and action space objects (the codebase uses `jaxmarl.environments.spaces` compatible objects).

Other recommended helpers
-------------------------
- `get_step_count(self, state: WrappedEnvState) -> jnp.array` — (optional) Returns current global step count. Both the LBF and Overcooked wrappers provide this.

Design patterns used in this repo
---------------------------------
- Dict-of-agents: observations/rewards/avail_actions/actions are represented as dictionaries keyed by agent ids (e.g., `'agent_0'`, `'agent_1'`). Wrappers typically expose `self.agents` (list of str) and `self.num_agents`.
- Flatten observations (optional): many wrappers flatten or concatenate observation components into a single 1D array per agent. This simplifies downstream models that expect vector inputs.
- Auto-reset: wrappers are required to implement auto-reset semantics (see `LBFWrapper.step()`), returning reset observations/state when `done` is True.
- JIT and static args: use `@partial(jax.jit, static_argnums=(0,))` for instance methods where the `self` object should be treated as static.

Dones and episode termination
-----------------------------
- The `dones` dict MUST contain the special key `"__all__"` indicating whether the episode has terminated.
- Wrappers may also populate per-agent done flags (e.g., when agents can terminate individually). Many grid/foraging games use a shared termination flag.

Infos and additional fields
---------------------------
- The `infos` return can carry environment-specific diagnostics. Example: `overcooked_wrapper` includes a `base_reward` field and writes a `base_return` into new_info.
- Use `infos` to surface non-essential telemetry (debugging, reward shaping diagnostics) but keep the core contract in `obs/state/rewards/dones`.

Example minimal wrapper template
--------------------------------
```python
from functools import partial
from typing import Dict, Tuple
import chex
import jax.numpy as jnp
import jax
from .base_env import BaseEnv, WrappedEnvState

class MyEnvWrapper(BaseEnv):
    def __init__(self, ...):
        # create underlying env and set self.agents / self.num_agents
        self.agents = ["agent_0", "agent_1"]
        self.num_agents = len(self.agents)
        # define observation/action spaces

    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, jnp.ndarray], WrappedEnvState]:
        obs_raw, raw_state = self.env.reset(key)
        obs = {agent: ... for agent in self.agents}
        state = WrappedEnvState(env_state=raw_state, base_return_so_far=jnp.zeros(self.num_agents), avail_actions=jnp.zeros(self.num_agents), step=jnp.array(0, dtype=jnp.int32))
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key: chex.PRNGKey, state: WrappedEnvState, actions: Dict[str, chex.Array]):
        obs_raw, env_state, rewards_raw, dones_raw, infos_raw = self.env.step(key, state.env_state, actions)
        obs = {agent: ... for agent in self.agents}
        # build WrappedEnvState and return
        return obs, new_state, rewards, dones, infos
```

Notes about current wrappers
---------------------------
- `LBFWrapper` (jumanji): Uses Jumanji's `reset`/`step` and auto-resets via `jax.lax.select`.
- `OvercookedWrapper` (jaxmarl overcooked): Flattens observations and logs `base_return` into the `info` dictionary.

If you add a new wrapper and want me to review it for compatibility with the rest of the codebase (JIT safety, typing, auto-reset semantics), paste the new file and I'll review and suggest edits.
