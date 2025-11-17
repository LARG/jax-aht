Environment Wrapper Specification
==================================

Environment wrappers should implement `BaseEnv` (see `envs/base_env.py`) and use `WrappedEnvState` to hold state.

## WrappedEnvState Fields

- `env_state: Any` — Underlying environment state
- `base_return_so_far: jnp.ndarray` — Per-agent accumulated return
- `avail_actions: jnp.ndarray` — Per-agent action masks
- `step: jnp.array` — Current step count

## Required Methods

**`reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], WrappedEnvState]`**
- Reset environment and return observations dict and initial state
- `key`: PRNG key for randomness

**`step(self, key: chex.PRNGKey, state: WrappedEnvState, actions: Dict[str, chex.Array], reset_state: Optional[WrappedEnvState] = None) -> Tuple[Dict[str, chex.Array], WrappedEnvState, Dict[str, float], Dict[str, bool], Dict]`**
- Step environment forward with actions
- Returns: `(obs, new_state, rewards, dones, infos)`
  - `obs`: Dict[agent_id -> observation array]
  - `new_state`: WrappedEnvState with updated env_state and bookkeeping
  - `rewards`: Dict[agent_id -> scalar reward]
  - `dones`: Dict[agent_id -> bool], must include `"__all__"` key
  - `infos`: Dict with optional debug/telemetry data

**`get_avail_actions(self, state: WrappedEnvState) -> Dict[str, jnp.ndarray]`**
- Returns per-agent action masks (1/0 or boolean)

**`observation_space(self, agent: str)` and `action_space(self, agent: str)`**
- Return environment-specific space objects (jaxmarl.environments.spaces compatible)

## Key Patterns

- **Dict-based API**: All agent data uses dicts keyed by agent IDs
- **Auto-reset**: Wrappers must auto-reset on episode termination
- **JIT-friendly**: Use `@partial(jax.jit, static_argnums=(0,))` for methods
