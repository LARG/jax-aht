"""Functional JAX RDDL environment.

pyRDDLGym_jax.core.env.JaxRDDLEnv does not exist in any published PyPI or
GitHub release of pyRDDLGym_jax — the package restructured away from a
gym-style interface.  This module re-implements that class using the
JaxRDDLSimulator's compiled JAX functions, which ARE available.

Calling convention discovered from pyRDDLGym_jax/core/simulator.py:
    value, key, error, model_params = cpf_fn(fls, nfls, model_params, key)

The API exposed here matches exactly what the grid_10x10 and pizza_v2
wrappers expect.
"""

from functools import partial
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax.struct import dataclass

from pyRDDLGym.core.env import RDDLEnv as _BaseRDDLEnv
from pyRDDLGym_jax.core.simulator import JaxRDDLSimulator


# ---------------------------------------------------------------------------
# State / timestep types
# ---------------------------------------------------------------------------

@dataclass
class EnvState:
    subs: Any          # dict[str, jnp.ndarray] — all fluent values
    model_params: Any  # dict — distribution params (usually constant)
    key: jnp.ndarray
    timestep: jnp.ndarray


@dataclass
class TimeStep:
    observation: Any   # dict[str, jnp.ndarray] — state-fluent values only
    reward: jnp.ndarray
    done: jnp.ndarray
    truncated: jnp.ndarray


# ---------------------------------------------------------------------------
# Minimal space stub
# ---------------------------------------------------------------------------

class _Space:
    """Minimal space with .shape, .high, and .dtype — enough for the wrappers."""
    def __init__(self, shape, high=1.0, dtype=jnp.float32):
        self.shape = shape
        self.high = np.full(shape, high, dtype=np.float32)
        self.dtype = dtype


# ---------------------------------------------------------------------------
# JaxRDDLEnv
# ---------------------------------------------------------------------------

class JaxRDDLEnv:
    """Functional JAX RDDL environment built on JaxRDDLSimulator compiled fns.

    reset(key)            -> (EnvState, TimeStep)
    step(state, actions)  -> (EnvState, TimeStep)
    get_available_actions(state) -> dict[str, jnp.ndarray]
    """

    def __init__(self, domain: str, instance: str, **kwargs):
        # --- Parse model ---------------------------------------------------
        base_env = _BaseRDDLEnv(domain=domain, instance=instance)
        model = base_env.model

        # --- Compile JAX functions -----------------------------------------
        sim = JaxRDDLSimulator(rddl=model, raise_error=False)
        # sim._compile() is called inside JaxRDDLSimulator.__init__ via super()

        # Compiled CPFs: list of (name, jax_fn, dtype)
        # Each jax_fn(fls, nfls, model_params, key) -> (value, key, err, mp)
        self._cpfs = sim.cpfs
        self._reward_fn = sim.reward
        self._terminal_fns = sim.terminals

        # Non-fluents are constant across all steps — store as fixed JAX arrays
        self._nfls = {k: jnp.asarray(v) for k, v in sim.nfls.items()}

        # model_params: distribution params — constant for pizza_v2
        self._model_params_init = {
            k: (jnp.asarray(v) if hasattr(v, '__array__') else v)
            for k, v in sim.model_params.items()
        }

        # Initial fluent values (state + action + intermediate, all as JAX arrays)
        # fls contains ALL fluents so the dict structure stays constant every step
        self._init_subs = {k: jnp.asarray(v) for k, v in sim.fls.items()}

        # Action keys (used to filter observations)
        self._action_keys = set(sim.noop_actions.keys())
        # State-fluent keys = everything in init_subs that is not an action
        # (intermediate fluents also end up here but their initial value is 0/False)
        state_vars = {k for k, t in model.variable_types.items()
                      if t == 'state-fluent'}
        self._state_keys = {k for k in self._init_subs if k in state_vars}

        # --- Public properties ---------------------------------------------
        self.horizon = base_env.horizon
        self.model = model
        self.noop_actions = {k: jnp.asarray(v) for k, v in sim.noop_actions.items()}
        self.init_values = {k: jnp.asarray(v) for k, v in sim.init_values.items()}

        # --- Action space --------------------------------------------------
        # Wrappers call sorted(env.action_space.keys()) to get action fluent names,
        # then pass the space to their own _convert_rddl_action_spec_to_jaxmarl_space
        # which reads MAX-CONNECTIONS from non_fluents directly.  So we only need
        # the keys to be correct; .high is a safe fallback.
        self.action_space = {}
        for var in self._action_keys:
            arr = self._init_subs.get(var, jnp.zeros(()))
            n_trucks = arr.shape[0] if arr.ndim > 0 else 1
            max_conn = int(model.non_fluents.get('MAX-CONNECTIONS', 4))
            self.action_space[var] = _Space(
                shape=(n_trucks,),
                high=float(2 + max_conn),
                dtype=jnp.int32,
            )

        # --- Observation space ---------------------------------------------
        # Wrappers call observation_space.keys() and space[k].shape
        self.observation_space = {
            k: _Space(shape=self._init_subs[k].shape, dtype=self._init_subs[k].dtype)
            for k in self._state_keys
            if k in self._init_subs
        }

    # -----------------------------------------------------------------------
    # Core API
    # -----------------------------------------------------------------------

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jnp.ndarray) -> Tuple[EnvState, TimeStep]:
        key, subkey = jax.random.split(key)
        state = EnvState(
            subs=self._init_subs,
            model_params=self._model_params_init,
            key=subkey,
            timestep=jnp.zeros((), jnp.int32),
        )
        obs = {k: self._init_subs[k] for k in self._state_keys
               if k in self._init_subs}
        ts = TimeStep(
            observation=obs,
            reward=jnp.zeros(()),
            done=jnp.zeros((), jnp.bool_),
            truncated=jnp.zeros((), jnp.bool_),
        )
        return state, ts

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        env_state: EnvState,
        actions: Dict[str, jnp.ndarray],
    ) -> Tuple[EnvState, TimeStep]:
        fls = dict(env_state.subs)
        nfls = self._nfls
        mp = env_state.model_params
        key = env_state.key

        # Inject action fluents into fls
        fls.update(actions)

        # Apply all CPFs in topological order (Python loop unrolled at trace time)
        for (cpf, expr, dtype) in self._cpfs:
            value, key, _error, mp = expr(fls, nfls, mp, key)
            fls[cpf] = value.astype(dtype)

        # Propagate next-state values: pyRDDLGym stores state-fluent CPF results
        # under primed keys (e.g. "numPizzasInTruck'") but the next step reads
        # from the unprimed keys ("numPizzasInTruck"). Copy x' -> x so the state
        # actually advances each step.
        for k in list(fls.keys()):
            if k.endswith("'"):
                base = k[:-1]
                if base in fls:
                    fls[base] = fls[k]

        # Reward
        reward, key, _, mp = self._reward_fn(fls, nfls, mp, key)

        # Termination (episode done)
        done = jnp.zeros((), jnp.bool_)
        for terminal_fn in self._terminal_fns:
            t_val, key, _, _ = terminal_fn(fls, nfls, mp, key)
            done = done | jnp.asarray(t_val, jnp.bool_)

        new_t = env_state.timestep + 1
        truncated = new_t >= self.horizon

        new_state = EnvState(
            subs=fls,
            model_params=mp,
            key=key,
            timestep=new_t,
        )
        obs = {k: fls[k] for k in self._state_keys if k in fls}
        ts = TimeStep(
            observation=obs,
            reward=jnp.asarray(reward),
            done=done,
            truncated=jnp.asarray(truncated),
        )
        return new_state, ts

    @partial(jax.jit, static_argnums=(0,))
    def get_available_actions(self, env_state: EnvState) -> Dict[str, jnp.ndarray]:
        """Return action availability masks.

        Returns all-True for now (the wrapper's enforce_action_constraints
        projects invalid actions to noop via pyRDDLGym precondition checks).
        Shape: {action_key: (num_agents, num_actions)} float32 mask.
        """
        result = {}
        for var in self._action_keys:
            space = self.action_space[var]
            n_trucks = space.shape[0]
            n_actions = int(space.high[0]) + 1
            result[var] = jnp.ones((n_trucks, n_actions), dtype=jnp.float32)
        return result
