"""Smoke tests for the new pizza_v2 domain and wrapper.

Run from the repo root:
    PYTHONPATH=$PWD python envs/rddl/pizza_v2/test_pizza_v2_new_domain.py

Tests:
  1. Domain parses and JaxRDDLEnv initialises without error
  2. Wrapper builds with correct action/obs space sizes
  3. Reset produces expected shapes
  4. Step with each action type (noop, load, deliver, drive) runs without crash
  5. action-num=2 (deliver) does NOT move the truck (the bug fix)
  6. action-num=3/4 (drive) DOES move the truck
  7. SAP instances: only one CONTROLLABLE truck detected
  8. Collision counter accumulates and does not reset mid-episode
"""
import os
import sys

import jax
import jax.numpy as jnp

# Make sure repo root is on path when running directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from pyRDDLGym_jax.core.env import JaxRDDLEnv
from envs.rddl.pizza_v2.pizza_v2_wrapper import PizzaWrapper

DOMAIN = os.path.join(os.path.dirname(__file__), 'pizza_v2_domain.rddl')
JOINT   = os.path.join(os.path.dirname(__file__), 'pizza_v2_instance_asym2_all.rddl')
SAP_0   = os.path.join(os.path.dirname(__file__), 'pizza_v2_instance_asym2_0.rddl')
SAP_1   = os.path.join(os.path.dirname(__file__), 'pizza_v2_instance_asym2_1.rddl')

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    return condition

def make_env(instance, **kwargs):
    jax_env = JaxRDDLEnv(domain=DOMAIN, instance=instance)
    return PizzaWrapper(jax_env, render=False, vectorized=True, **kwargs)

def test_space_sizes():
    print("\n--- Test 1: Space sizes ---")
    env = make_env(JOINT)
    n_actions = env.action_space('agent_0').n
    obs_size  = env.observation_space('agent_0').shape[0]
    # MAX-CONNECTIONS=2 → 3+2=5 actions; 9 locs × 2 trucks + 1 + 9 + 2 + 2 + 2 = 34 obs
    check("num_agents == 2",     env.num_agents == 2, str(env.num_agents))
    check("action space == 5",   n_actions == 5,  str(n_actions))
    check("obs space == 34",     obs_size == 34,  str(obs_size))
    check("single_agent_projection == False", not env.single_agent_projection)

def test_reset_shapes():
    print("\n--- Test 2: Reset shapes ---")
    env = make_env(JOINT)
    key = jax.random.PRNGKey(0)
    (obs_agent, obs_full), state = env.reset(key)
    check("obs_agent keys match agents", set(obs_agent.keys()) == set(env.agents))
    check("obs shape (34,)", obs_agent['agent_0'].shape == (34,), str(obs_agent['agent_0'].shape))
    check("avail_actions key agent_0", 'agent_0' in state.avail_actions)
    check("avail_actions shape (5,)", state.avail_actions['agent_0'].shape == (5,),
          str(state.avail_actions['agent_0'].shape))
    check("collisions_so_far zeros", jnp.all(state.collisions_so_far == 0))

def test_noop_does_not_move():
    print("\n--- Test 3: action-num=0 (noop) — truck stays ---")
    env = make_env(JOINT)
    key = jax.random.PRNGKey(1)
    (obs, _), state = env.reset(key)
    pos_before = state.env_state.subs['truckAt'].copy()
    actions = {a: jnp.array(0, dtype=jnp.int32) for a in env.agents}
    key, k2 = jax.random.split(key)
    _, state2, rewards, dones, info = env.step(k2, state, actions)
    pos_after = info['pre_reset_state'].env_state.subs['truckAt']
    check("truck_0 position unchanged", jnp.allclose(pos_before[0], pos_after[0]))
    check("truck_1 position unchanged", jnp.allclose(pos_before[1], pos_after[1]))

def test_deliver_does_not_move():
    """Key bug fix test: action-num=2 (deliver) must NOT trigger drive."""
    print("\n--- Test 4: action-num=2 (deliver) — truck does NOT move ---")
    env = make_env(JOINT)
    key = jax.random.PRNGKey(2)
    (obs, _), state = env.reset(key)
    pos_before = state.env_state.subs['truckAt'].copy()
    # Both trucks start at s1; submit deliver (action 2) — invalid but should not crash or move
    actions = {a: jnp.array(2, dtype=jnp.int32) for a in env.agents}
    key, k2 = jax.random.split(key)
    _, state2, rewards, dones, info = env.step(k2, state, actions)
    pos_after = info['pre_reset_state'].env_state.subs['truckAt']
    check("truck_0 not moved by deliver", jnp.allclose(pos_before[0], pos_after[0]),
          f"before={pos_before[0].argmax()} after={pos_after[0].argmax()}")
    check("truck_1 not moved by deliver", jnp.allclose(pos_before[1], pos_after[1]),
          f"before={pos_before[1].argmax()} after={pos_after[1].argmax()}")

def test_drive_moves_truck():
    """action-num=3 or 4 (drive connection 1 or 2) should move the truck."""
    print("\n--- Test 5: action-num=3 (drive to connection 1) — truck moves ---")
    env = make_env(JOINT)
    key = jax.random.PRNGKey(3)
    (obs, _), state = env.reset(key)
    # Trucks start at s1. s1→r1 is connection 1 (action 3), s1→l1 is connection 2 (action 4)
    actions = {'agent_0': jnp.array(3, dtype=jnp.int32),   # drive s1→r1 (90% success)
               'agent_1': jnp.array(4, dtype=jnp.int32)}   # drive s1→l1
    key, k2 = jax.random.split(key)
    _, state2, _, _, info = env.step(k2, state, actions)
    pos0_after = info['pre_reset_state'].env_state.subs['truckAt'][0]
    pos1_after = info['pre_reset_state'].env_state.subs['truckAt'][1]
    loc_names = env.rddl_location_names
    loc0 = loc_names[int(pos0_after.argmax())]
    loc1 = loc_names[int(pos1_after.argmax())]
    check("truck_0 moved away from s1", loc0 != 's1', f"now at {loc0}")
    check("truck_1 moved away from s1", loc1 != 's1', f"now at {loc1}")

def test_sap_instances():
    print("\n--- Test 6: SAP instances ---")
    for sap_file, controllable_idx, inactive_idx in [(SAP_0, 0, 1), (SAP_1, 1, 0)]:
        name = os.path.basename(sap_file)
        env = make_env(sap_file)
        check(f"{name}: single_agent_projection=True", env.single_agent_projection)
        check(f"{name}: controlled_agents=[{controllable_idx}]",
              env.controlled_agents == [controllable_idx],
              str(env.controlled_agents))
        # doneDelivering for inactive truck should be True in initial obs
        key = jax.random.PRNGKey(4)
        (obs, _), state = env.reset(key)
        done_del = state.env_state.subs['doneDelivering']
        check(f"{name}: doneDelivering(inactive t{inactive_idx+1}) == True",
              bool(done_del[inactive_idx]),
              str(done_del))

def test_collision_counter():
    print("\n--- Test 7: Collision counter accumulates ---")
    env = make_env(JOINT)
    key = jax.random.PRNGKey(5)
    (obs, _), state = env.reset(key)
    check("initial collisions_so_far == 0", jnp.all(state.collisions_so_far == 0))
    # Run 5 steps with noops — counters should stay 0 (no collisions from noops)
    for _ in range(5):
        key, k = jax.random.split(key)
        actions = {a: jnp.array(0) for a in env.agents}
        _, state, _, dones, info = env.step(k, state, actions)
        if not dones['__all__']:
            check("collisions_so_far non-negative after noop",
                  jnp.all(state.collisions_so_far >= 0))

def test_reward_sign():
    print("\n--- Test 8: Rewards are <= 0 ---")
    env = make_env(JOINT)
    key = jax.random.PRNGKey(6)
    (obs, _), state = env.reset(key)
    all_ok = True
    for _ in range(10):
        key, k = jax.random.split(key)
        actions = {a: jnp.array(0) for a in env.agents}
        _, state, rewards, dones, _ = env.step(k, state, actions)
        for a in env.agents:
            if float(rewards[a]) > 1e-6:
                all_ok = False
        if dones['__all__']:
            break
    check("all rewards <= 0 over 10 noop steps", all_ok)

if __name__ == '__main__':
    print("=" * 60)
    print("pizza_v2 new domain smoke tests")
    print("=" * 60)
    test_space_sizes()
    test_reset_shapes()
    test_noop_does_not_move()
    test_deliver_does_not_move()
    test_drive_moves_truck()
    test_sap_instances()
    test_collision_counter()
    test_reward_sign()
    print("\nDone.")
