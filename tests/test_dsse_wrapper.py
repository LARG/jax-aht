"""Smoke and coordination-invariant tests for the DSSE wrapper.

The basic rollout test confirms make_env('dsse', ...) resets, steps,
and terminates. The coordination invariant test is the important one:
on the 7x7 ndr=2 coordination config, a single drone acting alone must
never trigger a detection. Reverting the coordination rule would make
this test fail loudly. Run with:

    .venv/bin/python tests/test_dsse_wrapper.py
"""
import jax
import jax.numpy as jnp

from envs import make_env
from envs.log_wrapper import LogWrapper

NUM_EPISODES = 3
SEED = 17


def basic_rollout():
    """Smoke test: 7x7 coordination resets, steps, and terminates."""
    env = make_env(
        "dsse",
        env_kwargs={
            "grid_size": 7,
            "n_drones": 2,
            "n_targets": 1,
            "timestep_limit": 100,
            "n_drones_to_rescue": 2,
            "target_cluster_radius": 1,
            "share_rewards": True,
        },
    )
    wrapper = LogWrapper(env)

    key = jax.random.PRNGKey(SEED)
    for ep in range(NUM_EPISODES):
        key, sk = jax.random.split(key)
        obs, state = wrapper.reset(sk)
        done = {"__all__": False}
        total = 0.0
        steps = 0
        while not bool(done["__all__"]):
            actions = {}
            for agent in wrapper.agents:
                key, ak = jax.random.split(key)
                actions[agent] = int(wrapper.action_space(agent).sample(ak))
            key, sk = jax.random.split(key)
            obs, state, rew, done, info = wrapper.step(sk, state, actions)
            steps += 1
            total += float(sum(rew.values()))
            assert steps <= 200, f"episode failed to terminate after {steps} steps"
        # share_rewards=True normalizes return roughly to [0, n_drones];
        # we just sanity-bound it to a sane range.
        assert -10.0 <= total <= 10.0, f"episode return out of range: {total}"
        print(f"  episode {ep}: steps={steps} total_reward={total:.4f}")


def coordination_invariant():
    """ndr=2 means a single drone alone cannot trigger detection.

    We launch 32 envs, take random actions for ONLY drone 0 while drone 1
    sits at action 0 (NOOP-equivalent: any single fixed action is fine
    because what matters is that the second drone never co-locates), and
    confirm the cumulative reward across the episode is exactly 0 in the
    overwhelming majority of envs (allowing for accidental co-location).
    """
    env = make_env(
        "dsse",
        env_kwargs={
            "grid_size": 7,
            "n_drones": 2,
            "n_targets": 1,
            "timestep_limit": 50,
            "n_drones_to_rescue": 2,
            "target_cluster_radius": 1,
            "share_rewards": True,
        },
    )

    n_envs = 32
    key = jax.random.PRNGKey(SEED + 1)
    keys = jax.random.split(key, n_envs)
    reset_v = jax.vmap(env.reset)
    step_v = jax.vmap(env.step)
    obs, state = reset_v(keys)

    cumrew = jnp.zeros(n_envs)
    for t in range(50):
        key, k0 = jax.random.split(key)
        # Drone 0 acts randomly; drone 1 takes a fixed action so the only
        # way to co-locate is via random drone-0 walking onto drone-1's
        # initial cell.
        a0 = jax.random.randint(k0, (n_envs,), 0, env.action_space("agent_0").n)
        a1 = jnp.zeros((n_envs,), dtype=jnp.int32)
        actions = {"agent_0": a0, "agent_1": a1}
        step_keys = jax.random.split(key, n_envs)
        key, _ = jax.random.split(key)
        obs, state, rew, done, info = step_v(step_keys, state, actions)
        cumrew = cumrew + rew["agent_0"]
    # Most envs should have 0 reward; we tolerate at most 25% nonzero
    # (a generous slack for random co-location given the tiny 7x7 grid).
    nonzero = int((cumrew != 0).sum())
    print(f"  envs with any reward when drone-0 acts alone: {nonzero}/{n_envs}")
    assert nonzero <= n_envs // 4, (
        f"coordination invariant violated: {nonzero}/{n_envs} envs got reward "
        "without coordinated action; ndr=2 + cluster_radius=1 is not enforcing "
        "joint detection"
    )


def main():
    print("[dsse] basic 7x7 coordination rollout")
    basic_rollout()
    print("[dsse] coordination joint-action invariant (ndr=2, radius=1)")
    coordination_invariant()
    print("[dsse] OK")


if __name__ == "__main__":
    main()
