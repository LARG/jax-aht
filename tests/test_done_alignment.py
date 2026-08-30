"""Verify that replaying a trajectory with the stored pre-step done (prev_done)
reproduces the stepwise rollout outputs exactly, and that replaying with the
post-step done (the old behavior) does not.

Rollouts step recurrent nets one step at a time with the done observed BEFORE
the step (prev_done); the nets reset the carry before processing input t. The
PPO replay must therefore also be fed prev_done, not the post-step done stored
for GAE.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # deterministic numerics for exact comparison

import jax
import jax.numpy as jnp
import pytest

from agents.rnn_actor_critic_agent import RNNActorCriticPolicy
from agents.s5_actor_critic_agent import S5ActorCriticPolicy

T, B, OBS, ACT = 8, 4, 6, 5
# Episode boundary mid-trajectory: post-step done fires at t=3, so prev_done
# fires at t=4.
DONE_NEXT = jnp.zeros((T, B), dtype=bool).at[3].set(True)
PREV_DONE = jnp.zeros((T, B), dtype=bool).at[4].set(True)


def _make(policy_cls, **kwargs):
    policy = policy_cls(action_dim=ACT, obs_dim=OBS, **kwargs)
    params = policy.init_params(jax.random.PRNGKey(0))
    return policy, params


def _rollout_stepwise(policy, params, obs, prev_done, avail):
    hstate = policy.init_hstate(B)
    vals, logits = [], []
    for t in range(T):
        _, v, pi, hstate = policy.get_action_value_policy(
            params=params,
            obs=obs[t][None],
            done=prev_done[t][None],
            avail_actions=avail[t][None],
            hstate=hstate,
            rng=jax.random.PRNGKey(0),
        )
        vals.append(v.squeeze(0))
        logits.append(pi.logits.squeeze(0))
    return jnp.stack(vals), jnp.stack(logits)


def _replay(policy, params, obs, done, avail):
    _, v, pi, _ = policy.get_action_value_policy(
        params=params,
        obs=obs,
        done=done,
        avail_actions=avail,
        hstate=policy.init_hstate(B),
        rng=jax.random.PRNGKey(0),
    )
    return v, pi.logits


@pytest.mark.parametrize("policy_cls,kwargs", [
    (RNNActorCriticPolicy, {}),
    (S5ActorCriticPolicy, {}),
])
def test_replay_with_prev_done_matches_rollout(policy_cls, kwargs):
    policy, params = _make(policy_cls, **kwargs)
    rng = jax.random.PRNGKey(1)
    obs = jax.random.normal(rng, (T, B, OBS))
    avail = jnp.ones((T, B, ACT))

    roll_v, roll_logits = _rollout_stepwise(policy, params, obs, PREV_DONE, avail)
    replay_v, replay_logits = _replay(policy, params, obs, PREV_DONE, avail)
    good_diff = max(
        float(jnp.abs(roll_v - replay_v).max()),
        float(jnp.abs(roll_logits - replay_logits).max()),
    )
    assert good_diff < 1e-5, f"replay with prev_done diverges from rollout (max diff {good_diff})"

    # The old behavior (replaying with post-step done) resets one step early
    # and must diverge at/after the episode boundary.
    _, bad_logits = _replay(policy, params, obs, DONE_NEXT, avail)
    bad_diff = float(jnp.abs(roll_logits - bad_logits).max())
    assert bad_diff > 100 * max(good_diff, 1e-7), (
        "replay with post-step done unexpectedly matched rollout; test is not discriminating"
    )
