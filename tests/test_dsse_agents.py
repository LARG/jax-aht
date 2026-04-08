"""Smoke test for DSSE heuristic agents (random, greedy search, sweep).

Mirrors tests/test_lbf_agents.py: rolls out a few episodes with each pair of
agents and checks that the env steps without error and returns finite rewards.
"""
from typing import Dict, Tuple

import jax
import numpy as np

from envs import make_env
from agents.dsse import RandomAgent, GreedySearchAgent, SweepAgent


GRID_SIZE = 7
ENV_KWARGS = {
    "grid_size": GRID_SIZE,
    "n_drones": 2,
    "n_targets": 1,
    "n_drones_to_rescue": 2,
    "target_cluster_radius": 1,
    "timestep_limit": 50,
    "vector_x": 1.1,
    "vector_y": 1.0,
    "dispersion_inc": 0.1,
}


def run_episode(env, agent0, agent1, key) -> Tuple[Dict[str, float], int]:
    key, subkey = jax.random.split(key)
    obs, state = env.reset(subkey)

    done = {agent: False for agent in env.agents}
    done["__all__"] = False
    total_rewards = {agent: 0.0 for agent in env.agents}
    num_steps = 0

    agent0_state = agent0.init_agent_state(0)
    agent1_state = agent1.init_agent_state(1)

    while not done["__all__"]:
        key, act0_rng, act1_rng = jax.random.split(key, 3)
        action0, agent0_state = agent0.get_action(obs["agent_0"], state, agent0_state, act0_rng)
        action1, agent1_state = agent1.get_action(obs["agent_1"], state, agent1_state, act1_rng)
        actions = {"agent_0": action0, "agent_1": action1}

        key, subkey = jax.random.split(key)
        obs, state, rewards, done, info = env.step(subkey, state, actions)

        for agent in env.agents:
            total_rewards[agent] += float(rewards[agent])
        num_steps += 1
        if num_steps > ENV_KWARGS["timestep_limit"] + 5:
            raise RuntimeError("Episode exceeded timestep_limit; env did not terminate.")
    return total_rewards, num_steps


def _run_pair(agent0, agent1, num_episodes=2, seed=0):
    env = make_env(env_name="dsse", env_kwargs=ENV_KWARGS)
    key = jax.random.PRNGKey(seed)
    returns = []
    for _ in range(num_episodes):
        key, subkey = jax.random.split(key)
        total_rewards, _ = run_episode(env, agent0, agent1, subkey)
        for r in total_rewards.values():
            assert np.isfinite(r), f"Non-finite reward: {total_rewards}"
        returns.append(np.mean(list(total_rewards.values())))
    return float(np.mean(returns))


def test_random_pair_runs():
    mean_ret = _run_pair(RandomAgent(), RandomAgent(), num_episodes=2, seed=0)
    assert np.isfinite(mean_ret)


def test_greedy_pair_runs():
    mean_ret = _run_pair(
        GreedySearchAgent(grid_size=GRID_SIZE),
        GreedySearchAgent(grid_size=GRID_SIZE),
        num_episodes=2,
        seed=1,
    )
    assert np.isfinite(mean_ret)


def test_sweep_pair_runs():
    mean_ret = _run_pair(
        SweepAgent(grid_size=GRID_SIZE),
        SweepAgent(grid_size=GRID_SIZE),
        num_episodes=2,
        seed=2,
    )
    assert np.isfinite(mean_ret)


def test_mixed_pair_runs():
    mean_ret = _run_pair(
        GreedySearchAgent(grid_size=GRID_SIZE),
        SweepAgent(grid_size=GRID_SIZE),
        num_episodes=2,
        seed=3,
    )
    assert np.isfinite(mean_ret)


if __name__ == "__main__":
    test_random_pair_runs()
    test_greedy_pair_runs()
    test_sweep_pair_runs()
    test_mixed_pair_runs()
    print("DSSE agent smoke tests passed.")
