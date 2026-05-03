"""Smoke test for Hanabi heuristic agents.

Rolls out a few episodes with random agent pairs and checks that the
env steps without error and returns finite rewards. Full Hanabi
(5c/5r) and mini-Hanabi (3c/3r) are both tested.

OBL agent test is skipped if pretrained weights are not downloaded
(bash agents/hanabi/download_obl_r2d2.sh).
"""
from typing import Dict, Tuple
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from envs import make_env
from agents.hanabi import RandomAgent
from agents.hanabi.rule_based_agent import RuleBasedAgent, VALID_STRATEGIES


FULL_HANABI_KWARGS = {
    "num_agents": 2,
    "num_colors": 5,
    "num_ranks": 5,
    "max_info_tokens": 8,
    "max_life_tokens": 3,
    "num_cards_of_rank": [3, 2, 2, 2, 1],
}

MINI_HANABI_KWARGS = {
    "num_agents": 2,
    "num_colors": 3,
    "num_ranks": 3,
    "hand_size": 3,
    "max_info_tokens": 5,
    "max_life_tokens": 3,
    "num_cards_of_rank": [2, 2, 1],
}

# Hanabi episodes end when deck is empty + all hands played, or when
# 3 life tokens are lost, or when all fireworks are complete.
# Upper bound on episode length for the termination guard.
MAX_EPISODE_STEPS = 200


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
        action0, agent0_state = agent0.get_action(
            obs["agent_0"], state, agent0_state, act0_rng)
        action1, agent1_state = agent1.get_action(
            obs["agent_1"], state, agent1_state, act1_rng)
        actions = {"agent_0": action0, "agent_1": action1}

        key, subkey = jax.random.split(key)
        obs, state, rewards, done, info = env.step(subkey, state, actions)

        for agent in env.agents:
            total_rewards[agent] += float(rewards[agent])
        num_steps += 1
        if num_steps > MAX_EPISODE_STEPS:
            raise RuntimeError(
                f"Episode exceeded {MAX_EPISODE_STEPS} steps; env did not terminate.")
    return total_rewards, num_steps


def _run_pair(env_kwargs, agent0, agent1, num_episodes=2, seed=0):
    env = make_env(env_name="hanabi", env_kwargs=env_kwargs)
    key = jax.random.PRNGKey(seed)
    returns = []
    for _ in range(num_episodes):
        key, subkey = jax.random.split(key)
        total_rewards, _ = run_episode(env, agent0, agent1, subkey)
        for r in total_rewards.values():
            assert np.isfinite(r), f"Non-finite reward: {total_rewards}"
        returns.append(np.mean(list(total_rewards.values())))
    return float(np.mean(returns))


def _make_random_agent(env_kwargs):
    """Create a RandomAgent with the correct action count for the variant."""
    env = make_env(env_name="hanabi", env_kwargs=env_kwargs)
    num_actions = env.action_space(env.agents[0]).n
    return RandomAgent(num_actions=num_actions, agent_names=list(env.agents))


def _make_rule_based_agent(env_kwargs, strategy="cautious"):
    """Create a RuleBasedAgent with correct params for the variant."""
    env = make_env(env_name="hanabi", env_kwargs=env_kwargs)
    num_actions = env.action_space(env.agents[0]).n
    hand_size = env_kwargs.get("hand_size", 5)
    num_colors = env_kwargs.get("num_colors", 5)
    num_ranks = env_kwargs.get("num_ranks", 5)
    return RuleBasedAgent(
        strategy=strategy, hand_size=hand_size,
        num_colors=num_colors, num_ranks=num_ranks,
        num_actions=num_actions, agent_names=list(env.agents),
    )


def test_random_pair_full():
    agent = _make_random_agent(FULL_HANABI_KWARGS)
    mean_ret = _run_pair(FULL_HANABI_KWARGS, agent, agent,
                         num_episodes=2, seed=0)
    assert np.isfinite(mean_ret)


def test_random_pair_mini():
    agent = _make_random_agent(MINI_HANABI_KWARGS)
    mean_ret = _run_pair(MINI_HANABI_KWARGS, agent, agent,
                         num_episodes=2, seed=1)
    assert np.isfinite(mean_ret)


def test_rule_based_all_strategies_full():
    """Test all 4 rule-based strategies on full Hanabi."""
    for strategy in VALID_STRATEGIES:
        agent = _make_rule_based_agent(FULL_HANABI_KWARGS, strategy=strategy)
        mean_ret = _run_pair(FULL_HANABI_KWARGS, agent, agent,
                             num_episodes=2, seed=hash(strategy) % 1000)
        assert np.isfinite(mean_ret), f"strategy={strategy} returned non-finite"


def test_rule_based_all_strategies_mini():
    """Test all 4 rule-based strategies on mini-Hanabi."""
    for strategy in VALID_STRATEGIES:
        agent = _make_rule_based_agent(MINI_HANABI_KWARGS, strategy=strategy)
        mean_ret = _run_pair(MINI_HANABI_KWARGS, agent, agent,
                             num_episodes=2, seed=hash(strategy) % 1000)
        assert np.isfinite(mean_ret), f"strategy={strategy} returned non-finite"


def test_rule_based_vs_random():
    """Test rule-based agent paired with random agent (mixed pair)."""
    rb = _make_rule_based_agent(FULL_HANABI_KWARGS, strategy="aggressive")
    rand = _make_random_agent(FULL_HANABI_KWARGS)
    mean_ret = _run_pair(FULL_HANABI_KWARGS, rb, rand,
                         num_episodes=2, seed=42)
    assert np.isfinite(mean_ret)


def test_obl_agent():
    """Test OBL R2D2 agent if weights are available."""
    weight_file = ("agents/hanabi/obl-r2d2-flax/icml_OBL1/"
                   "OFF_BELIEF1_SHUFFLE_COLOR0_BZA0_BELIEF_a.safetensors")
    if not os.path.exists(weight_file):
        pytest.skip(
            f"OBL weights not found at {weight_file}; "
            "run: bash agents/hanabi/download_obl_r2d2.sh"
        )

    from jaxmarl.wrappers.baselines import load_params
    from agents.hanabi.obl_r2d2_agent import OBLAgentR2D2

    params = load_params(weight_file)
    agent = OBLAgentR2D2()
    env = make_env(env_name="hanabi", env_kwargs=FULL_HANABI_KWARGS)

    key = jax.random.PRNGKey(42)
    obs, state = env.reset(key)
    carry = agent.initialize_carry(jax.random.PRNGKey(0),
                                   batch_dims=(env.num_agents,))

    total_reward = 0.0
    for step in range(100):
        obs_batch = jnp.stack([obs[a] for a in env.agents])
        avail = env.get_avail_actions(state)
        legal_batch = jnp.stack([avail[a] for a in env.agents])
        carry, act = agent.greedy_act(params, carry,
                                      (obs_batch, legal_batch))
        act_dict = {a: act[i] for i, a in enumerate(env.agents)}
        key, step_key = jax.random.split(key)
        obs, state, rew, done, info = env.step(step_key, state, act_dict)
        total_reward += sum(float(rew[a]) for a in env.agents)
        if done["__all__"]:
            break

    print(f"OBL episode: reward={total_reward:.1f}, steps={step + 1}")
    assert total_reward > 0, (
        f"OBL scored 0 in {step + 1} steps; pretrained agent should score >0")


if __name__ == "__main__":
    print("[hanabi] random pair (full 5c/5r)")
    test_random_pair_full()
    print("[hanabi] random pair (mini 3c/3r)")
    test_random_pair_mini()
    print("[hanabi] rule-based all strategies (full)")
    test_rule_based_all_strategies_full()
    print("[hanabi] rule-based all strategies (mini)")
    test_rule_based_all_strategies_mini()
    print("[hanabi] rule-based vs random (mixed)")
    test_rule_based_vs_random()
    print("[hanabi] OBL R2D2 agent")
    test_obl_agent()
    print("[hanabi] all agent smoke tests passed.")
