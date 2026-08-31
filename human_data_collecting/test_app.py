import json
import time
from pathlib import Path

import jax

from agents.lbf import SequentialFruitAgent
from envs import make_env


def test_environment():
    env = make_env(
        "lbf",
        {
            "time_limit": 5,
            "grid_size": 7,
            "num_agents": 2,
            "num_food": 3,
            "highlight_agent_idx": 0,
        },
    )

    observations, _ = env.reset(jax.random.PRNGKey(0))

    assert observations["agent_0"].shape == (15,)


def test_agent():
    agent = SequentialFruitAgent(
        grid_size=7,
        num_fruits=3,
        ordering_strategy="nearest_agent",
    )

    assert agent.init_agent_state(1) is not None
    assert agent.get_name()


def test_episode():
    env = make_env(
        "lbf",
        {"time_limit": 5, "grid_size": 7, "num_agents": 2, "num_food": 3},
    )
    agent = SequentialFruitAgent(7, 3, "nearest_agent")
    rng = jax.random.PRNGKey(42)
    observations, state = env.reset(rng)
    agent_state = agent.init_agent_state(1)

    rng, human_key, agent_key, step_key = jax.random.split(rng, 4)
    human_action = jax.random.randint(human_key, (), 0, 6)
    agent_action, agent_state = agent.get_action(
        observations["agent_1"], state, agent_state, agent_key
    )
    observations, _, rewards, dones, _ = env.step(
        step_key,
        state,
        {"agent_0": human_action, "agent_1": agent_action},
    )

    assert set(observations) == set(env.agents)
    assert set(rewards) == set(env.agents)
    assert "__all__" in dones


def test_flask_imports():
    from flask import Flask, jsonify
    from flask_cors import CORS

    assert Flask
    assert jsonify
    assert CORS


def test_data_timestamps(monkeypatch, tmp_path):
    import human_data_collecting.app as app_module

    def choose_test_agent(game):
        return SequentialFruitAgent(
            game.grid_size,
            game.num_fruits,
            "nearest_agent",
        )

    monkeypatch.setattr(app_module.GameSession, "_choose_agent", choose_test_agent)
    monkeypatch.setattr(app_module, "__file__", str(tmp_path / "app.py"))

    game = app_module.GameSession(
        session_id="test-session",
        max_steps=5,
        env_kwargs={"grid_size": 7, "num_food": 3},
    )
    game.step(0)
    game.done = True
    game.end_time = time.time()
    path = Path(game.save_episode(player_name="tester"))

    assert path.parent == tmp_path / "collected_data"
    data = json.loads(path.read_text())
    assert data["start_time"] is not None
    assert data["end_time"] is not None
    assert data["duration"] >= 0
    assert data["trajectory"][0]["timestamp"] is not None
    assert data["trajectory"][0]["elapsed"] >= 0
