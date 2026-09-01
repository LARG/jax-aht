import jax

from envs import make_env


def test_overcooked_wrapper_reset_and_step():
    env = make_env(
        "overcooked-v1",
        {
            "layout": "cramped_room",
            "random_reset": True,
            "max_steps": 1,
        },
    )
    key = jax.random.PRNGKey(0)
    key, reset_key = jax.random.split(key)
    observations, state = env.reset(reset_key)

    assert set(observations) == set(env.agents)
    assert all(observations[agent].ndim == 1 for agent in env.agents)

    actions = {}
    for agent in env.agents:
        key, action_key = jax.random.split(key)
        actions[agent] = env.action_space(agent).sample(action_key)

    key, step_key = jax.random.split(key)
    observations, state, rewards, dones, info = env.step(step_key, state, actions)

    assert set(observations) == set(env.agents)
    assert set(rewards) == set(env.agents)
    assert set(dones) == {*env.agents, "__all__"}
    assert bool(dones["__all__"])
    assert int(env.get_step_count(state)) == 0
    assert "base_return" in info
    assert env.get_avail_actions(state).keys() == observations.keys()

    key, step_key = jax.random.split(key)
    _, state, _, dones, _ = env.step(step_key, state, actions)
    assert bool(dones["__all__"])
    assert int(env.get_step_count(state)) == 0
