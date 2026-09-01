"""Validation and optional integration tests for heldout teammate configs."""

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
import yaml

from envs import make_env
from envs.log_wrapper import LogWrapper
from evaluation.heldout_core import load_heldout_set

REPO_ROOT = Path(__file__).resolve().parents[1]
HELDOUT_CONFIG = REPO_ROOT / "evaluation/configs/global_heldout_settings.yaml"
TASK_CONFIG_DIR = REPO_ROOT / "evaluation/configs/task"


def load_heldout_config():
    return yaml.safe_load(HELDOUT_CONFIG.read_text())["heldout_set"]


def load_task_config(task_name):
    return yaml.safe_load((TASK_CONFIG_DIR / f"{task_name}.yaml").read_text())


def assert_agent_can_act(agent_name, agent, env, rng):
    policy, params, test_mode, _ = agent
    rng, reset_rng, action_rng = jax.random.split(rng, 3)
    observations, env_state = env.reset(reset_rng)
    acting_agent = env.agents[1]
    available_actions = env.get_avail_actions(env_state)[acting_agent].astype(
        jnp.float32
    )
    hstate = policy.init_hstate(1, aux_info={"agent_id": 1})

    action, _ = policy.get_action(
        params=params,
        obs=observations[acting_agent].reshape(1, 1, -1),
        done=jnp.zeros((1, 1), dtype=bool),
        avail_actions=available_actions,
        hstate=hstate,
        rng=action_rng,
        aux_obs=None,
        env_state=env_state,
        test_mode=test_mode,
    )
    action = int(jnp.asarray(action).squeeze())

    assert 0 <= action < env.action_space(acting_agent).n, (
        f"{agent_name}: returned invalid action {action}"
    )


def test_heldout_tasks_have_valid_environments():
    for task_name, heldout_config in load_heldout_config().items():
        task_config = load_task_config(task_name)
        env_kwargs = task_config["ENV_KWARGS"] or {}

        assert heldout_config, f"{task_name}: heldout set is empty"
        assert make_env(task_config["ENV_NAME"], env_kwargs).agents


def test_heldout_checkpoint_paths_are_repo_relative():
    for task_name, heldout_config in load_heldout_config().items():
        for agent_name, agent_config in heldout_config.items():
            assert "actor_type" in agent_config, (
                f"{task_name}/{agent_name}: missing actor_type"
            )
            for path_key in ("path", "weight_file"):
                if path_key not in agent_config:
                    continue

                checkpoint_path = Path(agent_config[path_key])
                assert not checkpoint_path.is_absolute(), (
                    f"{task_name}/{agent_name}: {path_key} must be relative to the repository"
                )
                assert ".." not in checkpoint_path.parts
                assert checkpoint_path.parts[0] == "eval_teammates"


@pytest.mark.eval_data
def test_load_all_heldout_agents():
    if os.environ.get("JAX_AHT_RUN_HELDOUT_LOADING") != "1":
        pytest.skip(
            "set JAX_AHT_RUN_HELDOUT_LOADING=1 after running download_eval_data.py"
        )

    rng = jax.random.PRNGKey(0)
    for task_name, heldout_config in load_heldout_config().items():
        task_config = load_task_config(task_name)
        env_kwargs = task_config["ENV_KWARGS"] or {}
        env = LogWrapper(make_env(task_config["ENV_NAME"], env_kwargs))

        agents = load_heldout_set(heldout_config, env, task_name, env_kwargs, rng)

        assert agents, f"{task_name}: no heldout agents loaded"
        for agent_name, agent in agents.items():
            rng, action_rng = jax.random.split(rng)
            assert_agent_can_act(agent_name, agent, env, action_rng)
