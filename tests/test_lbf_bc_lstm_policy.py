import jax
import jax.numpy as jnp
import pytest
import yaml

from agents.bc import BCLSTMAgent, BCLSTMConfig
from agents.lbf.agent_policy_wrappers import LBFBCLSTMPolicyWrapper
from common.agent_loader_from_config import initialize_heuristic_agent_from_config
from envs import make_env
from envs.log_wrapper import LogWrapper


def _write_checkpoint(tmp_path):
    config = BCLSTMConfig(
        obs_dim=24,
        action_dim=6,
        preprocess_dim=16,
        lstm_dim=8,
        postprocess_dim=8,
    )
    agent = BCLSTMAgent(config)
    agent.init_params(jax.random.PRNGKey(0))

    weight_path = tmp_path / "policy.safetensors"
    agent.save_weights(str(weight_path))
    with open(tmp_path / "policy.yaml", "w") as config_file:
        yaml.safe_dump(config._asdict(), config_file)
    return weight_path


def test_lbf_bc_lstm_loader_pads_7x7_observations(tmp_path):
    weight_path = _write_checkpoint(tmp_path)
    policy = initialize_heuristic_agent_from_config(
        {
            "actor_type": "bc_lstm",
            "weight_file": str(weight_path),
            "greedy": True,
        },
        agent_name="human_proxy",
        task_name="lbf/lbf_7x7_nolevels",
    )

    obs_7x7 = jnp.arange(15, dtype=jnp.float32).reshape(1, 1, 15)
    obs_12x12 = jnp.pad(obs_7x7, ((0, 0), (0, 0), (0, 9)))
    legal_actions = jnp.array([0, 0, 0, 1, 0, 0], dtype=jnp.float32)
    done = jnp.array([[False]])
    hstate = policy.init_hstate(1)

    action_7x7, carry_7x7 = policy.get_action(
        None, obs_7x7, done, legal_actions, hstate,
        jax.random.PRNGKey(1),
    )
    action_12x12, carry_12x12 = policy.get_action(
        None, obs_12x12, done, legal_actions, hstate,
        jax.random.PRNGKey(1), env_state=None, test_mode=True,
    )

    assert isinstance(policy, LBFBCLSTMPolicyWrapper)
    assert int(action_7x7) == 3
    assert int(action_12x12) == 3
    assert all(jax.tree.leaves(
        jax.tree.map(jnp.array_equal, carry_7x7, carry_12x12)
    ))


def test_lbf_bc_lstm_loader_rejects_oversized_observations(tmp_path):
    weight_path = _write_checkpoint(tmp_path)
    policy = LBFBCLSTMPolicyWrapper(str(weight_path))

    with pytest.raises(ValueError, match="expects at most 24"):
        policy._prepare_obs(jnp.zeros(25))


@pytest.mark.parametrize(
    ("env_kwargs", "expected_obs_dim"),
    [
        ({"grid_size": 7, "num_food": 3, "different_levels": False}, 15),
        ({"grid_size": 12, "num_food": 6, "different_levels": True}, 24),
    ],
)
def test_lbf_bc_lstm_policy_steps_both_domains(
        tmp_path, env_kwargs, expected_obs_dim):
    weight_path = _write_checkpoint(tmp_path)
    policy = LBFBCLSTMPolicyWrapper(str(weight_path), greedy=True)
    env = LogWrapper(make_env("lbf", env_kwargs))

    rng = jax.random.PRNGKey(2)
    rng, reset_rng, action_rng, step_rng = jax.random.split(rng, 4)
    obs, env_state = env.reset(reset_rng)
    assert obs["agent_1"].shape[-1] == expected_obs_dim

    avail_actions = env.get_avail_actions(env_state)["agent_1"]
    action, _ = policy.get_action(
        None,
        obs["agent_1"].reshape(1, 1, -1),
        jnp.array([[False]]),
        avail_actions,
        policy.init_hstate(1),
        action_rng,
        env_state=env_state,
        test_mode=True,
    )
    env.step(
        step_rng,
        env_state,
        {"agent_0": jnp.array(0), "agent_1": action.squeeze()},
    )
