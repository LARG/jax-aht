import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.agent_interface import AgentPolicy
from agents.bc.bc_lstm import BCLSTMAgent, BCLSTMConfig, BCLSTMPolicyWrapper
from agents.lbf.agent_policy_wrappers import (
    LBFGreedyHeuristicPolicyWrapper,
    LBFRandomPolicyWrapper,
    LBFSequentialFruitPolicyWrapper,
)
from common.agent_loader_from_config import initialize_rl_agent_from_config
from evaluation.heldout_evaluator import extract_params
from envs import make_env


LBF_CONFIGS = {
    "grid7_food3_nolevels": {
        "grid_size": 7,
        "num_food": 3,
        "different_levels": False,
    },
    "grid7_food3_levels": {
        "grid_size": 7,
        "num_food": 3,
        "different_levels": True,
    },
    "grid12_food6_nolevels": {
        "grid_size": 12,
        "num_food": 6,
        "different_levels": False,
    },
    "grid12_food6_levels": {
        "grid_size": 12,
        "num_food": 6,
        "different_levels": True,
    },
}

IPPO_PARTNER_CONFIGS = {
    "ippo_mlp": {
        "paths": [
            "human_data/teammates/ippo-lbf-7/saved_train_run",
            "eval_teammates/lbf/ippo/2025-04-21_23-41-17/saved_train_run",
            "eval_teammates/eval_teammates/lbf/ippo/"
            "2025-04-21_23-41-17/saved_train_run",
        ],
        "actor_type": "mlp",
        "ckpt_key": "final_params",
        "idx_list": [0],
        "test_mode": False,
    },
    "ippo_mlp_s2c0": {
        "paths": [
            "human_data/teammates/ippo-lbf-7/saved_train_run",
            "eval_teammates/lbf/ippo/2025-04-21_23-41-17/saved_train_run",
            "eval_teammates/eval_teammates/lbf/ippo/"
            "2025-04-21_23-41-17/saved_train_run",
        ],
        "actor_type": "mlp",
        "idx_list": [[2, 0]],
        "test_mode": False,
    },
    "ippo_mlp_7_levels": {
        "paths": [
            "human_data/teammates/ippo-lbf-7-levels/saved_train_run",
            "eval_teammates/eval_teammates/lbf/ippo/"
            "ippo-lbf-7-levels/saved_train_run",
        ],
        "actor_type": "mlp",
        "ckpt_key": "final_params",
        "idx_list": [0],
        "test_mode": False,
    },
    "ippo_mlp_12": {
        "paths": [
            "human_data/teammates/ippo-lbf-12/saved_train_run",
            "eval_teammates/eval_teammates/lbf/ippo/"
            "ippo-lbf-12/saved_train_run",
        ],
        "actor_type": "mlp",
        "ckpt_key": "final_params",
        "idx_list": [0],
        "test_mode": False,
    },
    "ippo_mlp_12_levels": {
        "paths": [
            "human_data/teammates/ippo-lbf-12-levels/saved_train_run",
            "eval_teammates/eval_teammates/lbf/ippo/"
            "ippo-lbf-12-levels/saved_train_run",
        ],
        "actor_type": "mlp",
        "ckpt_key": "final_params",
        "idx_list": [0],
        "test_mode": False,
    },
}


class BoundPolicyWrapper(AgentPolicy):
    def __init__(self, policy: AgentPolicy, params, test_mode: bool):
        super().__init__(policy.action_dim, policy.obs_dim)
        self.policy = policy
        self.params = params
        self.test_mode = test_mode

    def init_hstate(self, batch_size: int, aux_info=None):
        return self.policy.init_hstate(batch_size, aux_info=aux_info)

    def get_action(self, params, obs, done, avail_actions, hstate, rng,
                   aux_obs=None, env_state=None, test_mode=False):
        return self.policy.get_action(
            params=self.params,
            obs=obs,
            done=done,
            avail_actions=avail_actions,
            hstate=hstate,
            rng=rng,
            aux_obs=aux_obs,
            env_state=env_state,
            test_mode=self.test_mode,
        )


def load_bc_config(path: str) -> BCLSTMConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return BCLSTMConfig(
        obs_dim=int(raw["obs_dim"]),
        action_dim=int(raw["action_dim"]),
        preprocess_dim=int(raw.get("preprocess_dim", 256)),
        lstm_dim=int(raw.get("lstm_dim", 128)),
        postprocess_dim=int(raw.get("postprocess_dim", 64)),
        dropout_rate=float(raw.get("dropout_rate", 0.0)),
    )


def resolve_first_existing_path(paths: list[str]) -> str:
    for path in paths:
        if (REPO_ROOT / path).exists() or Path(path).exists():
            return path
    raise FileNotFoundError(
        f"None of the candidate checkpoint paths exist: {paths}"
    )


def make_ippo_partner(partner_label: str, config_name: str):
    if partner_label not in IPPO_PARTNER_CONFIGS:
        raise ValueError(f"No IPPO config for partner label: {partner_label}")

    raw_config = dict(IPPO_PARTNER_CONFIGS[partner_label])
    raw_config["path"] = resolve_first_existing_path(raw_config.pop("paths"))
    test_mode = bool(raw_config.pop("test_mode"))
    env = make_env("lbf", LBF_CONFIGS[config_name])
    rng = jax.random.PRNGKey(0)
    raw_config["custom_loader"] = {"name": "partial_load"}
    policy, params, init_params, idx_labels = initialize_rl_agent_from_config(
        raw_config,
        partner_label,
        env,
        rng,
    )
    params_list, _ = extract_params(params, init_params, idx_labels)
    if len(params_list) != 1:
        raise ValueError(
            f"Expected one checkpoint for {partner_label}, got {len(params_list)}"
        )
    return BoundPolicyWrapper(policy, params_list[0], test_mode)


def make_partner(args):
    if args.partner_label:
        env_kwargs = LBF_CONFIGS[args.lbf_config]
        if args.partner_label.startswith("SequentialFruitAgent("):
            ordering_strategy = args.partner_label.removeprefix(
                "SequentialFruitAgent("
            ).removesuffix(")")
            return LBFSequentialFruitPolicyWrapper(
                grid_size=env_kwargs["grid_size"],
                num_fruits=env_kwargs["num_food"],
                ordering_strategy=ordering_strategy,
            )
        if args.partner_label.startswith("GreedyHeuristicAgent("):
            heuristic = args.partner_label.removeprefix(
                "GreedyHeuristicAgent("
            ).removesuffix(")")
            return LBFGreedyHeuristicPolicyWrapper(
                grid_size=env_kwargs["grid_size"],
                num_fruits=env_kwargs["num_food"],
                heuristic=heuristic,
            )
        if args.partner_label.startswith("ippo_mlp"):
            return make_ippo_partner(args.partner_label, args.lbf_config)
        raise ValueError(
            f"Unsupported partner label for direct evaluation: {args.partner_label}"
        )

    if args.partner == "random":
        return LBFRandomPolicyWrapper()
    if args.partner == "seq":
        env_kwargs = LBF_CONFIGS[args.lbf_config]
        return LBFSequentialFruitPolicyWrapper(
            grid_size=env_kwargs["grid_size"],
            num_fruits=env_kwargs["num_food"],
            ordering_strategy=args.ordering_strategy,
        )
    raise ValueError(f"Unknown partner type: {args.partner}")


def run_episode(rng, env, bc_policy, bc_params, partner_policy, max_steps, test_mode):
    rng, reset_rng = jax.random.split(rng)
    obs, env_state = env.reset(reset_rng)
    done = {
        "agent_0": jnp.array(False),
        "agent_1": jnp.array(False),
        "__all__": jnp.array(False),
    }
    bc_hstate = bc_policy.init_hstate(1)
    partner_hstate = partner_policy.init_hstate(1, aux_info={"agent_id": 1})
    total_return = 0.0
    episode_length = 0

    for _ in range(max_steps):
        avail_actions = env.get_avail_actions(env_state)
        rng, bc_rng, partner_rng, step_rng = jax.random.split(rng, 4)

        bc_action, bc_hstate = bc_policy.get_action(
            params=bc_params,
            obs=obs["agent_0"].reshape(1, 1, -1),
            done=done["agent_0"].reshape(1, 1),
            avail_actions=avail_actions["agent_0"],
            hstate=bc_hstate,
            rng=bc_rng,
            env_state=env_state,
            test_mode=test_mode,
        )
        partner_action, partner_hstate = partner_policy.get_action(
            params=None,
            obs=obs["agent_1"].reshape(1, 1, -1),
            done=done["agent_1"].reshape(1, 1),
            avail_actions=avail_actions["agent_1"],
            hstate=partner_hstate,
            rng=partner_rng,
            env_state=env_state,
            test_mode=True,
        )

        actions = {
            "agent_0": bc_action.squeeze(),
            "agent_1": partner_action.squeeze(),
        }
        obs, env_state, reward, done, _ = env.step(step_rng, env_state, actions)
        total_return += float(reward["agent_0"])
        episode_length += 1
        if bool(done["__all__"]):
            break

    return total_return, episode_length, bool(done["__all__"])


def evaluate(args):
    config = load_bc_config(args.config)
    bc_agent = BCLSTMAgent(config, weight_path=args.checkpoint)
    bc_policy = BCLSTMPolicyWrapper(config)
    partner_policy = make_partner(args)
    env = make_env("lbf", LBF_CONFIGS[args.lbf_config])

    rng = jax.random.PRNGKey(args.seed)
    returns = []
    lengths = []
    completed = []
    for _ in range(args.num_episodes):
        rng, ep_rng = jax.random.split(rng)
        ep_return, ep_length, ep_done = run_episode(
            ep_rng,
            env,
            bc_policy,
            bc_agent.params,
            partner_policy,
            args.max_steps,
            args.test_mode,
        )
        returns.append(ep_return)
        lengths.append(ep_length)
        completed.append(ep_done)

    returns_arr = np.array(returns)
    lengths_arr = np.array(lengths)
    completed_arr = np.array(completed)
    print(f"[eval_lbf_bc] config={args.lbf_config} partner={args.partner}")
    if args.partner_label:
        print(f"[eval_lbf_bc] partner_label={args.partner_label}")
    print(
        f"[eval_lbf_bc] episodes={args.num_episodes} "
        f"completed={completed_arr.sum()}/{args.num_episodes}"
    )
    print(
        f"[eval_lbf_bc] return_mean={returns_arr.mean():.4f} "
        f"return_std={returns_arr.std():.4f}"
    )
    print(f"[eval_lbf_bc] length_mean={lengths_arr.mean():.2f}")
    print(f"[eval_lbf_bc] returns={returns_arr.tolist()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate an LBF BC-LSTM checkpoint.")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to .safetensors BC weights")
    parser.add_argument("--config", required=True,
                        help="Path to BC checkpoint YAML config")
    parser.add_argument("--lbf_config", required=True, choices=sorted(LBF_CONFIGS))
    parser.add_argument("--partner", choices=["seq", "random"], default="seq")
    parser.add_argument(
        "--partner_label",
        default=None,
        help="Exact dataset partner label, e.g. SequentialFruitAgent(lexicographic)",
    )
    parser.add_argument("--ordering_strategy", default="lexicographic")
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test_mode", action="store_true",
                        help="Use greedy BC actions")
    evaluate(parser.parse_args())
