import argparse
import csv
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.bc.evaluate_lbf import LBF_CONFIGS
from common.save_load_utils import load_checkpoints
from ego_agent_training.utils import initialize_ego_agent
from envs import make_env
from human_data_processing.load_lbf_data import load_bc_data_padded


def parse_index(value: str) -> tuple[int, ...]:
    if not value:
        return ()
    return tuple(int(part) for part in value.split(","))


def take_index(params, index: tuple[int, ...]):
    if not index:
        return params
    return jax.tree.map(lambda x: x[index], params)


def pad_obs(obs: jnp.ndarray, target_dim: int) -> jnp.ndarray:
    if obs.shape[-1] > target_dim:
        raise ValueError(f"obs_dim={obs.shape[-1]} exceeds policy obs_dim={target_dim}")
    if obs.shape[-1] == target_dim:
        return obs
    return jnp.pad(obs, [(0, 0), (0, 0), (0, target_dim - obs.shape[-1])])


def make_policy_config(args) -> dict:
    config = {
        "ENV_NAME": "lbf",
        "ENV_KWARGS": dict(LBF_CONFIGS[args.lbf_config]),
        "EGO_ACTOR_TYPE": args.actor_type,
        "ACTIVATION": args.activation,
        "FC_HIDDEN_DIM": args.fc_hidden_dim,
        "GRU_HIDDEN_DIM": args.gru_hidden_dim,
        "S5_D_MODEL": args.s5_d_model,
        "S5_SSM_SIZE": args.s5_ssm_size,
        "S5_N_LAYERS": args.s5_n_layers,
        "S5_BLOCKS": args.s5_blocks,
        "S5_ACTOR_CRITIC_HIDDEN_DIM": args.s5_actor_critic_hidden_dim,
        "FC_N_LAYERS": args.fc_n_layers,
    }
    if args.policy_input_dim is not None:
        config["POLICY_INPUT_DIM"] = args.policy_input_dim
    return config


def evaluate_batch(policy, params, obs, actions, avail, mask):
    done = jnp.logical_not(mask)
    hstate = policy.init_hstate(obs.shape[0])
    _, _, pi, _ = policy.get_action_value_policy(
        params=params,
        obs=jnp.swapaxes(obs, 0, 1),
        done=jnp.swapaxes(done, 0, 1),
        avail_actions=jnp.swapaxes(avail, 0, 1),
        hstate=hstate,
        rng=jax.random.PRNGKey(0),
    )
    log_prob = jnp.swapaxes(pi.log_prob(jnp.swapaxes(actions, 0, 1)), 0, 1)
    pred = jnp.swapaxes(pi.mode(), 0, 1)
    action_available = jnp.take_along_axis(
        avail,
        actions[..., None],
        axis=-1,
    ).squeeze(-1)
    valid = mask & action_available
    nll_sum = (-log_prob * valid).sum()
    correct = ((pred == actions) & valid).sum()
    valid_count = valid.sum()
    return nll_sum, correct, valid_count


def evaluate(args):
    data = load_bc_data_padded(
        config_name=args.lbf_config,
        exclude_uncertain=args.exclude_uncertain,
    )
    env = make_env("lbf", LBF_CONFIGS[args.lbf_config])
    policy_config = make_policy_config(args)
    policy, _ = initialize_ego_agent(policy_config, env, jax.random.PRNGKey(args.seed))
    params = load_checkpoints(
        args.train_run,
        ckpt_key=args.ckpt_key,
        custom_loader_cfg={"name": "partial_load"},
    )
    params = take_index(params, parse_index(args.param_index))

    obs = pad_obs(data.obs, policy.obs_dim)
    actions = data.actions
    if args.invalid_action_mode == "noop":
        picked = jnp.take_along_axis(
            data.avail_actions,
            actions[..., None],
            axis=-1,
        ).squeeze(-1)
        actions = jnp.where(data.mask & ~picked, jnp.zeros_like(actions), actions)

    rows = []
    total_nll = 0.0
    total_correct = 0.0
    total_valid = 0.0
    num_episodes = obs.shape[0]
    for start in range(0, num_episodes, args.batch_episodes):
        stop = min(start + args.batch_episodes, num_episodes)
        nll_sum, correct, valid_count = evaluate_batch(
            policy,
            params,
            obs[start:stop],
            actions[start:stop],
            data.avail_actions[start:stop],
            data.mask[start:stop],
        )
        total_nll += float(nll_sum)
        total_correct += float(correct)
        total_valid += float(valid_count)

    nll = total_nll / max(total_valid, 1.0)
    accuracy = total_correct / max(total_valid, 1.0)
    row = {
        "train_run": args.train_run,
        "ckpt_key": args.ckpt_key,
        "param_index": args.param_index,
        "lbf_config": args.lbf_config,
        "actor_type": args.actor_type,
        "valid_steps": int(total_valid),
        "human_cross_entropy": nll,
        "human_action_accuracy": accuracy,
        "bc_nll": nll,
        "bc_accuracy": accuracy,
    }
    rows.append(row)
    print(row)

    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_run", required=True)
    parser.add_argument("--ckpt_key", default="final_params")
    parser.add_argument("--param_index", default="0")
    parser.add_argument("--lbf_config", required=True, choices=sorted(LBF_CONFIGS))
    parser.add_argument("--actor_type", choices=["mlp", "rnn", "s5"], default="mlp")
    parser.add_argument("--policy_input_dim", type=int, default=None)
    parser.add_argument("--activation", default="tanh")
    parser.add_argument("--fc_hidden_dim", type=int, default=64)
    parser.add_argument("--gru_hidden_dim", type=int, default=64)
    parser.add_argument("--s5_d_model", type=int, default=16)
    parser.add_argument("--s5_ssm_size", type=int, default=16)
    parser.add_argument("--s5_n_layers", type=int, default=2)
    parser.add_argument("--s5_blocks", type=int, default=1)
    parser.add_argument("--s5_actor_critic_hidden_dim", type=int, default=64)
    parser.add_argument("--fc_n_layers", type=int, default=2)
    parser.add_argument("--batch_episodes", type=int, default=64)
    parser.add_argument("--invalid_action_mode", choices=["keep", "noop"], default="noop")
    parser.add_argument("--exclude_uncertain", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_csv", default=None)
    evaluate(parser.parse_args())
