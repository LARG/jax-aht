import argparse
import csv
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.bc.evaluate_lbf import LBF_CONFIGS
from agents.lbf.agent_policy_wrappers import (
    LBFEntitledPolicyWrapper,
    LBFGreedyHeuristicPolicyWrapper,
    LBFSequentialFruitPolicyWrapper,
)
from ego_agent_training.ppo_br import HeuristicPolicyPopulation
from ego_agent_training.ppo_ego import train_ppo_ego_agent
from ego_agent_training.utils import initialize_ego_agent
from envs import make_env
from envs.log_wrapper import LogWrapper


def make_sweep_config(
    args,
    lbf_config: str,
    human_reg_coef: float,
    gail_reward_coef: float,
    seed: int,
) -> dict:
    return {
        "ENV_NAME": "lbf",
        "ENV_KWARGS": dict(LBF_CONFIGS[lbf_config]),
        "EGO_ACTOR_TYPE": args.actor_type,
        "NUM_EGO_TRAIN_SEEDS": 1,
        "TOTAL_TIMESTEPS": args.total_timesteps,
        "NUM_CHECKPOINTS": args.num_checkpoints,
        "NUM_ENVS": args.num_envs,
        "ROLLOUT_LENGTH": args.rollout_length,
        "NUM_EVAL_EPISODES": args.num_eval_episodes,
        "LR": args.lr,
        "UPDATE_EPOCHS": args.update_epochs,
        "NUM_MINIBATCHES": args.num_minibatches,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": args.clip_eps,
        "ENT_COEF": args.ent_coef,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 1.0,
        "ANNEAL_LR": False,
        "HUMAN_REG_COEF": human_reg_coef,
        "HUMAN_REG_MODE": args.human_reg_mode,
        "HUMAN_REG_LBF_CONFIG": lbf_config,
        "HUMAN_REG_INVALID_ACTION_MODE": "noop",
        "HUMAN_REG_BATCH_EPISODES": args.human_batch_episodes,
        "HUMAN_REF_BC_CHECKPOINT": args.human_ref_bc_checkpoint,
        "HUMAN_REF_BC_CONFIG": args.human_ref_bc_config,
        "GAIL_REWARD_COEF": gail_reward_coef,
        "GAIL_ENV_REWARD_COEF": args.gail_env_reward_coef,
        "GAIL_DISC_LR": args.gail_disc_lr,
        "GAIL_DISC_EPOCHS": args.gail_disc_epochs,
        "GAIL_DISC_BATCH_EPISODES": args.gail_disc_batch_episodes,
        "GAIL_DISC_HIDDEN_DIM": args.gail_disc_hidden_dim,
        "TRAIN_SEED": seed,
    }


def make_partner_population(partner_label: str, lbf_config: str, init_ego_params):
    env_kwargs = LBF_CONFIGS[lbf_config]
    if partner_label.startswith("SequentialFruitAgent("):
        ordering_strategy = partner_label.removeprefix(
            "SequentialFruitAgent("
        ).removesuffix(")")
        partner_policy = LBFSequentialFruitPolicyWrapper(
            grid_size=env_kwargs["grid_size"],
            num_fruits=env_kwargs["num_food"],
            ordering_strategy=ordering_strategy,
            using_log_wrapper=True,
        )
    elif partner_label.startswith("GreedyHeuristicAgent("):
        heuristic = partner_label.removeprefix(
            "GreedyHeuristicAgent("
        ).removesuffix(")")
        partner_policy = LBFGreedyHeuristicPolicyWrapper(
            grid_size=env_kwargs["grid_size"],
            num_fruits=env_kwargs["num_food"],
            heuristic=heuristic,
            using_log_wrapper=True,
        )
    elif partner_label == "EntitledAgent":
        partner_policy = LBFEntitledPolicyWrapper(
            grid_size=env_kwargs["grid_size"],
            num_fruits=env_kwargs["num_food"],
            using_log_wrapper=True,
        )
    else:
        raise ValueError(f"Unsupported heuristic partner label: {partner_label}")
    if "ippo" in partner_label:
        raise ValueError("This sweep script currently supports heuristic partners only")
    partner_params = jax.tree.map(lambda x: x[jnp.newaxis, ...], init_ego_params)
    return HeuristicPolicyPopulation(partner_policy), partner_params


def summarize_run(out: dict) -> dict:
    train_returns = np.asarray(out["metrics"]["returned_episode_returns"])
    eval_returns = np.asarray(out["metrics"]["eval_ep_last_info"]["returned_episode_returns"])
    human_reg = np.asarray(out["metrics"]["human_reg_loss"])
    actor_loss = np.asarray(out["metrics"]["actor_loss"])
    value_loss = np.asarray(out["metrics"]["value_loss"])
    gail_reward = np.asarray(out["metrics"]["gail_reward"])
    gail_disc_loss = np.asarray(out["metrics"]["gail_disc_loss"])
    gail_disc_expert_acc = np.asarray(out["metrics"]["gail_disc_expert_acc"])
    gail_disc_policy_acc = np.asarray(out["metrics"]["gail_disc_policy_acc"])
    return {
        "train_return_mean": float(train_returns.mean()),
        "train_return_last_mean": float(train_returns[:, -1].mean()),
        "eval_return_mean": float(eval_returns.mean()),
        "eval_return_last_mean": float(eval_returns[:, -1].mean()),
        "human_reg_loss_mean": float(human_reg.mean()),
        "human_reg_loss_last_mean": float(human_reg[:, -1].mean()),
        "gail_reward_mean": float(gail_reward.mean()),
        "gail_reward_last_mean": float(gail_reward[:, -1].mean()),
        "gail_disc_loss_mean": float(gail_disc_loss.mean()),
        "gail_disc_expert_acc_mean": float(gail_disc_expert_acc.mean()),
        "gail_disc_policy_acc_mean": float(gail_disc_policy_acc.mean()),
        "actor_loss_mean": float(actor_loss.mean()),
        "value_loss_mean": float(value_loss.mean()),
    }


def run_one(
    args,
    lbf_config: str,
    partner_label: str,
    human_reg_coef: float,
    gail_reward_coef: float,
    seed: int,
) -> dict:
    config = make_sweep_config(
        args,
        lbf_config,
        human_reg_coef,
        gail_reward_coef,
        seed,
    )
    env = LogWrapper(make_env("lbf", config["ENV_KWARGS"]))

    rng = jax.random.PRNGKey(seed)
    rng, init_rng, train_rng = jax.random.split(rng, 3)
    ego_policy, init_ego_params = initialize_ego_agent(config, env, init_rng)
    partner_population, partner_params = make_partner_population(
        partner_label,
        lbf_config,
        init_ego_params,
    )

    start = time.time()
    out = train_ppo_ego_agent(
        config=config,
        env=env,
        train_rng=train_rng,
        ego_policy=ego_policy,
        init_ego_params=init_ego_params,
        n_ego_train_seeds=1,
        partner_population=partner_population,
        partner_params=partner_params,
    )
    elapsed = time.time() - start
    row = {
        "lbf_config": lbf_config,
        "partner_label": partner_label,
        "human_reg_coef": human_reg_coef,
        "human_reg_mode": args.human_reg_mode,
        "gail_reward_coef": gail_reward_coef,
        "gail_env_reward_coef": args.gail_env_reward_coef,
        "seed": seed,
        "total_timesteps": args.total_timesteps,
        "actor_type": args.actor_type,
        "elapsed_sec": elapsed,
    }
    row.update(summarize_run(out))
    print(row, flush=True)
    return row


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(args):
    rows = []
    for lbf_config in args.lbf_config:
        for partner_label in args.partner_label:
            for human_reg_coef in args.human_reg_coef:
                for gail_reward_coef in args.gail_reward_coef:
                    for seed in args.seed:
                        rows.append(
                            run_one(
                                args,
                                lbf_config,
                                partner_label,
                                human_reg_coef,
                                gail_reward_coef,
                                seed,
                            )
                        )
                        write_rows(Path(args.output_csv), rows)
    write_rows(Path(args.output_csv), rows)
    print(f"[human_reg_sweep] wrote {args.output_csv}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LBF human-regularized PPO sweeps.")
    parser.add_argument("--lbf_config", nargs="+", required=True, choices=sorted(LBF_CONFIGS))
    parser.add_argument("--partner_label", nargs="+", required=True)
    parser.add_argument("--human_reg_coef", nargs="+", type=float, default=[0.0, 0.01, 0.05, 0.1])
    parser.add_argument("--human_reg_mode", choices=["dataset_nll", "reference_kl"], default="dataset_nll")
    parser.add_argument("--human_ref_bc_checkpoint", default=None)
    parser.add_argument("--human_ref_bc_config", default=None)
    parser.add_argument("--gail_reward_coef", nargs="+", type=float, default=[0.0])
    parser.add_argument("--gail_env_reward_coef", type=float, default=1.0)
    parser.add_argument("--gail_disc_lr", type=float, default=3e-4)
    parser.add_argument("--gail_disc_epochs", type=int, default=1)
    parser.add_argument("--gail_disc_batch_episodes", type=int, default=32)
    parser.add_argument("--gail_disc_hidden_dim", type=int, default=64)
    parser.add_argument("--seed", nargs="+", type=int, default=[0])
    parser.add_argument("--total_timesteps", type=int, default=65536)
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--rollout_length", type=int, default=64)
    parser.add_argument("--num_eval_episodes", type=int, default=16)
    parser.add_argument("--num_checkpoints", type=int, default=4)
    parser.add_argument("--update_epochs", type=int, default=2)
    parser.add_argument("--num_minibatches", type=int, default=4)
    parser.add_argument("--human_batch_episodes", type=int, default=16)
    parser.add_argument("--actor_type", choices=["mlp", "rnn", "s5"], default="mlp")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip_eps", type=float, default=0.1)
    parser.add_argument("--ent_coef", type=float, default=1e-3)
    parser.add_argument("--output_csv", required=True)
    main(parser.parse_args())
