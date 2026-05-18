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

from agents.bc.bc_lstm import BCLSTMAgent, BCLSTMPolicyWrapper
from agents.bc.evaluate_lbf import LBF_CONFIGS, load_bc_config
from agents.bc.evaluate_ppo_human_likeness import evaluate_batch, pad_obs
from common.run_episodes import run_episodes
from envs import make_env
from envs.log_wrapper import LogWrapper
from human_data_processing.load_lbf_data import load_bc_data_padded
from marl.ippo import initialize_agent, make_train


class NullLogger:
    def log_item(self, *args, **kwargs):
        pass

    def commit(self):
        pass


def make_config(args, coef: float) -> dict:
    env_kwargs = dict(LBF_CONFIGS[args.lbf_config])
    return {
        "ALG": "ippo",
        "ENV_NAME": "lbf",
        "ENV_KWARGS": env_kwargs,
        "TASK_NAME": "lbf",
        "ACTOR_TYPE": args.actor_type,
        "TOTAL_TIMESTEPS": args.total_timesteps,
        "NUM_CHECKPOINTS": args.num_checkpoints,
        "NUM_ENVS": args.num_envs,
        "ROLLOUT_LENGTH": args.rollout_length,
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
        "HUMAN_REG_COEF": coef,
        "HUMAN_REG_MODE": "reference_kl" if coef > 0 else "none",
        "HUMAN_REG_SCHEDULE": args.human_reg_schedule,
        "HUMAN_REG_WARMUP_FRAC": args.human_reg_warmup_frac,
        "HUMAN_REG_RAMP_FRAC": args.human_reg_ramp_frac,
        "HUMAN_REF_BC_CHECKPOINT": args.human_ref_bc_checkpoint,
        "HUMAN_REF_BC_CONFIG": args.human_ref_bc_config,
    }


def evaluate_policy_return(env, policy, params, args, seed: int | None = None) -> tuple[float, float]:
    eval_seed = args.eval_seed if seed is None else seed
    out = run_episodes(
        jax.random.PRNGKey(eval_seed),
        env,
        agent_0_param=params,
        agent_0_policy=policy,
        agent_1_param=params,
        agent_1_policy=policy,
        max_episode_steps=args.max_steps,
        num_eps=args.num_eval_episodes,
        agent_0_test_mode=args.test_mode,
        agent_1_test_mode=args.test_mode,
    )
    returns = np.asarray(out["returned_episode_returns"])[:, 0]
    return float(returns.mean()), float(np.median(returns))


def take_checkpoint(params, idx: int):
    return jax.tree.map(lambda x: x[idx], params)


def select_by_return(env, policy, out, args) -> dict:
    candidates = [("final", out["final_params"])]
    if not args.final_only:
        ckpt_count = int(np.asarray(out.get("final_ckpt_idx", 0)))
        for idx in range(ckpt_count):
            candidates.append((f"ckpt_{idx}", take_checkpoint(out["checkpoints"], idx)))

    best = None
    for label, params in candidates:
        mean_return, median_return = evaluate_policy_return(
            env,
            policy,
            params,
            args,
            seed=args.select_eval_seed,
        )
        item = {
            "label": label,
            "params": params,
            "selection_mean_return": mean_return,
            "selection_median_return": median_return,
        }
        if best is None or (mean_return, median_return) > (
            best["selection_mean_return"],
            best["selection_median_return"],
        ):
            best = item
    return best


def evaluate_human_likeness(policy, params, data, args) -> tuple[float, float, int]:
    obs = pad_obs(data.obs, policy.obs_dim)
    actions = data.actions
    total_nll = 0.0
    total_correct = 0.0
    total_valid = 0.0
    num_episodes = obs.shape[0]
    for start in range(0, num_episodes, args.human_batch_episodes):
        stop = min(start + args.human_batch_episodes, num_episodes)
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
    return (
        total_nll / max(total_valid, 1.0),
        total_correct / max(total_valid, 1.0),
        int(total_valid),
    )


def evaluate_bc_human_likeness(agent, data, args) -> tuple[float, float, int]:
    total_nll = 0.0
    total_correct = 0.0
    total_valid = 0.0
    num_episodes = data.obs.shape[0]
    for start in range(0, num_episodes, args.human_batch_episodes):
        stop = min(start + args.human_batch_episodes, num_episodes)
        obs = data.obs[start:stop]
        if obs.shape[-1] < agent.config.obs_dim:
            obs = jnp.pad(obs, [(0, 0), (0, 0), (0, agent.config.obs_dim - obs.shape[-1])])
        actions = data.actions[start:stop]
        avail = data.avail_actions[start:stop]
        mask = data.mask[start:stop]
        carry = agent.init_carry((obs.shape[0],))
        _, logits = jax.lax.scan(
            lambda c, x: agent.network.apply({"params": agent.params}, c, x),
            carry,
            jnp.swapaxes(obs, 0, 1),
        )
        logits = jnp.swapaxes(logits, 0, 1)
        logits = jnp.where(avail > 0, logits, -1e9)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        action_log_probs = jnp.take_along_axis(
            log_probs,
            actions[..., None],
            axis=-1,
        ).squeeze(-1)
        pred = jnp.argmax(logits, axis=-1)
        action_available = jnp.take_along_axis(
            avail,
            actions[..., None],
            axis=-1,
        ).squeeze(-1)
        valid = mask & action_available
        total_nll += float((-action_log_probs * valid).sum())
        total_correct += float(((pred == actions) & valid).sum())
        total_valid += float(valid.sum())
    return (
        total_nll / max(total_valid, 1.0),
        total_correct / max(total_valid, 1.0),
        int(total_valid),
    )


def evaluate_bc(args, data) -> dict:
    config = load_bc_config(args.human_ref_bc_config)
    env_kwargs = LBF_CONFIGS[args.lbf_config]
    if config.lbf_feature_mode != "none":
        config = config._replace(
            lbf_grid_size=env_kwargs["grid_size"],
            lbf_num_food=env_kwargs["num_food"],
        )
    agent = BCLSTMAgent(config, weight_path=args.human_ref_bc_checkpoint)
    policy = BCLSTMPolicyWrapper(config)
    env = LogWrapper(make_env("lbf", env_kwargs))
    mean_return, median_return = evaluate_policy_return(env, policy, agent.params, args)
    nll, accuracy, valid_steps = evaluate_bc_human_likeness(agent, data, args)
    return {
        "lbf_config": args.lbf_config,
        "method": "BC",
        "lambda": "BC",
        "seed": "BC",
        "eval_seed": args.eval_seed,
        "select_eval_seed": args.select_eval_seed,
        "total_timesteps": 0,
        "human_reg_schedule": "BC",
        "human_reg_warmup_frac": 0.0,
        "human_reg_ramp_frac": 0.0,
        "mean_return": mean_return,
        "median_return": median_return,
        "final_mean_return": mean_return,
        "final_median_return": median_return,
        "selection_mean_return": mean_return,
        "selection_median_return": median_return,
        "selected_checkpoint": "BC",
        "human_cross_entropy": nll,
        "human_action_accuracy": accuracy,
        "bc_nll": nll,
        "bc_accuracy": accuracy,
        "valid_steps": valid_steps,
        "human_reg_loss_mean": 0.0,
        "human_reg_coef_mean": 0.0,
        "elapsed_sec": 0.0,
    }


def run_one(args, coef: float, data) -> dict:
    config = make_config(args, coef)
    env = LogWrapper(make_env("lbf", config["ENV_KWARGS"]))
    rng = jax.random.PRNGKey(args.seed)
    start = time.time()
    out = jax.jit(make_train(config, env, NullLogger()))(rng)
    elapsed = time.time() - start
    policy, _ = initialize_agent(
        config["ACTOR_TYPE"],
        config,
        env,
        jax.random.PRNGKey(args.seed + 1),
    )
    final_mean, final_median = evaluate_policy_return(env, policy, out["final_params"], args)
    selected = select_by_return(env, policy, out, args)
    params = selected["params"]
    mean_return, median_return = evaluate_policy_return(env, policy, params, args)
    if data is None:
        nll, accuracy, valid_steps = float("nan"), float("nan"), 0
    else:
        nll, accuracy, valid_steps = evaluate_human_likeness(policy, params, data, args)
    human_reg = np.asarray(out["metrics"].get("human_reg_loss", 0.0))
    human_reg_coef = np.asarray(out["metrics"].get("human_reg_coef", 0.0))
    return {
        "lbf_config": args.lbf_config,
        "method": f"HR-IPPO-{args.human_reg_schedule}" if coef > 0 else "IPPO",
        "lambda": coef,
        "seed": args.seed,
        "eval_seed": args.eval_seed,
        "select_eval_seed": args.select_eval_seed,
        "total_timesteps": args.total_timesteps,
        "human_reg_schedule": args.human_reg_schedule if coef > 0 else "none",
        "human_reg_warmup_frac": args.human_reg_warmup_frac if coef > 0 else 0.0,
        "human_reg_ramp_frac": args.human_reg_ramp_frac if coef > 0 else 0.0,
        "mean_return": mean_return,
        "median_return": median_return,
        "final_mean_return": final_mean,
        "final_median_return": final_median,
        "selection_mean_return": selected["selection_mean_return"],
        "selection_median_return": selected["selection_median_return"],
        "selected_checkpoint": selected["label"],
        "human_cross_entropy": nll,
        "human_action_accuracy": accuracy,
        "bc_nll": nll,
        "bc_accuracy": accuracy,
        "valid_steps": valid_steps,
        "human_reg_loss_mean": float(human_reg.mean()) if human_reg.size else 0.0,
        "human_reg_coef_mean": float(human_reg_coef.mean()) if human_reg_coef.size else 0.0,
        "elapsed_sec": elapsed,
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(args):
    needs_reference = args.include_bc or any(coef > 0 for coef in args.human_reg_coef)
    if needs_reference and (
        not args.human_ref_bc_checkpoint or not args.human_ref_bc_config
    ):
        raise ValueError(
            "Set --human_ref_bc_checkpoint and --human_ref_bc_config for BC or KL runs"
        )

    data = None
    if not args.skip_human_likeness or args.include_bc:
        data = load_bc_data_padded(
            config_name=args.lbf_config,
            exclude_uncertain=args.exclude_uncertain,
        )
    rows = []
    if args.include_bc:
        row = evaluate_bc(args, data)
        rows.append(row)
        print(row, flush=True)
        write_rows(Path(args.output_csv), rows)
    for coef in args.human_reg_coef:
        row = run_one(args, coef, data)
        rows.append(row)
        print(row, flush=True)
        write_rows(Path(args.output_csv), rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lbf_config", required=True, choices=sorted(LBF_CONFIGS))
    parser.add_argument("--human_reg_coef", nargs="+", type=float, required=True)
    parser.add_argument("--human_ref_bc_checkpoint")
    parser.add_argument("--human_ref_bc_config")
    parser.add_argument("--actor_type", choices=["mlp", "rnn", "s5"], default="mlp")
    parser.add_argument("--total_timesteps", type=int, default=65536)
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--rollout_length", type=int, default=64)
    parser.add_argument("--num_eval_episodes", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--num_checkpoints", type=int, default=4)
    parser.add_argument("--update_epochs", type=int, default=2)
    parser.add_argument("--num_minibatches", type=int, default=4)
    parser.add_argument("--human_batch_episodes", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip_eps", type=float, default=0.1)
    parser.add_argument("--ent_coef", type=float, default=1e-3)
    parser.add_argument("--human_reg_schedule",
                        choices=["constant", "linear_ramp"],
                        default="constant")
    parser.add_argument("--human_reg_warmup_frac", type=float, default=0.0)
    parser.add_argument("--human_reg_ramp_frac", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_seed", type=int, default=1)
    parser.add_argument("--select_eval_seed", type=int, default=2)
    parser.add_argument("--test_mode", action="store_true")
    parser.add_argument("--exclude_uncertain", action="store_true")
    parser.add_argument("--include_bc", action="store_true")
    parser.add_argument("--skip_human_likeness", action="store_true")
    parser.add_argument("--final_only", action="store_true")
    parser.add_argument("--output_csv", required=True)
    main(parser.parse_args())
