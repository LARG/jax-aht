import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.bc.bc_lstm import BCLSTMAgent, BCLSTMPolicyWrapper
from agents.bc.evaluate_lbf import (
    LBF_CONFIGS,
    load_bc_config,
    make_partner,
    run_episode,
)
from envs import make_env


def is_supported_partner(label: str) -> bool:
    return (
        label.startswith("SequentialFruitAgent(")
        or label.startswith("GreedyHeuristicAgent(")
        or label.startswith("ippo_mlp")
    )


def checkpoint_paths(checkpoint_dir: Path, config_name: str) -> tuple[Path, Path]:
    ckpt = checkpoint_dir / f"{config_name}.safetensors"
    cfg = checkpoint_dir / f"{config_name}.yaml"
    if not ckpt.exists() or not cfg.exists():
        raise FileNotFoundError(
            f"Missing checkpoint/config for {config_name} in {checkpoint_dir}"
        )
    return ckpt, cfg


def rows_for_config(summary_csv: Path, config_name: str) -> list[dict]:
    rows = []
    with open(summary_csv) as f:
        for row in csv.DictReader(f):
            if row["config"] == config_name and row["agent_type"] != "ALL":
                rows.append(row)
    return rows


def evaluate_partner(config_name: str, partner_label: str, bc_policy, bc_params,
                     num_episodes: int, max_steps: int, seed: int) -> tuple[float, int]:
    env = make_env("lbf", LBF_CONFIGS[config_name])
    partner_args = SimpleNamespace(
        lbf_config=config_name,
        partner_label=partner_label,
        partner="seq",
        ordering_strategy="lexicographic",
    )
    partner_policy = make_partner(partner_args)

    rng = jax.random.PRNGKey(seed)
    returns = []
    completed = 0
    for _ in range(num_episodes):
        rng, ep_rng = jax.random.split(rng)
        ep_return, _, ep_done = run_episode(
            ep_rng,
            env,
            bc_policy,
            bc_params,
            partner_policy,
            max_steps,
            True,
        )
        returns.append(ep_return)
        completed += int(ep_done)
    return float(np.mean(returns)), completed


def main(args):
    summary_csv = Path(args.summary_csv)
    checkpoint_dir = Path(args.checkpoint_dir)

    for config_name in args.lbf_config:
        ckpt_path, cfg_path = checkpoint_paths(checkpoint_dir, config_name)
        bc_config = load_bc_config(str(cfg_path))
        bc_agent = BCLSTMAgent(bc_config, weight_path=str(ckpt_path))
        bc_policy = BCLSTMPolicyWrapper(bc_config)

        weighted_human = 0.0
        weighted_bc = 0.0
        supported_n = 0
        unsupported = []

        print(f"config,partner,n,human_return,bc_return,completed")
        for row in rows_for_config(summary_csv, config_name):
            partner_label = row["agent_type"]
            n = int(row["num_episodes"])
            human_return = float(row["avg_return"])
            if not is_supported_partner(partner_label):
                unsupported.append(partner_label)
                continue

            try:
                bc_return, completed = evaluate_partner(
                    config_name,
                    partner_label,
                    bc_policy,
                    bc_agent.params,
                    args.num_episodes,
                    args.max_steps,
                    args.seed,
                )
            except (FileNotFoundError, ValueError) as exc:
                unsupported.append(f"{partner_label} ({exc})")
                continue
            weighted_human += human_return * n
            weighted_bc += bc_return * n
            supported_n += n
            print(
                f"{config_name},{partner_label},{n},"
                f"{human_return:.4f},{bc_return:.4f},{completed}/{args.num_episodes}"
            )

        if supported_n:
            print(
                f"{config_name},SUPPORTED_WEIGHTED_AVG,{supported_n},"
                f"{weighted_human / supported_n:.4f},"
                f"{weighted_bc / supported_n:.4f},-"
            )
        if unsupported:
            print(f"{config_name},UNSUPPORTED,{len(unsupported)},"
                  f"{'|'.join(sorted(unsupported))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate BC-LSTM against dataset partner labels."
    )
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument(
        "--summary_csv",
        default="human_data_processing/processed/summary_stats.csv",
    )
    parser.add_argument(
        "--lbf_config",
        nargs="+",
        required=True,
        choices=sorted(LBF_CONFIGS),
    )
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
