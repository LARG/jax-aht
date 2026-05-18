import argparse
import csv
import re
import sys
from pathlib import Path

import jax
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.bc.bc_lstm import BCLSTMAgent, BCLSTMPolicyWrapper
from agents.bc.evaluate_lbf import LBF_CONFIGS, load_bc_config
from agents.bc.evaluate_lbf_dataset_partners import (
    evaluate_partner,
    is_supported_partner,
    rows_for_config,
)


def checkpoint_stem(path: Path) -> str:
    return path.name.removesuffix(".safetensors")


def discover_checkpoints(checkpoint_dir: Path, pattern: str) -> list[str]:
    regex = re.compile(pattern)
    stems = [
        checkpoint_stem(path)
        for path in checkpoint_dir.glob("*.safetensors")
        if regex.search(path.name)
    ]
    return sorted(set(stems))


def load_policy_for_config(checkpoint_dir: Path, stem: str, config_name: str):
    cfg_path = checkpoint_dir / f"{stem}.yaml"
    ckpt_path = checkpoint_dir / f"{stem}.safetensors"
    bc_config = load_bc_config(str(cfg_path))
    env_kwargs = LBF_CONFIGS[config_name]
    if bc_config.lbf_feature_mode != "none":
        bc_config = bc_config._replace(
            lbf_grid_size=env_kwargs["grid_size"],
            lbf_num_food=env_kwargs["num_food"],
        )
    bc_agent = BCLSTMAgent(bc_config, weight_path=str(ckpt_path))
    return BCLSTMPolicyWrapper(bc_config), bc_agent.params


def evaluate_checkpoint(args, stem: str) -> tuple[list[dict], dict]:
    checkpoint_dir = Path(args.checkpoint_dir)
    summary_csv = Path(args.summary_csv)
    detail_rows = []
    total_human = 0.0
    total_bc = 0.0
    total_n = 0

    for config_name in args.lbf_config:
        bc_policy, bc_params = load_policy_for_config(
            checkpoint_dir, stem, config_name
        )
        config_human = 0.0
        config_bc = 0.0
        config_n = 0

        for row in rows_for_config(summary_csv, config_name):
            partner_label = row["agent_type"]
            if not is_supported_partner(partner_label):
                continue
            n = int(row["num_episodes"])
            human_return = float(row["avg_return"])
            try:
                bc_return, completed = evaluate_partner(
                    config_name,
                    partner_label,
                    bc_policy,
                    bc_params,
                    args.num_episodes,
                    args.max_steps,
                    args.seed,
                )
            except (FileNotFoundError, ValueError) as exc:
                detail_rows.append({
                    "checkpoint": stem,
                    "config": config_name,
                    "partner": partner_label,
                    "n": n,
                    "human_return": human_return,
                    "bc_return": np.nan,
                    "completed": "",
                    "error": str(exc),
                })
                continue

            detail_rows.append({
                "checkpoint": stem,
                "config": config_name,
                "partner": partner_label,
                "n": n,
                "human_return": human_return,
                "bc_return": bc_return,
                "completed": f"{completed}/{args.num_episodes}",
                "error": "",
            })
            config_human += human_return * n
            config_bc += bc_return * n
            config_n += n

        if config_n:
            total_human += config_human
            total_bc += config_bc
            total_n += config_n
            detail_rows.append({
                "checkpoint": stem,
                "config": config_name,
                "partner": "SUPPORTED_WEIGHTED_AVG",
                "n": config_n,
                "human_return": config_human / config_n,
                "bc_return": config_bc / config_n,
                "completed": "",
                "error": "",
            })

    summary = {
        "checkpoint": stem,
        "n": total_n,
        "human_return": total_human / total_n if total_n else np.nan,
        "bc_return": total_bc / total_n if total_n else np.nan,
        "gap": (total_human - total_bc) / total_n if total_n else np.nan,
    }
    return detail_rows, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(args):
    checkpoint_dir = Path(args.checkpoint_dir)
    stems = args.checkpoint_name or discover_checkpoints(
        checkpoint_dir,
        args.pattern,
    )
    if not stems:
        raise FileNotFoundError(
            f"No checkpoints found in {checkpoint_dir} matching {args.pattern}"
        )

    all_details = []
    summaries = []
    for stem in stems:
        print(f"[select_lbf] evaluating checkpoint={stem}", flush=True)
        detail_rows, summary = evaluate_checkpoint(args, stem)
        all_details.extend(detail_rows)
        summaries.append(summary)
        print(
            "[select_lbf] "
            f"checkpoint={stem} bc_return={summary['bc_return']:.4f} "
            f"human_return={summary['human_return']:.4f}",
            flush=True,
        )

    summaries = sorted(summaries, key=lambda row: row["bc_return"], reverse=True)
    print("checkpoint,n,human_return,bc_return,gap")
    for row in summaries:
        print(
            f"{row['checkpoint']},{row['n']},"
            f"{row['human_return']:.4f},{row['bc_return']:.4f},{row['gap']:.4f}"
        )

    if args.output_csv:
        output_path = Path(args.output_csv)
        write_csv(output_path, all_details)
        write_csv(output_path.with_suffix(".summary.csv"), summaries)
        print(f"[select_lbf] wrote {output_path}")
        print(f"[select_lbf] wrote {output_path.with_suffix('.summary.csv')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Select LBF BC-LSTM checkpoints by simulator rollout return."
    )
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--checkpoint_name", nargs="+", default=None)
    parser.add_argument("--pattern", default=r"epoch_|all|grid")
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
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_csv", default=None)
    main(parser.parse_args())
