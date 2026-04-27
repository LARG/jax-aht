"""Generate a heldout BR config by ingesting BR checkpoint log files.

This script reads checkpoint logs written by:
- scripts/run_lbf_br_jobs.sh
- scripts/run_lbf_extra_br_jobs.sh
- scripts/run_overcooked_br_jobs.sh
- scripts/run_overcooked_extra_br_jobs.sh

and updates best_response_set[*][*].path entries in a target config.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


JOB_RE = re.compile(r"job_key=([^\s]+)")
CKPT_RE = re.compile(r"checkpoint_path=(.+)")


@dataclass
class LogRecord:
    job_key: str
    checkpoint_path: str
    source_file: str


def _task_prefix_to_task_name(prefix: str) -> str:
    if prefix == "lbf":
        return "lbf"
    return f"overcooked-v1/{prefix}"


def _task_name_to_prefix(task_name: str) -> str:
    if task_name == "lbf":
        return "lbf"
    if task_name.startswith("overcooked-v1/"):
        return task_name.split("/", 1)[1]
    return task_name


def parse_checkpoint_logs(log_files: list[Path]) -> dict[str, LogRecord]:
    records: dict[str, LogRecord] = {}
    for log_file in sorted(log_files):
        text = log_file.read_text(encoding="utf-8", errors="ignore")
        blocks = re.split(r"\n\s*\n", text)
        for block in blocks:
            job_match = JOB_RE.search(block)
            ckpt_match = CKPT_RE.search(block)
            if not job_match or not ckpt_match:
                continue

            job_key = job_match.group(1).strip()
            checkpoint_path = ckpt_match.group(1).strip()
            records[job_key] = LogRecord(
                job_key=job_key,
                checkpoint_path=checkpoint_path,
                source_file=str(log_file),
            )
    return records


def _expected_job_keys(best_response_set: DictConfig) -> set[str]:
    keys: set[str] = set()
    for task_name, agents in best_response_set.items():
        prefix = _task_name_to_prefix(str(task_name))
        for agent_name in agents.keys():
            keys.add(f"{prefix}.{agent_name}")
    return keys


def generate_config_from_logs(
    *,
    base_config_path: Path,
    output_config_path: Path,
    log_files: list[Path],
) -> dict[str, Any]:
    cfg = OmegaConf.load(base_config_path)
    if not isinstance(cfg, DictConfig) or "best_response_set" not in cfg:
        raise ValueError(f"Config missing best_response_set: {base_config_path}")

    best_response_set = cfg["best_response_set"]
    records = parse_checkpoint_logs(log_files)

    updated = 0
    ignored_not_found = 0
    unknown_job_keys: list[str] = []

    for job_key, record in sorted(records.items()):
        if "." not in job_key:
            unknown_job_keys.append(job_key)
            continue

        prefix, agent_name = job_key.split(".", 1)
        task_name = _task_prefix_to_task_name(prefix)

        if task_name not in best_response_set or agent_name not in best_response_set[task_name]:
            unknown_job_keys.append(job_key)
            continue

        if record.checkpoint_path == "NOT_FOUND":
            ignored_not_found += 1
            continue

        best_response_set[task_name][agent_name]["path"] = record.checkpoint_path
        updated += 1

    expected_keys = _expected_job_keys(best_response_set)
    logged_keys = set(records.keys())
    missing_expected = sorted(expected_keys - logged_keys)

    output_config_path.parent.mkdir(parents=True, exist_ok=True)
    content = OmegaConf.to_yaml(cfg, resolve=False)
    output_config_path.write_text(content, encoding="utf-8")

    summary = {
        "base_config": str(base_config_path),
        "output_config": str(output_config_path),
        "log_files": [str(p) for p in sorted(log_files)],
        "job_records_seen": len(records),
        "updated_paths": updated,
        "ignored_not_found": ignored_not_found,
        "unknown_job_keys": unknown_job_keys,
        "missing_expected_job_keys": missing_expected,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("evaluation/configs/global_heldout_br.yaml"),
    )
    parser.add_argument(
        "--output-config",
        type=Path,
        default=Path("evaluation/configs/global_heldout_br_generated.yaml"),
    )
    parser.add_argument(
        "--logs-glob",
        default="*_br_checkpoints_gpu*.txt",
        help="Glob (relative to repo root) for checkpoint log files.",
    )
    args = parser.parse_args()

    log_files = list(Path(".").glob(args.logs_glob))
    summary = generate_config_from_logs(
        base_config_path=args.base_config,
        output_config_path=args.output_config,
        log_files=log_files,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
