"""Pull the best hyperparameter set from wandb sweep and apply them to the corresponding config file.

Wandb sweep ids are stored in vis/plot_globals.py
Usage:
    python scripts/manage_configs/apply_best_hparams.py --task lbf/lbf_7x7_nolevels --algorithm trajedi
    python scripts/manage_configs/apply_best_hparams.py --task lbf/lbf_7x7_nolevels --algorithm ppo_ego --dry-run
    python scripts/manage_configs/apply_best_hparams.py --task lbf/lbf_7x7_nolevels  # all algorithms
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from scripts.manage_configs.helpers import format_value, resolve_algo_config
from scripts.paper_vis.plot_globals import (
    HYPERPARAM_SWEEPS, ENTITY, HYPERPARAM_PROJECT,
)
from scripts.utils import ALGO_TO_ENTRY_POINT
from scripts.wandb_utils.wandb_cache import (
    load_sweep_df, build_hparam_df, fetch_sweep_last_run_date,
)


REPO_ROOT = Path(__file__).parent.parent.parent

# Ego algorithms use a different config directory name from the algorithm name.
_EGO_ALGO_TO_CONFIG_NAME = {
    "ppo_ego": "ppo_ego",
    "liam": "liam_ego",
    "meliba": "meliba_ego",
}


def _config_path(task: str, algorithm: str) -> Path:
    parts = task.split("/")
    if algorithm not in ALGO_TO_ENTRY_POINT:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Known: {list(ALGO_TO_ENTRY_POINT)}")
    root = ALGO_TO_ENTRY_POINT[algorithm]
    config_name = _EGO_ALGO_TO_CONFIG_NAME.get(algorithm, algorithm)
    if len(parts) == 1:
        return REPO_ROOT / root / "configs" / "algorithm" / config_name / f"{task}.yaml"
    task_family = "/".join(parts[:-1])
    task_name = parts[-1]
    return REPO_ROOT / root / "configs" / "algorithm" / config_name / task_family / f"{task_name}.yaml"


# MeLIBA's KL loss is now summed over time and *averaged* over the batch. Sweeps that
# ran before that fix summed over the batch too, so their DECODER_KL_WEIGHT has to be
# scaled by the minibatch size (NUM_ENVS / NUM_MINIBATCHES) to reproduce the same
# effective KL weighting. Sweeps run on or after the cutoff already use the new
# normalization and are left alone.
_MELIBA_KL_KEY = "DECODER_KL_WEIGHT"
_MELIBA_KL_FIX_DATE = "2026-08-30"

# Swept keys that must never be written back into an algorithm config.
_NON_HPARAM_KEYS = {"TOTAL_TIMESTEPS"}


def _scale_meliba_kl_weight(task: str, best_hparams: dict, config_path: Path,
                            sweep_id: str) -> None:
    """In-place: rescale MeLIBA's swept KL weight to the post-fix loss normalization.

    Only applied to sweeps whose most recent run predates _MELIBA_KL_FIX_DATE.
    """
    if _MELIBA_KL_KEY not in best_hparams:
        print(f"  WARNING: {_MELIBA_KL_KEY} not in the sweep for {task}; no KL rescaling applied.")
        return

    last_run = fetch_sweep_last_run_date(sweep_id, ENTITY, HYPERPARAM_PROJECT)
    if last_run is None:
        print(f"  WARNING: could not date sweep {sweep_id}; no KL rescaling applied.")
        return
    if last_run[:10] >= _MELIBA_KL_FIX_DATE:
        print(f"  MeLIBA KL rescale skipped: sweep last ran {last_run[:10]}, "
              f"on/after the {_MELIBA_KL_FIX_DATE} loss-normalization fix.")
        return

    algo_configs_root = config_path.parent
    while algo_configs_root.name != "algorithm":
        algo_configs_root = algo_configs_root.parent
    resolved = resolve_algo_config(config_path, algo_configs_root)

    # A swept value takes precedence over the config value.
    num_envs = float(best_hparams.get("NUM_ENVS", resolved["NUM_ENVS"]))
    num_minibatches = float(best_hparams.get("NUM_MINIBATCHES", resolved["NUM_MINIBATCHES"]))
    scale = num_envs / num_minibatches

    raw = float(best_hparams[_MELIBA_KL_KEY])
    best_hparams[_MELIBA_KL_KEY] = raw * scale
    print(f"  MeLIBA KL rescale: {_MELIBA_KL_KEY} {raw:g} * "
          f"(NUM_ENVS {num_envs:g} / NUM_MINIBATCHES {num_minibatches:g} = {scale:g}) "
          f"-> {raw * scale:g}")


def _set_top_level_key(content: str, key: str, new_val: str) -> str:
    """Update an existing top-level 'key: value' line, or append it at end-of-file."""
    pattern = rf"^({re.escape(key)}:[ \t]+)\S+"
    if re.search(pattern, content, flags=re.MULTILINE):
        return re.sub(pattern, rf"\g<1>{new_val}", content, flags=re.MULTILINE)
    return content.rstrip("\n") + f"\n{key}: {new_val}\n"


def _set_nested_key(content: str, parent: str, child: str, new_val: str) -> str:
    """Update a nested 'child: value' line inside the named parent block."""
    lines = content.splitlines(keepends=True)
    new_lines: list[str] = []
    in_parent = False
    parent_indent = -1

    for line in lines:
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        if in_parent:
            if stripped and not stripped.startswith("#") and indent <= parent_indent:
                in_parent = False
            else:
                m = re.match(rf"^([ \t]+{re.escape(child)}:[ \t]+)\S+", line)
                if m:
                    line = f"{m.group(1)}{new_val}\n"

        if re.match(rf"^{re.escape(parent)}[ \t]*:", line):
            in_parent = True
            parent_indent = indent

        new_lines.append(line)

    return "".join(new_lines)


def update_config_file(config_path: Path, best_hparams: dict[str, str], dry_run: bool = False) -> None:
    content = config_path.read_text()
    changed_keys: list[str] = []

    for key, new_val in best_hparams.items():
        if "." in key:
            parent, child = key.split(".", 1)
            updated = _set_nested_key(content, parent, child, new_val)
        else:
            updated = _set_top_level_key(content, key, new_val)

        if updated != content:
            changed_keys.append(key)
            content = updated

    if not changed_keys:
        print(f"  SKIP (up to date): {config_path}")
        return

    if dry_run:
        for key in changed_keys:
            print(f"  DRY RUN  {key}={best_hparams[key]}  →  {config_path}")
        return

    config_path.write_text(content)
    for key in changed_keys:
        print(f"  Set {key}={best_hparams[key]}  →  {config_path}")


def apply_algorithm(task: str, algorithm: str, force_recompute: bool, dry_run: bool,
                    max_hparams: int | None = None, seed: int = 0) -> None:
    raw_df, bare_keys = load_sweep_df(task, algorithm, force_recompute)
    # Some sweep YAMLs pin the (reduced) sweep training budget under `parameters` rather
    # than in the `command` block, so it comes back as a single-valued "swept" key. It is
    # not a hyperparameter — writing it would clobber the benchmark budget set by
    # update_timesteps.py — so drop it before choosing/applying the best setting.
    dropped = [k for k in bare_keys if k in _NON_HPARAM_KEYS]
    if dropped:
        print(f"  Ignoring non-hyperparameter swept key(s): {', '.join(dropped)}")
        bare_keys = [k for k in bare_keys if k not in _NON_HPARAM_KEYS]
    sweep_df = build_hparam_df(raw_df, algorithm, bare_keys)

    grouped = (
        sweep_df.groupby(bare_keys, dropna=False)["_score"]
        .mean()
        .reset_index()
        .sort_values("_score", ascending=False)
    )
    if max_hparams is not None and len(grouped) > max_hparams:
        print(f"Sampling {max_hparams} of {len(grouped)} hparam settings (seed={seed})")
        grouped = grouped.sample(n=max_hparams, random_state=seed).sort_values("_score", ascending=False)
    best_row = grouped.iloc[0]
    raw_hparams = {key: best_row[key] for key in bare_keys}

    config_path = _config_path(task, algorithm)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if algorithm == "meliba":
        _scale_meliba_kl_weight(task, raw_hparams, config_path,
                                HYPERPARAM_SWEEPS[task][algorithm])

    best_hparams = {key: format_value(val) for key, val in raw_hparams.items()}

    print(f"Best hyperparameters for {task}/{algorithm}:")
    for k, v in best_hparams.items():
        print(f"  {k}: {v}")
    print(f"  mean score: {best_row['_score']:.4f}")

    print(f"\nConfig file: {config_path}")

    print()
    update_config_file(config_path, best_hparams, dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the best sweep hyperparameters to the corresponding config file."
    )
    parser.add_argument("--task", required=True, help="Task name (e.g. lbf/lbf_7x7_nolevels).")
    parser.add_argument("--algorithm", default=None,
                        help="Algorithm name (e.g. trajedi). Omit to run all algorithms for the task.")
    parser.add_argument("--force-recompute", action="store_true",
                        help="Re-fetch from wandb, ignoring the local cache.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be written without modifying any files.")
    parser.add_argument("--max-hparams", type=int, default=None,
                        help="Maximum number of hparam settings to consider. If the sweep ran more, "
                             "a random subset of this size is drawn (seeded, for reproducibility).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed used when subsampling hparam settings via --max-hparams.")
    args = parser.parse_args()

    if args.task not in HYPERPARAM_SWEEPS:
        raise ValueError(f"Task '{args.task}' not found. Available: {list(HYPERPARAM_SWEEPS)}")

    algorithms = [args.algorithm] if args.algorithm else list(HYPERPARAM_SWEEPS[args.task])

    for algorithm in algorithms:
        if len(algorithms) > 1:
            print(f"\n{'='*60}")
        apply_algorithm(args.task, algorithm, args.force_recompute, args.dry_run,
                        max_hparams=args.max_hparams, seed=args.seed)


if __name__ == "__main__":
    main()
