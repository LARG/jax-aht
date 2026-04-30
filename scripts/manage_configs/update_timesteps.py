#!/usr/bin/env python3
"""
Adjust algorithm configs to achieve a target total timestep budget per difficulty tier.

For each algorithm × task combination the script:
  1. Resolves the algorithm config.
  2. Computes the current total timestep budget.
  3. Adjusts the minimal set of parameters to reach the target, using
     per-algorithm priority rules:
       - ROTATE / open_ended_minimax : NUM_OPEN_ENDED_ITERS and timesteps_per_iteration
       - FCP                         : PARTNER_POP_SIZE
       - COLE                        : TOTAL_TIMESTEPS_PER_ITERATION
       - CoMeDi                      : TOTAL_TIMESTEPS_PER_ITERATION and PARTNER_POP_SIZE
       - All others                  : TOTAL_TIMESTEPS
     ego_train_algorithm.TOTAL_TIMESTEPS is always kept fixed.
  4. Prints old → new totals and writes the updated configs (unless --dry-run).

At the end a summary table (via report_timesteps) is printed showing the new
totals across all algorithms and tasks.

Targets accept K / M / B suffixes: e.g. 130M, 1.3B, 500K, or plain numbers.

Usage:
    python scripts/manage_configs/update_timesteps.py <entry_point_dir> [<entry_point_dir> ...]
        --easy-target <N>
        --hard-target <N>
        [--skip-algos ALGO ...]
        [--dry-run]

Examples:
    python scripts/manage_configs/update_timesteps.py teammate_generation/ \\
        --easy-target 195M --hard-target 390M --dry-run

    python scripts/manage_configs/update_timesteps.py open_ended_training/ \\
        --easy-target 195M --hard-target 390M --skip-algos open_ended_minimax paired --dry-run

    python scripts/manage_configs/update_timesteps.py ego_agent_training/ \\
        --easy-target 11M --hard-target 23M --skip-algos ppo_br --dry-run
"""

import re
import sys
from pathlib import Path

from scripts.manage_configs.helpers import (
    EASY_TASKS,
    HARD_TASKS,
    ALL_TASKS,
    resolve_algo_config,
    compute_total_timesteps,
    compute_target_params,
    format_value,
    format_human,
    parse_human_timesteps,
)
from scripts.manage_configs.report_timesteps import report_entry_point


# ---------------------------------------------------------------------------
# In-place YAML text editing  (preserves comments and formatting)
# ---------------------------------------------------------------------------

def set_top_level_key(content: str, key: str, new_val: str) -> str:
    """Update an existing top-level 'key: value' line, or insert before
    ego_train_algorithm: (or at EOF if that block is absent)."""
    pattern = rf"^({re.escape(key)}:[ \t]+)\S+"
    if re.search(pattern, content, flags=re.MULTILINE):
        return re.sub(pattern, rf"\g<1>{new_val}", content, flags=re.MULTILINE)
    insert_before = re.search(r"^ego_train_algorithm:", content, flags=re.MULTILINE)
    if insert_before:
        pos = insert_before.start()
        return content[:pos] + f"{key}: {new_val}\n" + content[pos:]
    return content.rstrip("\n") + f"\n{key}: {new_val}\n"


def set_nested_key(content: str, parent: str, child: str, new_val: str) -> str:
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


# ---------------------------------------------------------------------------
# Config path helper
# ---------------------------------------------------------------------------

def config_path_for_task(algo_configs_root: Path, algo: str, task: str) -> Path:
    parts = task.split("/")
    if len(parts) == 1:
        return algo_configs_root / algo / f"{task}.yaml"
    return algo_configs_root / algo / "/".join(parts[:-1]) / f"{parts[-1]}.yaml"


# ---------------------------------------------------------------------------
# Per-file update
# ---------------------------------------------------------------------------

def update_config(
    cfg_path: Path,
    new_params: dict[str, float | int],
    dry_run: bool,
) -> bool:
    """Write new_params into a config YAML, preserving comments.

    Returns True if any value actually changed.
    """
    content = cfg_path.read_text()
    changed: list[str] = []

    for key, py_val in new_params.items():
        new_str = str(py_val) if isinstance(py_val, int) else format_value(py_val)
        if "." in key:
            parent, child = key.split(".", 1)
            updated = set_nested_key(content, parent, child, new_str)
        else:
            updated = set_top_level_key(content, key, new_str)
        if updated != content:
            changed.append(f"{key}={new_str}")
            content = updated

    if not changed:
        return False
    if not dry_run:
        cfg_path.write_text(content)
    return True


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def process_entry_point(
    root: Path,
    easy_target: float | None,
    hard_target: float | None,
    skip_algos: set[str],
    dry_run: bool,
) -> None:
    algo_configs_root = root / "configs" / "algorithm"
    if not algo_configs_root.exists():
        print(f"  WARNING: algorithm configs not found: {algo_configs_root}", file=sys.stderr)
        return

    tier_targets: dict[str, float] = {}
    if easy_target is not None:
        for t in EASY_TASKS:
            tier_targets[t] = easy_target
    if hard_target is not None:
        for t in HARD_TASKS:
            tier_targets[t] = hard_target

    algos = sorted(
        d.name for d in algo_configs_root.iterdir()
        if d.is_dir() and d.name not in skip_algos
    )

    for algo in algos:
        algo_header_printed = False
        for task in ALL_TASKS:
            if task not in tier_targets:
                continue
            target = tier_targets[task]

            cfg_path = config_path_for_task(algo_configs_root, algo, task)
            if not cfg_path.exists():
                continue

            try:
                resolved = resolve_algo_config(cfg_path, algo_configs_root)
            except Exception as e:
                print(f"  [{algo}] {task}: ERROR resolving config — {e}")
                continue

            current_total = compute_total_timesteps(resolved)
            if current_total is None:
                print(f"  [{algo}] {task}: WARNING — could not compute current total; skipping")
                continue

            new_params = compute_target_params(resolved, target)
            if not new_params:
                print(f"  [{algo}] {task}: WARNING — no adjustable parameter found; skipping")
                continue

            # Simulate the resulting total for reporting.
            sim_config = dict(resolved)
            for k, v in new_params.items():
                if "." not in k:
                    sim_config[k] = v
            new_total = compute_total_timesteps(sim_config)

            old_h = format_human(current_total)
            new_h = format_human(new_total) if new_total is not None else "?"
            tgt_h = format_human(target)

            if not algo_header_printed:
                print(f"\n[{algo}]")
                algo_header_printed = True

            # Check if already at target (within 0.5%).
            if new_total is not None and abs(new_total - current_total) / max(target, 1) < 0.005:
                print(f"  {task}: {old_h}  (up to date, target={tgt_h})")
                continue

            param_strs = ", ".join(
                f"{k}={str(v) if isinstance(v, int) else format_value(v)}"
                for k, v in new_params.items()
            )
            prefix = "DRY RUN  " if dry_run else ""
            changed = update_config(cfg_path, new_params, dry_run)
            if changed or dry_run:
                print(f"  {task}: {old_h} → {new_h}  [{param_strs}]  {prefix}")
            else:
                print(f"  {task}: {old_h}  (no change written)")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "entry_point_dirs",
        nargs="+",
        metavar="entry_point_dir",
        help="One or more entry point directories (e.g. teammate_generation/).",
    )
    parser.add_argument(
        "--easy-target",
        metavar="N",
        help="Target total timesteps for easy tasks (lbf, cramped_room, asymm_advantages).",
    )
    parser.add_argument(
        "--hard-target",
        metavar="N",
        help="Target total timesteps for hard tasks (coord_ring, counter_circuit, forced_coord).",
    )
    parser.add_argument(
        "--skip-algos",
        nargs="+",
        metavar="ALGO",
        default=[],
        help="Algorithm names to skip (e.g. --skip-algos open_ended_minimax paired ppo_br).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change without writing any files.",
    )
    args = parser.parse_args()

    if args.easy_target is None and args.hard_target is None:
        parser.error("At least one of --easy-target or --hard-target must be specified.")

    easy_target = parse_human_timesteps(args.easy_target) if args.easy_target else None
    hard_target = parse_human_timesteps(args.hard_target) if args.hard_target else None
    skip_algos = set(args.skip_algos)

    if easy_target is not None:
        print(f"Easy target: {format_human(easy_target)}  ({easy_target:.3g})")
    if hard_target is not None:
        print(f"Hard target: {format_human(hard_target)}  ({hard_target:.3g})")
    if dry_run := args.dry_run:
        print("(dry run — no files will be written)")

    for ep_dir in args.entry_point_dirs:
        root = Path(ep_dir)
        print(f"\n{'=' * 60}")
        print(f"Entry point: {root}")
        print(f"{'=' * 60}")
        process_entry_point(root, easy_target, hard_target, skip_algos, dry_run)

    # Summary table using report_timesteps machinery.
    if not dry_run:
        print(f"\n{'=' * 60}")
        print("Summary (updated totals)")
        print(f"{'=' * 60}")
        for ep_dir in args.entry_point_dirs:
            root = Path(ep_dir)
            print(f"\nEntry point: {root}")
            report_entry_point(root, skip_algos=skip_algos if skip_algos else None)

    print("\nDone.")


if __name__ == "__main__":
    main()
