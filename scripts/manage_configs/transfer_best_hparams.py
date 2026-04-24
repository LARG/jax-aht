#!/usr/bin/env python3
"""
Transfer best hyperparameters from swept task configs to unswept target task configs.

Detection: the `parameters:` section of the param_sweep YAML identifies which
hyperparameters were swept. Their current values in the swept task's algorithm
config are taken as the selected best values.

Transfer: those best values are written into each target task's algorithm config,
updating existing keys in-place or inserting missing ones before the
`ego_train_algorithm:` block (or at end-of-file if that block is absent).

SWEPT_TASK -> TARGET_TASKS mapping is defined in TASK_MAP below.

Usage:
    python temp/transfer_best_hparams.py <entry_point_dir> [--dry-run] [--skip-algos ALGO ...]

Options:
    --dry-run               Print what would change without writing any files.
    --skip-algos ALGO ...   Skip one or more algorithms by name. Useful when a
                            sweep is still running and best values have not yet
                            been selected (e.g. --skip-algos comedi rotate).

Examples:
    python scripts/manage_configs/transfer_best_hparams.py teammate_generation/
    python scripts/manage_configs/transfer_best_hparams.py teammate_generation/ --skip-algos comedi
    python scripts/manage_configs/transfer_best_hparams.py open_ended_training/ --dry-run --skip-algos rotate
"""

import math
import re
import sys
import yaml
from pathlib import Path

# Mapping: swept task path -> list of target task paths
TASK_MAP: dict[str, list[str]] = {
    "overcooked-v1/coord_ring": [
        "overcooked-v1/asymm_advantages",
        "overcooked-v1/counter_circuit",
        "overcooked-v1/cramped_room",
        "overcooked-v1/forced_coord",
    ],
}


# ---------------------------------------------------------------------------
# Value formatting
# ---------------------------------------------------------------------------

def format_value(val) -> str:
    """Format a hyperparameter value for YAML output.

    Integers are written as integers (e.g. 10, 512).
    Floats < 0.01 or >= 1e5 use scientific notation (e.g. 1e-3, 1e8).
    Other floats use decimal form (e.g. 0.03, 0.5).
    """
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        if val == int(val) and abs(val) < 1e5:
            return str(int(val))
        if abs(val) < 0.01 or abs(val) >= 1e5:
            exp = math.floor(math.log10(abs(val)))
            mantissa = round(val / (10 ** exp), 10)
            if mantissa == int(mantissa):
                return f"{int(mantissa)}e{exp}"
            return f"{mantissa:.10g}e{exp}"
        return f"{val:g}"
    return str(val)


# ---------------------------------------------------------------------------
# Config resolution (shared with update_sweep_timesteps.py)
# ---------------------------------------------------------------------------

def resolve_algo_config(config_path: Path, algo_configs_root: Path) -> dict:
    """Load an algorithm config with its Hydra defaults chain resolved (shallow merge)."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    defaults = raw.get("defaults", [])
    merged: dict = {}

    for entry in defaults:
        if entry == "_self_":
            merged.update({k: v for k, v in raw.items() if k != "defaults"})
        else:
            ref = algo_configs_root / f"{entry}.yaml"
            if ref.exists():
                merged.update(resolve_algo_config(ref, algo_configs_root))

    if not defaults:
        merged = {k: v for k, v in raw.items() if k != "defaults"}

    return merged


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def find_sweep_file(param_sweep_dir: Path, algo: str, task: str) -> Path | None:
    """
    Locate the param_sweep YAML for a given algorithm + task.

    Naming convention:
      - Single-level task (e.g. 'lbf'):
            <param_sweep_dir>/<algo>/lbf/param_sweep.yml
      - Multi-level task (e.g. 'overcooked-v1/coord_ring'):
            <param_sweep_dir>/<algo>/overcooked-v1/coord_ring_param_sweep.yml
    """
    parts = task.split("/")
    if len(parts) == 1:
        candidate = param_sweep_dir / algo / task / "param_sweep.yml"
    else:
        task_dir = "/".join(parts[:-1])
        task_leaf = parts[-1]
        candidate = param_sweep_dir / algo / task_dir / f"{task_leaf}_param_sweep.yml"
    return candidate if candidate.exists() else None


def get_swept_param_keys(sweep_file: Path) -> list[str]:
    """
    Return swept parameter key paths from a param_sweep YAML, with the leading
    'algorithm.' prefix stripped.  e.g. ['PARTNER_POP_SIZE', 'LR', 'ENT_COEF']
    Dot-separated paths (e.g. 'ego_train_algorithm.LR') are preserved as-is.
    """
    with open(sweep_file) as f:
        sweep = yaml.safe_load(f)
    return [k.removeprefix("algorithm.") for k in sweep.get("parameters", {})]


def get_best_values(
    config_path: Path,
    algo_configs_root: Path,
    keys: list[str],
) -> dict[str, str]:
    """
    Return {key_path: formatted_value} for each key from the resolved algorithm config.
    Dot-separated key paths index into nested dicts (e.g. 'ego_train_algorithm.LR').
    """
    config = resolve_algo_config(config_path, algo_configs_root)
    best: dict[str, str] = {}

    for key in keys:
        val = config
        for part in key.split("."):
            if isinstance(val, dict) and part in val:
                val = val[part]
            else:
                print(f"    WARNING: key '{key}' not found in resolved config {config_path.name}")
                val = None
                break
        if val is not None:
            best[key] = format_value(val)

    return best


# ---------------------------------------------------------------------------
# In-place YAML text editing
# ---------------------------------------------------------------------------

def set_top_level_key(content: str, key: str, new_val: str) -> str:
    """
    Update an existing top-level 'key: value' line, or insert the key before
    the `ego_train_algorithm:` block (or at end-of-file if that block is absent).
    """
    pattern = rf"^({re.escape(key)}:[ \t]+)\S+"
    if re.search(pattern, content, flags=re.MULTILINE):
        return re.sub(pattern, rf"\g<1>{new_val}", content, flags=re.MULTILINE)

    # Key absent — find insertion point.
    insert_before = re.search(r"^ego_train_algorithm:", content, flags=re.MULTILINE)
    if insert_before:
        pos = insert_before.start()
        return content[:pos] + f"{key}: {new_val}\n" + content[pos:]
    return content.rstrip("\n") + f"\n{key}: {new_val}\n"


def set_nested_key(content: str, parent: str, child: str, new_val: str) -> str:
    """
    Update a nested 'child: value' line inside the named parent block.
    Tracks indentation to stay within the correct block.
    """
    lines = content.splitlines(keepends=True)
    new_lines: list[str] = []
    in_parent = False
    parent_indent = -1

    for line in lines:
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        if in_parent:
            if stripped and not stripped.startswith("#") and indent <= parent_indent:
                in_parent = False  # left the parent block
            else:
                m = re.match(rf"^([ \t]+{re.escape(child)}:[ \t]+)\S+", line)
                if m:
                    line = f"{m.group(1)}{new_val}\n"

        if re.match(rf"^{re.escape(parent)}[ \t]*:", line):
            in_parent = True
            parent_indent = indent

        new_lines.append(line)

    return "".join(new_lines)


def update_target_config(
    target_path: Path,
    best_values: dict[str, str],
    dry_run: bool = False,
) -> None:
    """
    Write best hyperparameter values into a target algorithm config YAML.
    Reports only the keys whose values actually change.
    """
    content = target_path.read_text()
    rel = target_path.relative_to(target_path.parents[3])
    changed_keys: list[str] = []

    for key, new_val in best_values.items():
        if "." in key:
            parent, child = key.split(".", 1)
            updated = set_nested_key(content, parent, child, new_val)
        else:
            updated = set_top_level_key(content, key, new_val)

        if updated != content:
            changed_keys.append(key)
            content = updated

    if not changed_keys:
        print(f"    SKIP (up to date): {rel}")
        return

    if dry_run:
        for key in changed_keys:
            print(f"    DRY RUN  {key}={best_values[key]}  →  {rel}")
        return

    target_path.write_text(content)
    for key in changed_keys:
        print(f"    Set {key}={best_values[key]}  →  {rel}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def process_entry_point(
    root: Path, dry_run: bool = False, skip_algos: set[str] | None = None
) -> None:
    param_sweep_dir = root / "param_sweep"
    algo_configs_root = root / "configs" / "algorithm"

    algos = sorted(d.name for d in algo_configs_root.iterdir() if d.is_dir())

    for swept_task, target_tasks in TASK_MAP.items():
        print(f"\n=== Swept task: {swept_task} ===")

        for algo in algos:
            if skip_algos and algo in skip_algos:
                print(f"\n  [{algo}]  skipped (--skip-algos)")
                continue

            sweep_file = find_sweep_file(param_sweep_dir, algo, swept_task)
            if sweep_file is None:
                print(f"\n  [{algo}]  no param_sweep file found; skipping")
                continue

            swept_keys = get_swept_param_keys(sweep_file)

            swept_config = algo_configs_root / algo / f"{swept_task}.yaml"
            if not swept_config.exists():
                print(f"\n  [{algo}]  algorithm config not found: {swept_config}; skipping")
                continue

            best_values = get_best_values(swept_config, algo_configs_root, swept_keys)

            print(f"\n  [{algo}]  swept params + best values:")
            for k, v in best_values.items():
                print(f"    {k} = {v}")

            for target_task in target_tasks:
                target_config = algo_configs_root / algo / f"{target_task}.yaml"
                if not target_config.exists():
                    print(f"    target config not found: {target_config}; skipping")
                    continue
                update_target_config(target_config, best_values, dry_run=dry_run)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "entry_point_dir",
        help="Root directory of the entry point, e.g. teammate_generation/",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change without writing any files.",
    )
    parser.add_argument(
        "--skip-algos",
        nargs="+",
        metavar="ALGO",
        default=[],
        help="Algorithm names to skip (e.g. --skip-algos comedi rotate).",
    )
    args = parser.parse_args()

    root = Path(args.entry_point_dir)
    for label, path in [
        ("param_sweep", root / "param_sweep"),
        ("algorithm configs", root / "configs" / "algorithm"),
    ]:
        if not path.exists():
            print(f"ERROR: {label} directory not found: {path}", file=sys.stderr)
            sys.exit(1)

    process_entry_point(root, dry_run=args.dry_run, skip_algos=set(args.skip_algos))
    print("\nDone.")


if __name__ == "__main__":
    main()
