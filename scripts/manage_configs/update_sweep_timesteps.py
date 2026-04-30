#!/usr/bin/env python3
"""
Transfer TOTAL_TIMESTEPS (or equivalent) from algorithm configs to param_sweep configs.

For each param_sweep YAML, this script:
  1. Parses the `algorithm=<path>` CLI arg in the command section.
  2. Resolves the corresponding algorithm config (respecting Hydra defaults inheritance).
  3. Extracts the effective TOTAL_TIMESTEPS (or TOTAL_TIMESTEPS_PER_ITERATION) value,
     and also ego_train_algorithm.TOTAL_TIMESTEPS if present.
  4. Inserts them as CLI args before `${args_no_hyphens}` in the command section.

File content is manipulated as text so that comments and formatting are preserved.

Usage:
    python scripts/manage_configs/update_sweep_timesteps.py <entry_point_dir> [--dry-run]

Examples:
    python scripts/manage_configs/update_sweep_timesteps.py teammate_generation/
    python scripts/manage_configs/update_sweep_timesteps.py ego_agent_training/
    python scripts/manage_configs/update_sweep_timesteps.py open_ended_training/
"""

import math
import sys
import yaml
from pathlib import Path

TIMESTEP_KEYS = [
        "TOTAL_TIMESTEPS", "TOTAL_TIMESTEPS_PER_ITERATION", 
        # ROTATE specific
        "TIMESTEPS_PER_ITER_PARTNER", "TIMESTEPS_PER_ITER_EGO", 
        "NUM_OPEN_ENDED_ITERS"
        ]


def format_timesteps(val: float) -> str:
    """Format a timestep value for use in CLI args.

    Values in [0.1, 100] are written as plain numbers (e.g. 9, 50, 0.5).
    Values outside that range use scientific notation (e.g. 1e8, 4.5e7, 6e6).
    """
    if val == 0:
        return "0"
    if 0.1 <= abs(val) <= 100:
        return str(int(val)) if val == int(val) else f"{val:g}"
    exp = math.floor(math.log10(abs(val)))
    mantissa = round(val / (10 ** exp), 10)
    if mantissa == int(mantissa):
        return f"{int(mantissa)}e{exp}"
    return f"{mantissa:.10g}e{exp}"


def resolve_algo_config(config_path: Path, algo_configs_root: Path) -> dict:
    """
    Load an algorithm config, merging values from its Hydra defaults chain.

    Defaults are resolved relative to algo_configs_root (the algorithm/ configs dir).
    Later entries in the defaults list (including _self_) override earlier ones.
    The @package directive in the YAML is a comment and is ignored.
    """
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
            else:
                print(f"  WARNING: referenced config not found: {ref}")

    if not defaults:
        # No defaults block — use the file's own keys directly.
        merged = {k: v for k, v in raw.items() if k != "defaults"}

    return merged


def get_timestep_args(sweep_file: Path, algo_configs_root: Path) -> list[str]:
    """
    Derive all timestep CLI arg strings from a param_sweep YAML file.

    Returns a list of strings such as:
        ["algorithm.TOTAL_TIMESTEPS=1e8",
         "algorithm.ego_train_algorithm.TOTAL_TIMESTEPS=3e7"]

    The ego_train_algorithm arg is included only when that key is present in the
    resolved config (i.e. teammate_generation entry point); other entry points
    that lack it are handled gracefully.

    Returns an empty list if the main timestep cannot be determined.
    """
    with open(sweep_file) as f:
        sweep = yaml.safe_load(f)

    command = sweep.get("command", [])
    algo_path = next(
        (s.split("=", 1)[1] for s in command if isinstance(s, str) and s.startswith("algorithm=")),
        None,
    )
    if algo_path is None:
        print(f"  WARNING: no 'algorithm=' arg found in command section of {sweep_file}")
        return []

    algo_config_path = algo_configs_root / f"{algo_path}.yaml"
    if not algo_config_path.exists():
        print(f"  WARNING: algorithm config not found: {algo_config_path}")
        return []

    config = resolve_algo_config(algo_config_path, algo_configs_root)
    args: list[str] = []

    # Main training timesteps — collect all matching keys (ROTATE has several).
    for key in TIMESTEP_KEYS:
        if key in config:
            args.append(f"algorithm.{key}={format_timesteps(float(config[key]))}")

    if not args:
        print(f"  WARNING: no timestep key ({', '.join(TIMESTEP_KEYS)}) found in resolved config for {sweep_file}")
        return []

    # Ego-agent training timesteps (present in teammate_generation algorithms).
    ego_cfg = config.get("ego_train_algorithm")
    if isinstance(ego_cfg, dict) and "TOTAL_TIMESTEPS" in ego_cfg:
        args.append(
            f"algorithm.ego_train_algorithm.TOTAL_TIMESTEPS="
            f"{format_timesteps(float(ego_cfg['TOTAL_TIMESTEPS']))}"
        )

    return args


def update_sweep_file(sweep_file: Path, algo_configs_root: Path, dry_run: bool = False) -> None:
    """
    Insert or update timestep CLI args in a param_sweep YAML command section.

    For each desired arg:
      - If the exact "key=value" string is already present: skip (idempotent).
      - If the key is present with a different value: replace the value in-place.
      - If the key is absent: insert before `${args_no_hyphens}`.

    File content is manipulated as text so that comments and formatting are preserved.
    """
    ts_args = get_timestep_args(sweep_file, algo_configs_root)
    if not ts_args:
        return

    content = sweep_file.read_text()
    rel_path = sweep_file.relative_to(sweep_file.parents[3])

    to_insert: list[str] = []
    for arg in ts_args:
        key, value = arg.split("=", 1)
        if arg in content:
            # Exact match — already up to date.
            print(f"  SKIP (up to date): {arg}  →  {rel_path}")
        elif key in content:
            # Key present but value differs — update in-place.
            if dry_run:
                print(f"  DRY RUN  update '{key}' to '{value}'  →  {rel_path}")
            else:
                import re
                content = re.sub(
                    rf"(- {re.escape(key)}=)\S+",
                    rf"\g<1>{value}",
                    content,
                )
                print(f"  Updated '{key}={value}'  →  {rel_path}")
        else:
            to_insert.append(arg)

    if to_insert:
        if dry_run:
            for a in to_insert:
                print(f"  DRY RUN  add '{a}'  →  {rel_path}")
        else:
            lines = content.splitlines(keepends=True)
            new_lines = []
            inserted = False
            for line in lines:
                if "${args_no_hyphens}" in line and not inserted:
                    indent = len(line) - len(line.lstrip())
                    for a in to_insert:
                        new_lines.append(f"{' ' * indent}- {a}\n")
                    inserted = True
                new_lines.append(line)

            if not inserted:
                print(f"  NOTE: '${{args_no_hyphens}}' not found in {sweep_file.name}; appending at end.")
                for a in to_insert:
                    new_lines.append(f"  - {a}\n")

            content = "".join(new_lines)
            for a in to_insert:
                print(f"  Added '{a}'  →  {rel_path}")

    if not dry_run:
        sweep_file.write_text(content)


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
    args = parser.parse_args()

    root = Path(args.entry_point_dir)
    param_sweep_dir = root / "param_sweep"
    algo_configs_root = root / "configs" / "algorithm"

    errors = []
    if not param_sweep_dir.exists():
        errors.append(f"param_sweep directory not found: {param_sweep_dir}")
    if not algo_configs_root.exists():
        errors.append(f"algorithm configs directory not found: {algo_configs_root}")
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    sweep_files = sorted(param_sweep_dir.rglob("*.yml")) + sorted(param_sweep_dir.rglob("*.yaml"))
    print(f"Found {len(sweep_files)} sweep file(s) under {param_sweep_dir}\n")

    for sweep_file in sweep_files:
        update_sweep_file(sweep_file, algo_configs_root, dry_run=args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
