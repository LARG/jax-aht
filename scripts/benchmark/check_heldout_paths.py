"""Preflight checks for benchmark experiments.

Run before any training to catch two classes of misconfiguration early:

1. Every entry point's base config must compose
   evaluation/configs/global_heldout_settings (NOT global_validation_settings,
   which the hparam_search branch uses). Benchmark numbers must be reported
   against the heldout set.
2. Every checkpoint path referenced by the heldout set for the tasks being run
   must exist on this machine. Otherwise training completes and then dies in
   heldout evaluation, wasting the whole run.

Usage:
    PYTHONPATH=. python scripts/benchmark/check_heldout_paths.py \
        --tasks lbf/lbf_7x7_nolevels mini-hanabi
"""

import argparse
import os
import sys

from omegaconf import OmegaConf

REPO_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HELDOUT_SETTINGS = "evaluation/configs/global_heldout_settings"

BASE_CONFIGS = {
    "open_ended_training": "open_ended_training/configs/base_config_oel.yaml",
    "ego_agent_training": "ego_agent_training/configs/base_config_ego.yaml",
    "teammate_generation": "teammate_generation/configs/base_config_teammate.yaml",
}


def check_base_configs(entry_points):
    """Confirm each base config's defaults list pulls in global_heldout_settings."""
    errors = []
    for entry_point in entry_points:
        rel = BASE_CONFIGS[entry_point]
        cfg = OmegaConf.load(os.path.join(REPO_PATH, rel))
        defaults = [str(d) for d in cfg.get("defaults", [])]
        # Defaults are written relative to the config dir, e.g.
        # "../../evaluation/configs/global_heldout_settings".
        matched = [d for d in defaults if d.endswith(os.path.basename(HELDOUT_SETTINGS))]
        if matched:
            print(f"  OK   {rel} -> {matched[0]}")
        else:
            validation = [d for d in defaults if "global_validation_settings" in d]
            detail = f" (found {validation[0]!r})" if validation else ""
            errors.append(
                f"{rel} does not compose {HELDOUT_SETTINGS}{detail}"
            )
            print(f"  FAIL {rel}{detail}")
    return errors


def check_heldout_paths(tasks):
    """Confirm every checkpoint path used by the given tasks exists locally."""
    settings = OmegaConf.load(os.path.join(REPO_PATH, HELDOUT_SETTINGS + ".yaml"))
    heldout_set = settings.get("heldout_set", {})
    errors = []
    for task in tasks:
        if task not in heldout_set:
            errors.append(f"no heldout_set entry for task {task!r}")
            print(f"  FAIL {task}: no heldout_set entry")
            continue
        missing = []
        checked = 0
        for agent_name, agent_config in (heldout_set[task] or {}).items():
            # Entries may be nulled out to drop an inherited agent.
            if agent_config is None or "path" not in agent_config:
                continue
            path = agent_config["path"]
            resolved = path if os.path.isabs(path) else os.path.join(REPO_PATH, path)
            checked += 1
            if not os.path.exists(resolved):
                missing.append((agent_name, path))
        for agent_name, path in missing:
            errors.append(f"{task}: {agent_name} -> missing {path}")
        status = "FAIL" if missing else "OK  "
        print(f"  {status} {task}: {checked - len(missing)}/{checked} paths present")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", required=True,
                        help="Task names as used in the heldout_set config.")
    parser.add_argument("--entry-points", nargs="+", default=sorted(BASE_CONFIGS),
                        choices=sorted(BASE_CONFIGS),
                        help="Entry points whose base configs to check.")
    args = parser.parse_args()

    print("Checking base configs target the heldout settings...")
    errors = check_base_configs(args.entry_points)
    print("Checking heldout evaluation teammate paths exist locally...")
    errors += check_heldout_paths(args.tasks)

    if errors:
        print("\nPreflight FAILED:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    print("\nPreflight passed.")


if __name__ == "__main__":
    main()
