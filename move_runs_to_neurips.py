"""
Move the 440 paper-relevant runs from aht-benchmark → NEURIPS project.

Usage:
    python move_runs_to_neurips.py [--dry-run]

--dry-run: Just prints which runs would be moved, without actually moving them.
"""

import sys
import wandb
from wandb.apis import public as wandb_public

DRY_RUN = "--dry-run" in sys.argv

ENTITY = "jeffreychen287-the-university-of-texas-at-austin"
SRC_PROJECT = "aht-benchmark"
DST_PROJECT = "NEURIPS"

# The 440 paper-relevant label prefixes
# (these match exactly the label field in wandb config)
PAPER_LABEL_PREFIXES = [
    # CREPPO same-seed (80 runs)
    "creppo_no_law_",
    "creppo_law_0.0_",
    "creppo_law_0.1_",
    "creppo_law_0.2_",
    # IPPO baseline (80 runs — use canonical 5 seeds only)
    "ippo_no_law_",
    "ippo_law_0.0_",
    "ippo_law_0.1_",
    "ippo_law_0.2_",
    # MAPPO baseline (80 runs)
    "mappo_no_law_",
    "mappo_law_0.0_",
    "mappo_law_0.1_",
    "mappo_law_0.2_",
    # PPO baseline (80 runs)
    "ppo_no_law_",
    "ppo_law_0.0_",
    "ppo_law_0.1_",
    "ppo_law_0.2_",
    # IPPO curriculum (60 runs)
    "ippo_curriculum_law_0.0_",
    "ippo_curriculum_law_0.1_",
    "ippo_curriculum_law_0.2_",
    # MAPPO curriculum (60 runs)
    "mappo_curriculum_law_0.0_",
    "mappo_curriculum_law_0.1_",
    "mappo_curriculum_law_0.2_",
]

# Canonical eval seeds — only include runs with these seeds
# (excludes the extra seed=42 and timestamp seeds)
CANONICAL_SEEDS = {72128, 721280, 721281, 721282, 721283}

def is_paper_run(run):
    """Return True if this run belongs in the paper."""
    label = run.config.get("label", "")
    if not any(label.startswith(pfx) for pfx in PAPER_LABEL_PREFIXES):
        return False

    # Check seed — accept runs where TRAIN_SEED is a canonical seed
    seed = run.config.get("TRAIN_SEED")
    if seed is None:
        alg = run.config.get("algorithm", {})
        if isinstance(alg, dict):
            seed = alg.get("TRAIN_SEED")
    try:
        seed = int(seed)
    except (TypeError, ValueError):
        return False

    return seed in CANONICAL_SEEDS


def move_run(api, run, dst_project):
    """Move a single run to dst_project using WandB's internal GraphQL mutation."""
    mutation = """
    mutation MoveRun($runName: String!, $entityName: String!, $projectName: String!, $destinationProject: String!) {
      moveRun(input: {
        runName: $runName,
        entityName: $entityName,
        projectName: $projectName,
        destinationProjectName: $destinationProject,
        destinationEntityName: $entityName
      }) {
        task {
          id
        }
      }
    }
    """
    variables = {
        "runName": run.id,
        "entityName": ENTITY,
        "projectName": SRC_PROJECT,
        "destinationProject": dst_project,
    }
    try:
        result = api.client.execute(mutation, variables)
        return True
    except Exception as e:
        print(f"  ERROR moving {run.id}: {e}")
        return False


def main():
    api = wandb.Api(timeout=120)

    print(f"Fetching runs from {ENTITY}/{SRC_PROJECT} ...")
    all_runs = api.runs(f"{ENTITY}/{SRC_PROJECT}", per_page=1000)

    paper_runs = [r for r in all_runs if is_paper_run(r)]

    print(f"Found {len(paper_runs)} paper-relevant runs to move to '{DST_PROJECT}'")
    if DRY_RUN:
        print("\n[DRY RUN — no changes made]\n")
        for r in sorted(paper_runs, key=lambda r: r.config.get("label", "")):
            seed = r.config.get("TRAIN_SEED") or (r.config.get("algorithm") or {}).get("TRAIN_SEED")
            print(f"  {r.config.get('label', '?'):50s}  seed={seed}  id={r.id}")
        return

    moved = 0
    failed = 0
    for i, run in enumerate(paper_runs):
        label = run.config.get("label", "?")
        print(f"[{i+1}/{len(paper_runs)}] Moving {label} (id={run.id}) ...", end=" ", flush=True)
        if move_run(api, run, DST_PROJECT):
            print("✓")
            moved += 1
        else:
            print("✗")
            failed += 1

    print(f"\nDone. Moved: {moved}  Failed: {failed}")


if __name__ == "__main__":
    main()
