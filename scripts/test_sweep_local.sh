#!/usr/bin/env bash
# Local smoke test for the wandb sweep wiring (no slurm, no TACC).
#
# Validates everything launch_ls6_sweep.sh does EXCEPT the sbatch
# scheduling. Useful for catching shell errors, env var issues, and
# Hydra config-compose problems before paying for TACC time.
#
# What this checks (in order):
#   1. shellcheck on launch_ls6_sweep.sh (if shellcheck installed)
#   2. bash -n syntax check
#   3. Hydra config compose for base_config_teammate (no actual training)
#   4. Validation set yaml parses and lists the expected partners
#   5. (optional) Single-config dry-run via wandb agent --count 1
#      with TOTAL_TIMESTEPS=1e4 — only if RUN_AGENT=1 is set
#
# Usage:
#   bash scripts/test_sweep_local.sh                  # checks 1-4
#   RUN_AGENT=1 SWEEP_ID=... bash scripts/test_sweep_local.sh  # adds 5

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SLURM_SCRIPT="$REPO_ROOT/scripts/launch_ls6_sweep.sh"
PASS=0
FAIL=0

ok() { echo "  PASS: $1"; PASS=$((PASS+1)); }
fail() { echo "  FAIL: $1"; FAIL=$((FAIL+1)); }

# 1. shellcheck
echo "=== 1. shellcheck ==="
if command -v shellcheck >/dev/null 2>&1; then
    if shellcheck "$SLURM_SCRIPT"; then
        ok "shellcheck clean on launch_ls6_sweep.sh"
    else
        fail "shellcheck found issues in launch_ls6_sweep.sh"
    fi
else
    echo "  SKIP: shellcheck not installed (brew install shellcheck)"
fi

# 2. bash syntax
echo
echo "=== 2. bash -n syntax check ==="
if bash -n "$SLURM_SCRIPT"; then
    ok "bash -n on launch_ls6_sweep.sh"
else
    fail "bash -n found syntax errors"
fi

# 3. Hydra config compose
echo
echo "=== 3. Hydra config compose for base_config_teammate (validation routing) ==="
cd "$REPO_ROOT"
if PYTHONPATH="$REPO_ROOT" python3 teammate_generation/run.py \
        task=mini-hanabi \
        algorithm=brdiv/mini-hanabi \
        --cfg job >/dev/null 2>&1; then
    ok "hydra composes base_config_teammate + mini-hanabi + brdiv"
else
    fail "hydra compose failed; re-run without --cfg=job to see the error"
fi

# 4. Validation set yaml has expected mini-hanabi partners
echo
echo "=== 4. Validation set contents ==="
python3 - <<'PYEOF'
import sys
import yaml
with open("evaluation/configs/global_validation_settings.yaml") as f:
    cfg = yaml.safe_load(f)
mini = cfg.get("heldout_set", {}).get("mini-hanabi")
if mini is None:
    print("  FAIL: heldout_set.mini-hanabi missing from global_validation_settings.yaml")
    sys.exit(1)
expected = {"ippo-mlp", "lbrdiv-conf", "comedi", "trajedi"}
unexpected = {"brdiv-conf"}
missing = expected - set(mini.keys())
present_unexpected = unexpected & set(mini.keys())
if missing:
    print(f"  FAIL: missing partners {missing}")
    sys.exit(1)
if present_unexpected:
    print(f"  FAIL: brdiv-conf should have been dropped, still present")
    sys.exit(1)
print(f"  PASS: mini-hanabi has the right partners ({sorted(mini.keys())})")
PYEOF
if [ $? -eq 0 ]; then PASS=$((PASS+1)); else FAIL=$((FAIL+1)); fi

# 5. (optional) Single sweep agent run
if [ "${RUN_AGENT:-0}" = "1" ]; then
    echo
    echo "=== 5. Single sweep agent dry-run (RUN_AGENT=1) ==="
    if [ -z "${SWEEP_ID:-}" ]; then
        echo "  SKIP: SWEEP_ID env var required for this test"
    else
        export PYTHONPATH="$REPO_ROOT"
        export XLA_PYTHON_CLIENT_PREALLOCATE="false"
        if wandb agent "$SWEEP_ID" --count 1; then
            ok "wandb agent ran one sweep config"
        else
            fail "wandb agent failed"
        fi
    fi
fi

echo
echo "=== Summary: $PASS passed, $FAIL failed ==="
exit "$FAIL"
