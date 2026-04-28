#!/bin/bash
# Create stable mini-Hanabi IPPO held-out partners for the eval harness.
#
# Matches the LBF / Overcooked convention of storing checkpoints under
# eval_teammates/<task>/<algo>/<descriptor>/saved_train_run. Creates:
#
#   eval_teammates/hanabi/ippo/seed42/saved_train_run   -> seed-42 IPPO run
#   eval_teammates/hanabi/ippo/seed123/saved_train_run  -> seed-123 IPPO run
#
# Both are stable symlinks into results/hanabi/ippo/ so the yaml paths in
# evaluation/configs/global_heldout_settings.yaml don't need a timestamp.
#
# Uses mini-Hanabi (3c/3r) for fast training. Full Hanabi partners can be
# trained by overriding TASK=hanabi ALGO=ippo/hanabi.
#
# If a matching results/ directory already exists, the script just refreshes
# the symlink. No retraining.
#
# Usage:
#   PYTHONPATH=. bash scripts/regenerate_hanabi_eval_teammates.sh
#   PYTHONPATH=. bash scripts/regenerate_hanabi_eval_teammates.sh 42
#   PYTHONPATH=. bash scripts/regenerate_hanabi_eval_teammates.sh 42 123
#
# Default with no args: 42 and 123.
# Wall clock: ~1 minute per seed on one RTX 6000 Ada at 1e7 steps.

set -eo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

PYTHON="${PYTHON:-.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON="$(command -v python)"
    else
        echo "[regen-hanabi] ERROR: no python interpreter available."
        echo "[regen-hanabi]        set PYTHON=/path/to/python and try again."
        exit 1
    fi
fi

TASK="${TASK:-mini-hanabi}"
ALGO="${ALGO:-ippo/mini-hanabi}"

SEEDS=("$@")
if [ ${#SEEDS[@]} -eq 0 ]; then
    SEEDS=(42 123)
fi

mkdir -p eval_teammates/hanabi/ippo

ensure_partner() {
    local seed="$1"
    local label="hanabi_ippo_${TASK}_seed${seed}"
    local results_dir="results/hanabi/ippo/${label}"
    local link="eval_teammates/hanabi/ippo/seed${seed}"

    local latest=""
    if [ -d "$results_dir" ]; then
        latest="$(ls -1dt ${results_dir}/*/saved_train_run 2>/dev/null | head -n 1 || true)"
    fi

    if [ -z "$latest" ] || [ ! -d "$latest" ]; then
        echo "[regen-hanabi] no existing seed-${seed} training run; training a fresh one"
        "$PYTHON" marl/run.py task=$TASK algorithm=$ALGO \
            algorithm.SEED="$seed" \
            algorithm.NUM_SEEDS=1 \
            algorithm.TOTAL_TIMESTEPS=10000000 \
            label="$label" \
            logger.mode=disabled
        latest="$(ls -1dt ${results_dir}/*/saved_train_run 2>/dev/null | head -n 1 || true)"
    fi

    if [ -z "$latest" ] || [ ! -d "$latest" ]; then
        echo "[regen-hanabi] ERROR: failed to produce a saved_train_run for seed ${seed}"
        return 1
    fi

    local target_dir
    target_dir="$(dirname "$latest")"
    local rel_target="../../../${target_dir}"
    ln -sfn "$rel_target" "$link"
    echo "[regen-hanabi] seed-${seed}: ${link} -> ${rel_target}"
}

for seed in "${SEEDS[@]}"; do
    ensure_partner "$seed"
done

echo
echo "[regen-hanabi] done. Held-out partners:"
ls -la eval_teammates/hanabi/ippo/ 2>/dev/null || true
