#!/bin/bash
# regenerate_partner_fixtures.sh -- create the stable IPPO partner fixtures
# referenced by Phase A/B and the held-out evaluation harness on a fresh
# checkout.
#
# This script is the reproducibility entry point for two artefacts that
# Phase A, Phase B, and the held-out eval YAML all depend on:
#
#   evaluation/fixtures/dsse_ippo_seed42_pop3   ->  seed-42 IPPO training run
#   evaluation/fixtures/dsse_ippo_seed123_pop3  ->  seed-123 IPPO training run
#
# Both targets are stable directories (symlinks into results/dsse/ippo/...)
# that the phase scripts and evaluation/configs/global_heldout_settings.yaml
# can reference without pinning a local timestamp.
#
# Behaviour:
#   1. If a results/dsse/ippo/marl_ippo_7x7_2drone_staghunt_seed{N}/<timestamp>/
#      directory already exists, the script just refreshes the fixture
#      symlink to point at the most recent one. No retraining.
#   2. If no such directory exists, the script trains a fresh seed-{N} IPPO
#      partner via marl/run.py at the same 2-drone / 1-target / ndr=2
#      configuration the phase A/B/D ego training uses (the teammate-
#      generation pipeline inherits num_agents == 2 from the ego trainer,
#      so partner pools have to be 2-drone to match). Overrides
#      n_drones=2 and n_targets=1 explicitly because the default task
#      config marl/configs/task/dsse.yaml uses a 4-drone setup intended
#      for the throughput benchmark, not for partner pools.
#
# Usage:
#   PYTHONPATH=. bash scripts/regenerate_partner_fixtures.sh
#   PYTHONPATH=. bash scripts/regenerate_partner_fixtures.sh 42      # only seed 42
#   PYTHONPATH=. bash scripts/regenerate_partner_fixtures.sh 42 123  # both
#
# Default with no args: 42 and 123.
#
# Wall clock: ~6 minutes per seed at NUM_ENVS=64 on a single RTX 6000 Ada
# (only when retraining is required; the symlink-only path is instant).

set -eo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

# Pick a python interpreter. Defaults to .venv/bin/python (the layout
# the lovelace GPU node uses), but the user can override by setting
# PYTHON in the environment, e.g. for conda or system python:
#   PYTHON=$(which python) bash scripts/regenerate_partner_fixtures.sh
PYTHON="${PYTHON:-.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON="$(command -v python)"
    else
        echo "[regen] ERROR: no python interpreter available."
        echo "[regen]        set PYTHON=/path/to/python and try again."
        exit 1
    fi
fi

SEEDS=("$@")
if [ ${#SEEDS[@]} -eq 0 ]; then
    SEEDS=(42 123)
fi

mkdir -p evaluation/fixtures

ensure_partner() {
    local seed="$1"
    local label="marl_ippo_7x7_2drone_staghunt_seed${seed}"
    local results_dir="results/dsse/ippo/${label}"
    local fixture="evaluation/fixtures/dsse_ippo_seed${seed}_pop3"

    local latest=""
    if [ -d "$results_dir" ]; then
        latest="$(ls -1dt ${results_dir}/*/saved_train_run 2>/dev/null | head -n 1 || true)"
    fi

    if [ -z "$latest" ] || [ ! -d "$latest" ]; then
        echo "[regen] no existing seed-${seed} training run; training a fresh one"
        # The default marl task config is 4-drone (intended for the
        # throughput benchmark). Phase A/B/D ego training expects
        # 2-drone partners because the JaxAHT ego pipeline asserts
        # num_agents == 2, and the observation space depends on
        # n_drones, so a 4-drone partner policy is not loadable into
        # a 2-drone ego env. Force n_drones=2, n_targets=1 here.
        "$PYTHON" marl/run.py task=dsse algorithm=ippo/dsse \
            algorithm.SEED="$seed" \
            task.ENV_KWARGS.n_drones=2 \
            task.ENV_KWARGS.n_targets=1 \
            label="$label" \
            logger.mode=offline
        latest="$(ls -1dt ${results_dir}/*/saved_train_run 2>/dev/null | head -n 1 || true)"
    fi

    if [ -z "$latest" ] || [ ! -d "$latest" ]; then
        echo "[regen] ERROR: failed to produce a saved_train_run for seed ${seed}"
        return 1
    fi

    # The fixture target is the timestamp directory; the consumer paths
    # all append /saved_train_run themselves.
    local target_dir
    target_dir="$(dirname "$latest")"
    # Convert results/... to ../../results/... so the symlink resolves
    # from inside evaluation/fixtures/.
    local rel_target="../../${target_dir}"
    ln -sfn "$rel_target" "$fixture"
    echo "[regen] seed-${seed}: ${fixture} -> ${rel_target}"
}

for seed in "${SEEDS[@]}"; do
    ensure_partner "$seed"
done

echo
echo "[regen] done. Fixtures:"
ls -la evaluation/fixtures/ | grep '^l' || true
