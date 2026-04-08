#!/bin/bash
# run_dsse_scaling.sh - Benchmark JAX DSSE throughput scaling
#
# Tests: (1) NUM_ENVS scaling, (2) n_drones scaling, (3) grid_size scaling
#
# Usage: PYTHONPATH=. bash scripts/run_dsse_scaling.sh

set -eo pipefail

echo "=== JAX DSSE Scaling Benchmark ==="
echo ""

STEPS=50000

# ─── Test 1: Vectorized environment scaling ───
echo "--- NUM_ENVS scaling (how many parallel envs) ---"
for nenvs in 1 4 16 64 128 256 512; do
    echo -n "  NUM_ENVS=$nenvs: "
    start=$(date +%s%N)
    python marl/run.py task=dsse \
        algorithm.TOTAL_TIMESTEPS=$STEPS \
        algorithm.NUM_ENVS=$nenvs \
        algorithm.SEED=42 \
        label=scale_envs_${nenvs} \
        logger.mode=disabled \
        2>/dev/null
    end=$(date +%s%N)
    elapsed=$(( (end - start) / 1000000 ))
    echo "${elapsed}ms ($(( STEPS * 1000 / elapsed )) steps/sec)"
done

echo ""

# ─── Test 2: Agent count scaling ───
echo "--- n_drones scaling (team size) ---"
for ndrones in 2 4 6 8 10; do
    echo -n "  n_drones=$ndrones: "
    start=$(date +%s%N)
    python marl/run.py task=dsse \
        algorithm.TOTAL_TIMESTEPS=$STEPS \
        algorithm.NUM_ENVS=64 \
        algorithm.SEED=42 \
        task.ENV_KWARGS.n_drones=$ndrones \
        label=scale_drones_${ndrones} \
        logger.mode=disabled \
        2>/dev/null
    end=$(date +%s%N)
    elapsed=$(( (end - start) / 1000000 ))
    echo "${elapsed}ms"
done

echo ""

# ─── Test 3: Grid size scaling ───
echo "--- grid_size scaling ---"
for gs in 7 10 15 20 25; do
    echo -n "  grid_size=$gs: "
    start=$(date +%s%N)
    python marl/run.py task=dsse \
        algorithm.TOTAL_TIMESTEPS=$STEPS \
        algorithm.NUM_ENVS=64 \
        algorithm.SEED=42 \
        task.ENV_KWARGS.grid_size=$gs \
        label=scale_grid_${gs} \
        logger.mode=disabled \
        2>/dev/null
    end=$(date +%s%N)
    elapsed=$(( (end - start) / 1000000 ))
    echo "${elapsed}ms"
done

echo ""
echo "=== Done ==="
