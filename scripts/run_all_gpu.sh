#!/bin/bash
# run_all_gpu.sh - Run ALL DSSE JAX experiments on GPU
#
# Runs: (1) heuristic-policy metrics, (2) episode GIF, (3) scaling benchmark, (4) MARL training
#
# Usage:
#   cd jax-aht
#   PYTHONPATH=. bash scripts/run_all_gpu.sh
#
# Estimated total: ~2-3 hours on GPU

set -eo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="."

echo "=== DSSE JAX-AHT: Full GPU Experiment Suite ==="
echo "Started: $(date)"
echo ""

# ─── 1. Metrics benchmark (heuristic policies) ───
echo "--- 1/4: DSSE Metrics (coverage, overlap, coordination) ---"
.venv/bin/python scripts/dsse_metrics.py --episodes 200 --output results/dsse_metrics
echo ""

# ─── 2. Episode visualization ───
echo "--- 2/4: Episode GIF ---"
MPLBACKEND=Agg .venv/bin/python scripts/visualize_dsse_episode.py --output results/dsse_episode.gif --policy greedy --seed 42
echo ""

# ─── 3. JAX Scaling benchmark ───
echo "--- 3/4: JAX Scaling ---"
echo "NUM_ENVS scaling:"
for nenvs in 1 4 16 64 128 256; do
    echo -n "  NUM_ENVS=$nenvs: "
    start=$(.venv/bin/python -c "import time; print(int(time.time()*1000))")
    .venv/bin/python marl/run.py task=dsse \
        algorithm.TOTAL_TIMESTEPS=50000 \
        algorithm.NUM_ENVS=$nenvs \
        algorithm.SEED=42 \
        task.ENV_KWARGS.grid_size=7 \
        +task.ENV_KWARGS.n_drones_to_rescue=2 \
        label=scale_${nenvs} \
        logger.mode=disabled \
        2>/dev/null
    end=$(.venv/bin/python -c "import time; print(int(time.time()*1000))")
    elapsed=$((end - start))
    echo "${elapsed}ms"
done

echo ""
echo "n_drones scaling:"
for nd in 2 4 6 8; do
    echo -n "  n_drones=$nd: "
    start=$(.venv/bin/python -c "import time; print(int(time.time()*1000))")
    .venv/bin/python marl/run.py task=dsse \
        algorithm.TOTAL_TIMESTEPS=50000 \
        algorithm.NUM_ENVS=64 \
        algorithm.SEED=42 \
        task.ENV_KWARGS.grid_size=7 \
        task.ENV_KWARGS.n_drones=$nd \
        +task.ENV_KWARGS.n_drones_to_rescue=2 \
        label=drones_${nd} \
        logger.mode=disabled \
        2>/dev/null
    end=$(.venv/bin/python -c "import time; print(int(time.time()*1000))")
    elapsed=$((end - start))
    echo "${elapsed}ms"
done

echo ""

# ─── 4. MARL Training (3 seeds, 500K steps) ───
echo "--- 4/4: MARL IPPO Training (500K steps, 3 seeds, 7x7, ndr=2) ---"
for seed in 42 123 456; do
    echo -n "  IPPO seed=$seed: "
    start=$(.venv/bin/python -c "import time; print(int(time.time()*1000))")
    .venv/bin/python marl/run.py task=dsse \
        algorithm.TOTAL_TIMESTEPS=500000 \
        algorithm.NUM_ENVS=64 \
        algorithm.SEED=$seed \
        task.ENV_KWARGS.grid_size=7 \
        task.ENV_KWARGS.n_drones=2 \
        task.ENV_KWARGS.n_targets=1 \
        +task.ENV_KWARGS.n_drones_to_rescue=2 \
        label=marl_ippo_7x7_staghunt_${seed} \
        logger.mode=disabled \
        2>/dev/null
    end=$(.venv/bin/python -c "import time; print(int(time.time()*1000))")
    elapsed=$((end - start))
    echo "${elapsed}ms"
done

echo ""
echo "=== ALL DONE ==="
echo "Results in: results/"
echo "  results/dsse_metrics/    - coverage, overlap, coordination plots"
echo "  results/dsse_episode.gif - animated episode"
echo "  stdout above             - scaling benchmark numbers"
echo "Finished: $(date)"
