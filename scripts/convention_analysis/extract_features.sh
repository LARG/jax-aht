#!/usr/bin/env bash
# Extract convention features for the Overcooked eval teammates of every layout (two GPU lanes).
# Usage: extract_features.sh <out_dir> <cache_dir> <br_root> [gpu_a] [gpu_b]
#   br_root: local copy of the eval-teammate BR checkpoints (see --br-root in
#   compute_population_diversity.py), e.g. a download of jaxaht/eval-teammates-br/overcooked.
set -uo pipefail
OUT="${1:?out_dir}"; CACHE="${2:?cache_dir}"; BR_ROOT="${3:?br_root}"; GPU_A="${4:-0}"; GPU_B="${5:-1}"
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$W"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PY="${PYTHON:-python}"
cd "$W"
mkdir -p "$OUT"
LANE_A="cramped_room counter_circuit coord_ring"
LANE_B="forced_coord asymm_advantages"
run_lane() {
  local gpu="$1" layouts="$2"
  for layout in $layouts; do
    echo "=== $layout gpu$gpu $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES=$gpu $PY scripts/population_diversity/compute_population_diversity.py \
      --env overcooked --variant "$layout" --full-heldout --br-paired --br-root "$BR_ROOT" \
      --num-episodes 128 --batched --deterministic-reset \
      --cache-dir "$CACHE/overcooked-$layout" \
      --output-dir "$OUT/pd_$layout" > "$OUT/extract_$layout.log" 2>&1
    echo "=== done $layout rc=$? $(date +%H:%M:%S)"
  done
}
run_lane "$GPU_A" "$LANE_A" &
run_lane "$GPU_B" "$LANE_B" &
wait
echo ALL DONE
