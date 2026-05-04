#!/bin/bash
# Download pretrained OBL R2D2 weights (Flax/safetensors format) from
# HuggingFace. Available levels: OBL1 through OBL5, each with 5 seeds
# (a-e) and 2 BZA variants (BZA0, BZA1). Total ~50 checkpoints.
#
# Usage:
#   bash agents/hanabi/download_obl_r2d2.sh           # OBL1 seed-a only (default)
#   bash agents/hanabi/download_obl_r2d2.sh all        # all levels, all seeds (~1GB)
#   bash agents/hanabi/download_obl_r2d2.sh 1 4        # OBL1 and OBL4 seed-a
#
# Downloads to agents/hanabi/obl-r2d2-flax/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$SCRIPT_DIR/obl-r2d2-flax"
BASE_URL="https://huggingface.co/mttga/obl-r2d2-flax/resolve/main"

download_checkpoint() {
    local level="$1"
    local seed="$2"
    local bza="${3:-0}"

    local dir_name="icml_OBL${level}"
    local load_flag=""
    if [ "$level" -gt 1 ]; then
        load_flag="_LOAD1"
    fi
    local filename="OFF_BELIEF1_SHUFFLE_COLOR0${load_flag}_BZA${bza}_BELIEF_${seed}.safetensors"
    local filepath="$DEST/$dir_name/$filename"
    local url="$BASE_URL/$dir_name/$filename"

    if [ -f "$filepath" ]; then
        local size
        size=$(stat -c%s "$filepath" 2>/dev/null || stat -f%z "$filepath" 2>/dev/null || echo 0)
        if [ "$size" -gt 1000 ]; then
            echo "[obl] already have OBL${level} seed-${seed} BZA${bza} (${size} bytes)"
            return 0
        fi
        rm -f "$filepath"
    fi

    mkdir -p "$DEST/$dir_name"
    echo "[obl] downloading OBL${level} seed-${seed} BZA${bza}..."
    if curl -sL -o "$filepath" "$url"; then
        local size
        size=$(stat -c%s "$filepath" 2>/dev/null || stat -f%z "$filepath" 2>/dev/null || echo 0)
        if [ "$size" -gt 1000 ]; then
            echo "[obl]   -> $filepath (${size} bytes)"
            return 0
        fi
    fi
    echo "[obl] WARNING: failed to download OBL${level} seed-${seed} BZA${bza}"
    rm -f "$filepath"
    return 1
}

if [ "${1:-}" = "all" ]; then
    echo "[obl] downloading ALL OBL levels (1-5), all seeds, BZA0 only..."
    for level in 1 2 3 4 5; do
        for seed in a b c d e; do
            download_checkpoint "$level" "$seed" 0
        done
    done
elif [ $# -gt 0 ]; then
    for level in "$@"; do
        echo "[obl] downloading OBL${level} seed-a BZA0..."
        download_checkpoint "$level" "a" 0
    done
else
    echo "[obl] downloading OBL1 seed-a BZA0 (default)..."
    download_checkpoint 1 "a" 0
fi

echo
echo "[obl] done. Available checkpoints:"
find "$DEST" -name "*.safetensors" -exec ls -lh {} \; 2>/dev/null | head -20
