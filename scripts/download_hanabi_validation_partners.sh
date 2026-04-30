#!/usr/bin/env bash
# Download mini-Hanabi validation partners from HuggingFace into the
# val_teammates/hanabi/ layout that global_validation_settings.yaml expects.
#
# HF source:
#   lainwired/hanabi-aht-partners/mini_hanabi/validation_partners/<algo>/
# Local target:
#   val_teammates/hanabi/<algo>/default_label/saved_train_run/
#
# Each <algo>/ on HF is an Orbax checkpoint root (contains _CHECKPOINT_METADATA,
# _METADATA, manifest.ocdbt, d/, ocdbt.process_0/). This script flattens the
# HF prefix and wraps each in default_label/saved_train_run/ to match the
# yaml path convention used by LBF and Overcooked.
#
# Usage:
#   bash scripts/download_hanabi_validation_partners.sh
#
# Requirements:
#   - huggingface-cli on PATH (pip install huggingface_hub)
#   - HF_TOKEN env var or `huggingface-cli login` if the repo is private

set -euo pipefail

REPO="lainwired/hanabi-aht-partners"
DEST="val_teammates/hanabi"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

mkdir -p "$DEST"

# Map: HF subdirectory name → algo name in global_validation_settings.yaml.
# All names match 1:1 between HF and the yaml.
declare -A ALGOS=(
  [brdiv]=brdiv
  [comedi]=comedi
  [lbrdiv]=lbrdiv
  [trajedi]=trajedi
  [ippo_mlp]=ippo_mlp
)

for hf_name in "${!ALGOS[@]}"; do
  algo="${ALGOS[$hf_name]}"
  target="$DEST/$algo/default_label/saved_train_run"

  if [ -f "$target/_CHECKPOINT_METADATA" ]; then
    echo "[skip] $algo already downloaded at $target"
    continue
  fi

  echo "[download] $hf_name -> $target"
  mkdir -p "$target"
  huggingface-cli download "$REPO" \
    --local-dir "$target" \
    --include "mini_hanabi/validation_partners/$hf_name/*"

  # huggingface-cli preserves the source prefix, so files land at
  # $target/mini_hanabi/validation_partners/$hf_name/*. Flatten that.
  src_prefix="$target/mini_hanabi/validation_partners/$hf_name"
  if [ -d "$src_prefix" ]; then
    mv "$src_prefix"/* "$target/"
    rm -rf "$target/mini_hanabi"
  fi
done

echo
echo "Done. Verifying layout:"
for algo in brdiv comedi lbrdiv trajedi ippo_mlp; do
  if [ -f "$DEST/$algo/default_label/saved_train_run/_CHECKPOINT_METADATA" ]; then
    echo "  OK   $DEST/$algo/default_label/saved_train_run/"
  else
    echo "  MISS $DEST/$algo/default_label/saved_train_run/"
  fi
done
