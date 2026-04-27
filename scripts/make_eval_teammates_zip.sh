#!/usr/bin/env bash
set -euo pipefail

# Create a timestamped zip archive for the eval_teammates directory.
# Usage:
#   bash scripts/make_eval_teammates_zip.sh
#   bash scripts/make_eval_teammates_zip.sh --source eval_teammates --out-dir results/releases

SOURCE_DIR="eval_teammates"
OUT_DIR="results/releases"
PREFIX="eval_teammates"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      SOURCE_DIR="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --prefix)
      PREFIX="$2"
      shift 2
      ;;
    -h|--help)
      cat <<EOF
Create a timestamped zip archive for teammate data.

Options:
  --source <dir>   Source directory to package (default: eval_teammates)
  --out-dir <dir>  Output directory for zip/checksum (default: results/releases)
  --prefix <name>  Zip filename prefix (default: eval_teammates)
  -h, --help       Show this help
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

if ! command -v zip >/dev/null 2>&1; then
  echo "Error: 'zip' command not found. Install zip and rerun." >&2
  exit 1
fi

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "Error: source directory not found: $SOURCE_DIR" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

TS="$(date -u +%Y%m%d_%H%M%S)"
ZIP_PATH="${OUT_DIR}/${PREFIX}_${TS}.zip"

# Use the parent dir so the zip stores a clean top-level folder path.
SOURCE_PARENT="$(dirname "$SOURCE_DIR")"
SOURCE_BASENAME="$(basename "$SOURCE_DIR")"

(
  cd "$SOURCE_PARENT"
  zip -r "$OLDPWD/$ZIP_PATH" "$SOURCE_BASENAME" \
    -x "*/__pycache__/*" "*/.DS_Store"
)

if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "$ZIP_PATH" > "${ZIP_PATH}.sha256"
  echo "Wrote checksum: ${ZIP_PATH}.sha256"
fi

echo "Created zip: $ZIP_PATH"

REMOTE_USER="$(whoami)"
REMOTE_HOST_FQDN="$(hostname -f 2>/dev/null || hostname)"
REMOTE_HOST_SHORT="$(hostname 2>/dev/null || echo remote-host)"

echo
echo "Copy to local machine examples:"
echo "  scp ${REMOTE_USER}@${REMOTE_HOST_FQDN}:$PWD/${ZIP_PATH} ."
echo "  scp ${REMOTE_USER}@${REMOTE_HOST_SHORT}:$PWD/${ZIP_PATH} ."

if [[ -f "${ZIP_PATH}.sha256" ]]; then
  echo "  scp ${REMOTE_USER}@${REMOTE_HOST_FQDN}:$PWD/${ZIP_PATH}.sha256 ."
  echo "  scp ${REMOTE_USER}@${REMOTE_HOST_SHORT}:$PWD/${ZIP_PATH}.sha256 ."
fi
