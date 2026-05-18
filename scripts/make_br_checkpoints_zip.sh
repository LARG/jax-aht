#!/usr/bin/env bash
set -euo pipefail

# Create a timestamped zip archive of BR checkpoint directories listed in a
# best_response_set config (default: evaluation/configs/global_heldout_br.yaml).
#
# Examples:
#   ./scripts/make_br_checkpoints_zip.sh
#   ./scripts/make_br_checkpoints_zip.sh --task lbf
#   ./scripts/make_br_checkpoints_zip.sh --task overcooked-v1/forced_coord
#   ./scripts/make_br_checkpoints_zip.sh --config evaluation/configs/global_heldout_br_generated.yaml
#   ./scripts/make_br_checkpoints_zip.sh --dry-run

CONFIG_PATH="evaluation/configs/global_heldout_br.yaml"
TASK_FILTER="all"
OUT_DIR="results/releases"
PREFIX="br_checkpoints_from_global_heldout_br_all"
DRY_RUN="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --task)
      TASK_FILTER="$2"
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
    --dry-run)
      DRY_RUN="true"
      shift
      ;;
    -h|--help)
      cat <<EOF
Create a timestamped zip archive of BR checkpoints from best_response_set config.

Options:
  --config <path>   Config with best_response_set (default: evaluation/configs/global_heldout_br.yaml)
  --task <name>     Task filter (default: all)
  --out-dir <dir>   Output directory for zip/checksum (default: results/releases)
  --prefix <name>   Zip filename prefix (default: br_checkpoints_from_global_heldout_br_all)
  --dry-run         Print included/missing paths without creating zip
  -h, --help        Show help
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

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ ! -f "${REPO_ROOT}/${CONFIG_PATH}" ]]; then
  echo "Error: config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

mkdir -p "${REPO_ROOT}/${OUT_DIR}"

TMP_INCLUDE="$(mktemp)"
TMP_MISSING="$(mktemp)"
trap 'rm -f "$TMP_INCLUDE" "$TMP_MISSING"' EXIT

python3 - <<'PY' "${REPO_ROOT}" "${CONFIG_PATH}" "${TASK_FILTER}" "${TMP_INCLUDE}" "${TMP_MISSING}"
from pathlib import Path
import sys
from omegaconf import OmegaConf

repo_root = Path(sys.argv[1])
config_path = repo_root / sys.argv[2]
task_filter = sys.argv[3]
include_file = Path(sys.argv[4])
missing_file = Path(sys.argv[5])

cfg = OmegaConf.load(config_path)
if "best_response_set" not in cfg:
    raise SystemExit(f"Config missing best_response_set: {config_path}")

br_set = cfg["best_response_set"]
if task_filter != "all":
    if task_filter not in br_set:
        raise SystemExit(f"Task '{task_filter}' not found in best_response_set")
    task_items = [(task_filter, br_set[task_filter])]
else:
    task_items = list(br_set.items())

paths = []
missing = []
for task_name, task_cfg in task_items:
    for br_name, br_entry in task_cfg.items():
        if "path" not in br_entry:
            continue
        raw = str(br_entry["path"]).rstrip("/")
        p = Path(raw)
        abs_p = p if p.is_absolute() else (repo_root / p)
        if abs_p.exists():
            rel = abs_p.relative_to(repo_root) if not p.is_absolute() else abs_p
            paths.append(str(rel))
        else:
            missing.append((task_name, br_name, raw))

# unique + stable
paths = sorted(set(paths))

include_file.write_text("\n".join(paths) + ("\n" if paths else ""), encoding="utf-8")
missing_lines = [f"{t}\t{b}\t{p}" for (t, b, p) in missing]
missing_file.write_text("\n".join(missing_lines) + ("\n" if missing_lines else ""), encoding="utf-8")

print(f"included_paths={len(paths)}")
print(f"missing_paths={len(missing)}")
PY

if [[ -s "$TMP_MISSING" ]]; then
  echo "Warning: some configured BR paths are missing:" >&2
  head -n 20 "$TMP_MISSING" >&2
fi

if [[ ! -s "$TMP_INCLUDE" ]]; then
  echo "Error: no existing BR checkpoint paths found to zip." >&2
  exit 1
fi

if [[ "$DRY_RUN" == "true" ]]; then
  echo "Dry run: paths that would be zipped (first 40):"
  head -n 40 "$TMP_INCLUDE"
  exit 0
fi

TS="$(date -u +%Y%m%d_%H%M%S)"
SCOPE="${TASK_FILTER//\//_}"
ZIP_PATH="${OUT_DIR}/${PREFIX}_${SCOPE}_${TS}.zip"

(
  cd "$REPO_ROOT"
  zip -r "$ZIP_PATH" -@ < "$TMP_INCLUDE"
)

if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "${REPO_ROOT}/${ZIP_PATH}" > "${REPO_ROOT}/${ZIP_PATH}.sha256"
  echo "Wrote checksum: ${ZIP_PATH}.sha256"
fi

echo "Created zip: ${ZIP_PATH}"

REMOTE_USER="$(whoami)"
REMOTE_HOST_FQDN="$(hostname -f 2>/dev/null || hostname)"
REMOTE_HOST_SHORT="$(hostname 2>/dev/null || echo remote-host)"

echo
echo "Copy to local machine examples:"
echo "  scp ${REMOTE_USER}@${REMOTE_HOST_FQDN}:${REPO_ROOT}/${ZIP_PATH} ."
echo "  scp ${REMOTE_USER}@${REMOTE_HOST_SHORT}:${REPO_ROOT}/${ZIP_PATH} ."

if [[ -f "${REPO_ROOT}/${ZIP_PATH}.sha256" ]]; then
  echo "  scp ${REMOTE_USER}@${REMOTE_HOST_FQDN}:${REPO_ROOT}/${ZIP_PATH}.sha256 ."
  echo "  scp ${REMOTE_USER}@${REMOTE_HOST_SHORT}:${REPO_ROOT}/${ZIP_PATH}.sha256 ."
fi
