#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
workspace="${repository_root}/configs/workspace.json"
python_bin="${DREAM_EXE_PYTHON:-}"
asset_root=""
accept_noncommercial=false
accepted_licenses=()
check_only=false
offline=false
force=false
dry_run=false

usage() {
  cat <<'EOF'
Usage: bash integrations/download_checkpoints.sh [options]

Acquire the pinned public checkpoints required by the core Dream.exe pipeline.
Every file is downloaded into a temporary tree, checked for exact byte count
and SHA-256, and then published atomically under the workspace checkpoint root.

Options:
  --workspace PATH   workspace JSON (default: configs/workspace.json)
  --asset-root PATH  override the workspace checkpoint root
  --python PATH      Python >=3.10 (default: $DREAM_EXE_PYTHON, python, or python3)
  --accept-noncommercial-licenses
                     accept both CoTracker and DVD CC BY-NC 4.0 terms
  --accept-license ID
                     accept one manifest license id; may be repeated
  --check            verify existing public core assets without downloading
  --offline          prohibit downloads; report any missing assets
  --force            replace files that exist but fail verification
  --dry-run          plan only; perform no network access or writes
  -h, --help         show this help

This script does not acquire the separately released Dream.exe DVD LoRA bundle.
Review THIRD_PARTY.md and docs/MODEL_ASSETS.md before accepting model terms.
EOF
}

require_value() {
  if (($# < 2)) || [[ -z "$2" ]]; then
    printf 'error: %s requires a value\n' "$1" >&2
    exit 2
  fi
}

while (($#)); do
  case "$1" in
    --workspace)
      require_value "$@"
      workspace="$2"
      shift 2
      ;;
    --asset-root)
      require_value "$@"
      asset_root="$2"
      shift 2
      ;;
    --python)
      require_value "$@"
      python_bin="$2"
      shift 2
      ;;
    --accept-noncommercial-licenses)
      accept_noncommercial=true
      shift
      ;;
    --accept-license)
      require_value "$@"
      accepted_licenses+=("$2")
      shift 2
      ;;
    --check)
      check_only=true
      shift
      ;;
    --offline)
      offline=true
      shift
      ;;
    --force)
      force=true
      shift
      ;;
    --dry-run)
      dry_run=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'error: unknown option: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

supports_python_version() {
  "$1" -c 'import sys; sys.exit(sys.version_info < (3, 10))' >/dev/null 2>&1
}

if [[ -z "$python_bin" ]]; then
  for candidate in python python3; do
    if supports_python_version "$candidate"; then
      python_bin="$candidate"
      break
    fi
  done
fi
if [[ -z "$python_bin" ]] || ! supports_python_version "$python_bin"; then
  printf 'error: Python >=3.10 is required; activate a compatible environment or set DREAM_EXE_PYTHON / --python to its executable.\n' >&2
  exit 2
fi

if "$check_only" && "$dry_run"; then
  printf 'error: --check and --dry-run are mutually exclusive\n' >&2
  exit 2
fi
if "$check_only" && "$force"; then
  printf 'error: --check and --force are mutually exclusive\n' >&2
  exit 2
fi

if [[ "$workspace" != /* ]]; then
  workspace="${repository_root}/${workspace}"
fi

cd "$repository_root"

if [[ -z "$asset_root" ]]; then
  asset_root="$("$python_bin" -c \
    'from dream_exe.bench.data.workspace import load_workspace; import sys; print(load_workspace(sys.argv[1], require_existing_bench=False).checkpoint_root)' \
    "$workspace")"
elif [[ "$asset_root" != /* ]]; then
  asset_root="${repository_root}/${asset_root}"
fi

if "$check_only"; then
  exec "$python_bin" -m dream_exe verify-model-assets \
    --manifest core \
    --asset-root "$asset_root"
fi

if "$accept_noncommercial"; then
  accepted_licenses+=(
    "cotracker-cc-by-nc-4.0"
    "dvd-cc-by-nc-4.0"
  )
fi

command=(
  "$python_bin" -m dream_exe acquire-model-assets
  --manifest core
  --asset-root "$asset_root"
)
for license_id in "${accepted_licenses[@]}"; do
  command+=(--accept-license "$license_id")
done
if "$offline"; then
  command+=(--offline)
fi
if "$force"; then
  command+=(--force)
fi
if "$dry_run"; then
  command+=(--dry-run)
fi

"${command[@]}"
