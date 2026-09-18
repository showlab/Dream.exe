#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
workspace="${repository_root}/configs/workspace.json"
python_bin="${DREAM_EXE_PYTHON:-python}"
profile="core"
check_only=false
dry_run=false

usage() {
  cat <<'EOF'
Usage: bash integrations/setup_dependencies.sh [options]

Install one explicit Dream.exe dependency profile and prepare its pinned
provider source checkouts. Package installation never runs this script.

Options:
  --profile NAME     core (default), optional, or generation
  --workspace PATH   workspace JSON (default: configs/workspace.json)
  --python PATH      Python executable (default: $DREAM_EXE_PYTHON or python)
  --check            verify the selected environment and sources without writes
  --dry-run          print the commands without executing them
  -h, --help         show this help

The generation profile prepares `video_gen` and belongs in a separate
environment because its pinned dependencies conflict with the main pipeline
environment. The core and optional profiles prepare `exec` providers; VLM
`eval` configuration is separate from this script.
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
    --profile)
      require_value "$@"
      profile="$2"
      shift 2
      ;;
    --workspace)
      require_value "$@"
      workspace="$2"
      shift 2
      ;;
    --python)
      require_value "$@"
      python_bin="$2"
      shift 2
      ;;
    --check)
      check_only=true
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

case "$profile" in
  core|optional|generation) ;;
  *)
    printf 'error: --profile must be core, optional, or generation\n' >&2
    exit 2
    ;;
esac

if [[ "$workspace" != /* ]]; then
  workspace="${repository_root}/${workspace}"
fi

run() {
  if "$dry_run"; then
    printf 'DRY-RUN:'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

if "$check_only"; then
  run "$python_bin" -m pip check
  run "$python_bin" "${repository_root}/integrations/setup.py" \
    "$profile" \
    --repository-root "$repository_root" \
    --workspace "$workspace" \
    --check
  exit 0
fi

case "$profile" in
  core)
    run "$python_bin" -m pip install -e \
      "${repository_root}[assets,video,sim,robocasa-runtime,dvd-runtime,tracking-runtime,region-runtime,vlm]"
    run env SAM2_BUILD_CUDA=0 "$python_bin" -m pip install \
      --no-deps \
      --no-build-isolation \
      -r "${repository_root}/integrations/requirements/sam2.txt"
    ;;
  optional)
    run "$python_bin" -m pip install -e "${repository_root}[vda-runtime]"
    ;;
  generation)
    run "$python_bin" -m pip install -e \
      "${repository_root}[video,wan22-runtime]"
    ;;
esac

run "$python_bin" "${repository_root}/integrations/setup.py" \
  "$profile" \
  --repository-root "$repository_root" \
  --workspace "$workspace"
