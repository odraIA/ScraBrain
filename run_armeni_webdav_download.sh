#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: bash run_armeni_webdav_download.sh [options]

Launches the Armeni WebDAV downloader in a detached Docker container. The
container keeps running if this terminal closes.

Options:
  --build          Build scrabrain-megxl:latest before launching.
  --dry-run        List missing files without downloading.
  --required-only  Download only .meg4 and .res4 files (default).
  --all-files      Download every file in each .ds directory.
  --replace        Remove an existing downloader container with the same name.
  --logs           Follow logs after launching.
  -h, --help       Show this help.

Environment overrides:
  DATASETS_DIR=./datasets
  ARMENI_ROOT=./datasets/armeni
  DOWNLOAD_CONTAINER=scrabrain_download_armeni
  DOWNLOAD_IMAGE=scrabrain-megxl:latest
  RDR_USERNAME=
  RDR_PASSWORD=
  WEBDAV_USERNAME=
  WEBDAV_PASSWORD=

Extra downloader arguments can be passed after --, for example:
  bash run_armeni_webdav_download.sh -- --subjects sub-001 --sessions ses-001
USAGE
}

build=0
dry_run=0
required_only=1
all_files=0
replace=0
follow_logs=0
extra_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build)
      build=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    --required-only)
      required_only=1
      all_files=0
      shift
      ;;
    --all-files)
      required_only=0
      all_files=1
      shift
      ;;
    --replace)
      replace=1
      shift
      ;;
    --logs)
      follow_logs=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      extra_args=("$@")
      break
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

export DATASETS_DIR="${DATASETS_DIR:-./datasets}"
export ARMENI_ROOT="${ARMENI_ROOT:-./datasets/armeni}"
export DOWNLOAD_CONTAINER="${DOWNLOAD_CONTAINER:-scrabrain_download_armeni}"
export DOWNLOAD_IMAGE="${DOWNLOAD_IMAGE:-scrabrain-megxl:latest}"

repo_dir="$(pwd)"
mkdir -p "$DATASETS_DIR"

resolve_dataset_host_root() {
  local root="$1"

  case "$root" in
    ./datasets/*)
      echo "${DATASETS_DIR%/}/${root#./datasets/}"
      ;;
    /workspace/datasets/*)
      echo "${DATASETS_DIR%/}/${root#/workspace/datasets/}"
      ;;
    *)
      echo "$root"
      ;;
  esac
}

dataset_host_root="$(resolve_dataset_host_root "$ARMENI_ROOT")"
if [[ ! -d "$dataset_host_root" ]]; then
  echo "Armeni dataset root not found on host: $dataset_host_root" >&2
  echo "Set DATASETS_DIR and ARMENI_ROOT so the downloader can write to the dataset." >&2
  exit 1
fi

resolve_dataset_container_root() {
  local root="$1"

  case "$root" in
    ./datasets/*)
      echo "/workspace/datasets/${root#./datasets/}"
      ;;
    /workspace/datasets/*)
      echo "$root"
      ;;
    *)
      echo "/workspace/datasets/$(basename "$root")"
      ;;
  esac
}

existing_container="$(docker ps -aq --filter "name=^/${DOWNLOAD_CONTAINER}$")"
if [[ -n "$existing_container" ]]; then
  if [[ "$replace" -eq 1 ]]; then
    docker rm -f "$DOWNLOAD_CONTAINER" >/dev/null
  else
    echo "Container already exists: $DOWNLOAD_CONTAINER" >&2
    echo "Inspect it with: docker logs -f $DOWNLOAD_CONTAINER" >&2
    echo "Remove/replace it with: bash run_armeni_webdav_download.sh --replace" >&2
    exit 1
  fi
fi

if [[ "$build" -eq 1 ]]; then
  docker compose build eval_armeni
fi

download_args=(
  "scripts/download_armeni_webdav_missing.py"
  "--dataset-root" "$(resolve_dataset_container_root "$ARMENI_ROOT")"
)

if [[ "$dry_run" -eq 1 ]]; then
  download_args+=("--dry-run")
fi

if [[ "$required_only" -eq 1 ]]; then
  download_args+=("--required-only")
fi

if [[ "$all_files" -eq 1 ]]; then
  download_args+=("--all-files")
fi

download_args+=("${extra_args[@]}")

container_id="$(
  docker run -d \
    --name "$DOWNLOAD_CONTAINER" \
    --init \
    --user "$(id -u):$(id -g)" \
    --workdir /workspace \
    -e RDR_USERNAME="${RDR_USERNAME:-}" \
    -e RDR_PASSWORD="${RDR_PASSWORD:-}" \
    -e WEBDAV_USERNAME="${WEBDAV_USERNAME:-}" \
    -e WEBDAV_PASSWORD="${WEBDAV_PASSWORD:-}" \
    -v "$repo_dir:/workspace" \
    -v "$(realpath "$DATASETS_DIR"):/workspace/datasets" \
    "$DOWNLOAD_IMAGE" \
    python3.12 "${download_args[@]}"
)"

echo "Started detached downloader: $DOWNLOAD_CONTAINER"
echo "Container ID: $container_id"
echo
echo "Follow logs:"
echo "  docker logs -f $DOWNLOAD_CONTAINER"
echo
echo "Stop it:"
echo "  docker stop $DOWNLOAD_CONTAINER"
echo
echo "When it finishes, rerun the Armeni eval:"
echo "  bash run_armeni_evals.sh"

if [[ "$follow_logs" -eq 1 ]]; then
  docker logs -f "$DOWNLOAD_CONTAINER"
fi
