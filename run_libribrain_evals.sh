#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: bash run_libribrain_evals.sh [options]

Launches the two LibriBrain MEG-XL evaluation containers sequentially:
  - eval_libribrain on EVAL_GPU
  - eval_libribrain_linear_probe on LINEAR_PROBE_GPU
  - monitor on MONITOR_PORT

Options:
  --no-build             Do not run docker compose build first.
  --no-monitor           Do not launch the monitor service.
  --monitor-only         Launch only the monitor service.
  --eval-only            Launch only eval_libribrain.
  --linear-probe-only    Launch only eval_libribrain_linear_probe.
  --logs                 Follow logs after launching.
  -h, --help             Show this help.

Environment overrides:
  EVAL_GPU=0
  LINEAR_PROBE_GPU=1
  MONITOR_PORT=8080
  DATASETS_DIR=./datasets
  LIBRIBRAIN_ROOT=./datasets/libribrain
  CHECKPOINTS_DIR=./checkpoints
  CRISS_CROSS_CHECKPOINT=./checkpoints/baseline/meg-xl-med.ckpt
  WANDB_MODE=offline
USAGE
}

build=1
follow_logs=0
services=(eval_libribrain eval_libribrain_linear_probe)
launch_monitor=1
validate_eval_inputs=1

download_file() {
  local url="$1"
  local dest="$2"

  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$url" -o "$dest"
  elif command -v python3 >/dev/null 2>&1; then
    python3 - "$url" "$dest" <<'PY'
import sys
from urllib.request import urlopen

url, dest = sys.argv[1], sys.argv[2]
with urlopen(url, timeout=60) as response:
    data = response.read()
with open(dest, "wb") as f:
    f.write(data)
PY
  else
    return 1
  fi
}

ensure_libribrain_metadata() {
  local libribrain_host_root="$1"
  local metadata_dir="$libribrain_host_root/metadata"

  if [[ -e "$libribrain_host_root/meg_sensors_information.json" ]] ||
     [[ -e "$metadata_dir/meg_sensors_information.json" ]] ||
     [[ -e "$metadata_dir/sensor_xyz.json" && -e "$metadata_dir/channels.tsv" ]] ||
     [[ -e "$libribrain_host_root/sensor_xyz.json" ]]; then
    return 0
  fi

  if [[ ! -w "$libribrain_host_root" ]]; then
    echo "LibriBrain sensor metadata not found under: $libribrain_host_root" >&2
    echo "Expected metadata/sensor_xyz.json and metadata/channels.tsv, or meg_sensors_information.json." >&2
    echo "The dataset directory is not writable, so I cannot download the metadata automatically." >&2
    exit 1
  fi

  mkdir -p "$metadata_dir"
  echo "LibriBrain sensor metadata not found; downloading metadata/sensor_xyz.json and metadata/channels.tsv..."

  if ! download_file \
    "https://huggingface.co/datasets/pnpl/LibriBrain/raw/main/metadata/sensor_xyz.json" \
    "$metadata_dir/sensor_xyz.json"; then
    echo "Failed to download LibriBrain sensor_xyz.json." >&2
    exit 1
  fi

  if ! download_file \
    "https://huggingface.co/datasets/pnpl/LibriBrain/raw/main/metadata/channels.tsv" \
    "$metadata_dir/channels.tsv"; then
    echo "Failed to download LibriBrain channels.tsv." >&2
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-build)
      build=0
      shift
      ;;
    --no-monitor)
      launch_monitor=0
      shift
      ;;
    --monitor-only)
      services=(monitor)
      launch_monitor=0
      validate_eval_inputs=0
      shift
      ;;
    --eval-only)
      services=(eval_libribrain)
      shift
      ;;
    --linear-probe-only)
      services=(eval_libribrain_linear_probe)
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
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

export EVAL_GPU="${EVAL_GPU:-0}"
export LINEAR_PROBE_GPU="${LINEAR_PROBE_GPU:-1}"
export MONITOR_PORT="${MONITOR_PORT:-8080}"
export DATASETS_DIR="${DATASETS_DIR:-./datasets}"
export LIBRIBRAIN_ROOT="${LIBRIBRAIN_ROOT:-./datasets/libribrain}"
export CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-./checkpoints}"
export CRISS_CROSS_CHECKPOINT="${CRISS_CROSS_CHECKPOINT:-./checkpoints/baseline/meg-xl-med.ckpt}"
export WANDB_MODE="${WANDB_MODE:-offline}"

mkdir -p logs data/cache embeddings_cache hf_cache wandb "$CHECKPOINTS_DIR"

if [[ "$launch_monitor" -eq 1 ]]; then
  services+=(monitor)
fi

if [[ "$validate_eval_inputs" -eq 1 ]]; then
  checkpoint_host_path="$CRISS_CROSS_CHECKPOINT"
  case "$CRISS_CROSS_CHECKPOINT" in
    ./checkpoints/*)
      checkpoint_host_path="${CHECKPOINTS_DIR%/}/${CRISS_CROSS_CHECKPOINT#./checkpoints/}"
      ;;
    /workspace/checkpoints/*)
      checkpoint_host_path="${CHECKPOINTS_DIR%/}/${CRISS_CROSS_CHECKPOINT#/workspace/checkpoints/}"
      ;;
  esac

  if [[ ! -e "$checkpoint_host_path" ]]; then
    echo "Checkpoint not found on host: $checkpoint_host_path" >&2
    echo "Set CHECKPOINTS_DIR=/host/path/to/checkpoints and CRISS_CROSS_CHECKPOINT=./checkpoints/<file>.ckpt." >&2
    exit 1
  fi

  if [[ ! -d "$DATASETS_DIR" ]]; then
    echo "Dataset mount directory not found: $DATASETS_DIR" >&2
    echo "Set DATASETS_DIR=/path/to/datasets so it contains libribrain/." >&2
    exit 1
  fi

  libribrain_host_root="$LIBRIBRAIN_ROOT"
  case "$LIBRIBRAIN_ROOT" in
    ./datasets/libribrain)
      libribrain_host_root="${DATASETS_DIR%/}/libribrain"
      ;;
    /workspace/datasets/libribrain)
      libribrain_host_root="${DATASETS_DIR%/}/libribrain"
      ;;
    ./datasets/*)
      libribrain_host_root="${DATASETS_DIR%/}/${LIBRIBRAIN_ROOT#./datasets/}"
      ;;
    /workspace/datasets/*)
      libribrain_host_root="${DATASETS_DIR%/}/${LIBRIBRAIN_ROOT#/workspace/datasets/}"
      ;;
  esac

  if [[ ! -d "$libribrain_host_root" ]]; then
    echo "LibriBrain dataset root not found on host: $libribrain_host_root" >&2
    echo "Set DATASETS_DIR=/path/to/datasets so it contains libribrain/." >&2
    exit 1
  fi

  ensure_libribrain_metadata "$libribrain_host_root"
fi

if [[ "$build" -eq 1 ]]; then
  docker compose build "${services[@]}"
fi

if [[ " ${services[*]} " == *" eval_libribrain "* ]] &&
   [[ " ${services[*]} " == *" eval_libribrain_linear_probe "* ]]; then
  if [[ " ${services[*]} " == *" monitor "* ]]; then
    docker compose up -d monitor
    docker compose ps monitor
    echo "Monitor: http://localhost:${MONITOR_PORT}"
    echo
  fi

  echo "Running LibriBrain evals sequentially."
  echo "Stop:"
  echo "  docker compose stop eval_libribrain eval_libribrain_linear_probe"
  echo

  for service in eval_libribrain eval_libribrain_linear_probe; do
    echo "Starting ${service}..."
    docker compose up -d "$service"
    docker compose ps "$service"
    echo "Logs:"
    echo "  docker compose logs -f ${service}"

    log_pid=""
    if [[ "$follow_logs" -eq 1 ]]; then
      docker compose logs -f "$service" &
      log_pid="$!"
    fi

    container_id="$(docker compose ps -q "$service")"
    if [[ -z "$container_id" ]]; then
      echo "Could not find container id for ${service}." >&2
      exit 1
    fi

    exit_code="$(docker wait "$container_id")"
    if [[ -n "$log_pid" ]]; then
      wait "$log_pid" || true
    fi

    if [[ "$exit_code" != "0" ]]; then
      echo "${service} exited with status ${exit_code}; not starting the next eval." >&2
      exit "$exit_code"
    fi

    echo "${service} completed successfully."
    echo
  done

  echo "All selected LibriBrain evals completed successfully."
  exit 0
fi

docker compose up -d "${services[@]}"
docker compose ps "${services[@]}"

echo
echo "Containers launched detached. They keep running after this terminal closes."
if [[ " ${services[*]} " == *" monitor "* ]]; then
  echo "Monitor: http://localhost:${MONITOR_PORT}"
fi
echo "Logs:"
echo "  docker compose logs -f ${services[*]}"
echo "Stop:"
echo "  docker compose stop ${services[*]}"

if [[ "$follow_logs" -eq 1 ]]; then
  docker compose logs -f "${services[@]}"
fi
