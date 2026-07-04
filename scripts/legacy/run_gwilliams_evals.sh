#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: bash scripts/legacy/run_gwilliams_evals.sh [options]

Launches the Gwilliams MEG-XL evaluation container:
  - eval_gwilliams on GWILLIAMS_GPU
  - monitor on MONITOR_PORT

Options:
  --no-build      Do not run docker compose build first.
  --no-monitor    Do not launch the monitor service.
  --monitor-only  Launch only the monitor service.
  --logs          Follow logs for the evaluation while it runs.
  -h, --help      Show this help.

Environment overrides:
  GWILLIAMS_GPU=0
  EVAL_GPU=0
  MONITOR_PORT=8080
  DATASETS_DIR=./datasets
  GWILLIAMS_ROOT=./datasets/gwilliams
  CHECKPOINTS_DIR=./checkpoints
  CRISS_CROSS_CHECKPOINT=./checkpoints/baseline/meg-xl-med.ckpt
  WANDB_MODE=offline
USAGE
}

build=1
follow_logs=0
eval_service="eval_gwilliams"
launch_monitor=1
validate_eval_inputs=1

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
      eval_service=""
      launch_monitor=1
      validate_eval_inputs=0
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
export GWILLIAMS_GPU="${GWILLIAMS_GPU:-$EVAL_GPU}"
export MONITOR_PORT="${MONITOR_PORT:-8080}"
export DATASETS_DIR="${DATASETS_DIR:-./datasets}"
export GWILLIAMS_ROOT="${GWILLIAMS_ROOT:-./datasets/gwilliams}"
export CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-./checkpoints}"
export CRISS_CROSS_CHECKPOINT="${CRISS_CROSS_CHECKPOINT:-./checkpoints/baseline/meg-xl-med.ckpt}"
export WANDB_MODE="${WANDB_MODE:-offline}"

mkdir -p logs data/cache embeddings_cache hf_cache wandb "$CHECKPOINTS_DIR"

resolve_checkpoint_host_path() {
  local checkpoint="$1"

  case "$checkpoint" in
    ./checkpoints/*)
      echo "${CHECKPOINTS_DIR%/}/${checkpoint#./checkpoints/}"
      ;;
    /workspace/checkpoints/*)
      echo "${CHECKPOINTS_DIR%/}/${checkpoint#/workspace/checkpoints/}"
      ;;
    *)
      echo "$checkpoint"
      ;;
  esac
}

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

validate_dataset_root() {
  local label="$1"
  local configured_root="$2"
  local host_root

  host_root="$(resolve_dataset_host_root "$configured_root")"
  if [[ ! -d "$host_root" ]]; then
    echo "${label} dataset root not found on host: $host_root" >&2
    echo "Set DATASETS_DIR=/path/to/datasets and ${label^^}_ROOT to the dataset root visible inside the container." >&2
    exit 1
  fi
}

if [[ "$validate_eval_inputs" -eq 1 ]]; then
  checkpoint_host_path="$(resolve_checkpoint_host_path "$CRISS_CROSS_CHECKPOINT")"
  if [[ ! -e "$checkpoint_host_path" ]]; then
    echo "Checkpoint not found on host: $checkpoint_host_path" >&2
    echo "Set CHECKPOINTS_DIR=/host/path/to/checkpoints and CRISS_CROSS_CHECKPOINT=./checkpoints/<file>.ckpt." >&2
    exit 1
  fi

  if [[ ! -d "$DATASETS_DIR" ]]; then
    echo "Dataset mount directory not found: $DATASETS_DIR" >&2
    echo "Set DATASETS_DIR=/path/to/datasets so it contains the requested dataset root." >&2
    exit 1
  fi

  validate_dataset_root "gwilliams" "$GWILLIAMS_ROOT"
fi

services=()
if [[ -n "$eval_service" ]]; then
  services+=("$eval_service")
fi
if [[ "$launch_monitor" -eq 1 ]]; then
  services+=(monitor)
fi

if [[ "${#services[@]}" -eq 0 ]]; then
  echo "No services selected." >&2
  exit 2
fi

if [[ "$build" -eq 1 ]]; then
  docker compose build "${services[@]}"
fi

if [[ "$launch_monitor" -eq 1 ]]; then
  docker compose up -d monitor
  docker compose ps monitor
  echo "Monitor: http://localhost:${MONITOR_PORT}"
  echo
fi

if [[ -z "$eval_service" ]]; then
  echo "Monitor launched detached."
  echo "Logs:"
  echo "  docker compose logs -f monitor"
  exit 0
fi

echo "Running Gwilliams eval."
echo "Stop:"
echo "  docker compose stop ${eval_service}"
echo

echo "Starting ${eval_service}..."
docker compose up -d "$eval_service"
docker compose ps "$eval_service"
echo "Logs:"
echo "  docker compose logs -f ${eval_service}"

log_pid=""
if [[ "$follow_logs" -eq 1 ]]; then
  docker compose logs -f "$eval_service" &
  log_pid="$!"
fi

container_id="$(docker compose ps -q "$eval_service")"
if [[ -z "$container_id" ]]; then
  echo "Could not find container id for ${eval_service}." >&2
  exit 1
fi

exit_code="$(docker wait "$container_id")"
if [[ -n "$log_pid" ]]; then
  wait "$log_pid" || true
fi

if [[ "$exit_code" != "0" ]]; then
  echo "${eval_service} exited with status ${exit_code}." >&2
  exit "$exit_code"
fi

echo "${eval_service} completed successfully."
