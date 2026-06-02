#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: bash run_zuco_evals.sh [options]

Launches the ZuCo 2.0 NR MEG-XL evaluation container:
  - eval_zuco on ZUCO_GPU for both EEG-as-MEG and EEG-type variants
  - monitor on MONITOR_PORT

Options:
  --no-build      Do not run docker compose build first.
  --no-monitor    Do not launch the monitor service.
  --monitor-only  Launch only the monitor service.
  --eval-only     Launch only eval_zuco.
  --logs          Follow eval_zuco logs while it runs.
  --variant NAME  Run one variant only: meg or eeg. Can be repeated.
  -h, --help      Show this help.

Environment overrides:
  ZUCO_GPU=0
  EVAL_GPU=0
  MONITOR_PORT=8080
  DATASETS_DIR=./datasets
  ZUCO_ROOT=./datasets/zuco2/data/zuco2
  ZUCO_EVAL_VARIANTS="meg eeg"
  CHECKPOINTS_DIR=./checkpoints
  CRISS_CROSS_CHECKPOINT=./checkpoints/baseline/meg-xl-med.ckpt
  WANDB_MODE=offline
USAGE
}

build=1
follow_logs=0
eval_services=(eval_zuco)
launch_monitor=1
validate_eval_inputs=1
eval_variants=()

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
      eval_services=()
      launch_monitor=1
      validate_eval_inputs=0
      shift
      ;;
    --eval-only)
      eval_services=(eval_zuco)
      shift
      ;;
    --logs)
      follow_logs=1
      shift
      ;;
    --variant)
      if [[ $# -lt 2 ]]; then
        echo "--variant requires a value: meg or eeg" >&2
        exit 2
      fi
      eval_variants+=("$2")
      shift 2
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
export ZUCO_GPU="${ZUCO_GPU:-$EVAL_GPU}"
export MONITOR_PORT="${MONITOR_PORT:-8080}"
export DATASETS_DIR="${DATASETS_DIR:-./datasets}"
export ZUCO_ROOT="${ZUCO_ROOT:-./datasets/zuco2/data/zuco2}"
export ZUCO_EVAL_VARIANTS="${ZUCO_EVAL_VARIANTS:-meg eeg}"
export CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-./checkpoints}"
export CRISS_CROSS_CHECKPOINT="${CRISS_CROSS_CHECKPOINT:-./checkpoints/baseline/meg-xl-med.ckpt}"
export WANDB_MODE="${WANDB_MODE:-offline}"

mkdir -p logs data/cache embeddings_cache hf_cache wandb "$CHECKPOINTS_DIR"

if [[ "${#eval_services[@]}" -gt 0 ]]; then
  if [[ "${#eval_variants[@]}" -eq 0 ]]; then
    read -r -a eval_variants <<< "$ZUCO_EVAL_VARIANTS"
  fi

  if [[ "${#eval_variants[@]}" -eq 0 ]]; then
    echo "No ZuCo eval variants selected." >&2
    exit 2
  fi

  for variant in "${eval_variants[@]}"; do
    case "$variant" in
      meg|mag|eeg)
        ;;
      *)
        echo "Unknown ZuCo eval variant: $variant" >&2
        echo "Expected one of: meg, mag, eeg" >&2
        exit 2
        ;;
    esac
  done
fi

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

validate_zuco_root() {
  local configured_root="$1"
  local host_root

  host_root="$(resolve_dataset_host_root "$configured_root")"
  if [[ ! -d "$host_root" ]]; then
    echo "ZuCo dataset root not found on host: $host_root" >&2
    echo "Set DATASETS_DIR=/path/to/datasets and ZUCO_ROOT to the dataset root visible inside the container." >&2
    exit 1
  fi

  if [[ ! -d "$host_root/task1 - NR/Preprocessed" ]]; then
    echo "ZuCo preprocessed NR directory not found: $host_root/task1 - NR/Preprocessed" >&2
    echo "ZUCO_ROOT should usually be ./datasets/zuco2/data/zuco2." >&2
    exit 1
  fi

  if [[ ! -d "$host_root/task_materials" ]]; then
    echo "ZuCo task_materials directory not found: $host_root/task_materials" >&2
    echo "ZUCO_ROOT should usually be ./datasets/zuco2/data/zuco2." >&2
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
    echo "Set DATASETS_DIR=/path/to/datasets so it contains zuco2/." >&2
    exit 1
  fi

  validate_zuco_root "$ZUCO_ROOT"
fi

services=("${eval_services[@]}")
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

if [[ "${#eval_services[@]}" -eq 0 ]]; then
  echo "Monitor launched detached."
  echo "Logs:"
  echo "  docker compose logs -f monitor"
  exit 0
fi

echo "Running ZuCo eval variants: ${eval_variants[*]}"
echo "Stop:"
echo "  docker compose stop ${eval_services[*]}"
echo

for variant in "${eval_variants[@]}"; do
  export ZUCO_EEG_SENSOR_TYPE="$variant"
  echo "Starting ZuCo variant: ${variant}"

  for service in "${eval_services[@]}"; do
    echo "Starting ${service} with ZUCO_EEG_SENSOR_TYPE=${ZUCO_EEG_SENSOR_TYPE}..."
    docker compose up -d --force-recreate "$service"
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
      echo "${service} (${variant}) exited with status ${exit_code}." >&2
      exit "$exit_code"
    fi

    echo "${service} (${variant}) completed successfully."
    echo
  done
done

echo "ZuCo eval variants completed successfully."
