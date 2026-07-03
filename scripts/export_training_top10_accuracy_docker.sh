#!/usr/bin/env bash
set -Eeuo pipefail

# Run the top-10 training-curve exporter inside the project Docker image.
#
# Usage:
#   bash scripts/export_training_top10_accuracy_docker.sh
#   bash scripts/export_training_top10_accuracy_docker.sh --run-dir logs/.../random_init

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

COMPOSE_FILE="${TOP10_EXPORT_COMPOSE_FILE:-docker-compose.yml}"
SERVICE="${TOP10_EXPORT_SERVICE:-eval_eeg_listening}"
RUN_ID="${TOP10_EXPORT_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${TOP10_EXPORT_OUTPUT_DIR:-./results/training_top10_accuracy/$RUN_ID}"
BUILD_IMAGE="${BUILD_IMAGE:-0}"
WANDB_MODE="${WANDB_MODE:-offline}"
EEG_GPU="${EEG_GPU:-0}"

command -v docker >/dev/null 2>&1 || {
  echo "ERROR: docker is not available" >&2
  exit 2
}

mkdir -p "$OUTPUT_DIR" ./hf_cache/matplotlib ./logs ./results

if [[ "$BUILD_IMAGE" == "1" || "$BUILD_IMAGE" == "true" ]]; then
  docker compose -f "$COMPOSE_FILE" build "$SERVICE"
fi

echo "Exporting top-10 training curves in Docker"
echo "Compose file: $COMPOSE_FILE"
echo "Service: $SERVICE"
echo "Output: $OUTPUT_DIR"

docker compose -f "$COMPOSE_FILE" run --rm --no-deps \
  -e "NVIDIA_VISIBLE_DEVICES=$EEG_GPU" \
  -e "EEG_GPU=$EEG_GPU" \
  -e "WANDB_MODE=$WANDB_MODE" \
  -e "MPLCONFIGDIR=/workspace/hf_cache/matplotlib" \
  "$SERVICE" \
  uv run --no-sync python scripts/export_training_top10_accuracy.py \
    --output-dir "$OUTPUT_DIR" \
    "$@"

echo "Done: $OUTPUT_DIR"
