#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

COMPOSE_FILE="${EEG_MEG_COMPOSE_FILE:-docker-compose.eeg-reading-listening.yml}"
SERVICE="${EEG_MEG_SERVICE:-eeg_train_reading_listening}"
GPU="${EEG_MEG_GPU:-0}"
WANDB_MODE="${WANDB_MODE:-offline}"
STAMP="$(date +%Y%m%d_%H%M%S)"
EXPERIMENT="${EEG_MEG_EXPERIMENT:-megxl_to_all_eeg_with_meg_replay_${STAMP}}"
LOG_DIR="${EEG_MEG_LOG_DIR:-logs/eeg_reading_listening_megxl_joint}"
RUN_LOG="${LOG_DIR}/${EXPERIMENT}.log"

mkdir -p \
  "$LOG_DIR" \
  data/cache \
  checkpoints \
  logs \
  results \
  wandb \
  embeddings_cache \
  hf_cache

echo "Experiment: $EXPERIMENT"
echo "GPU: $GPU"
echo "Checkpoint: ${CRISS_CROSS_CHECKPOINT:-./checkpoints/baseline/meg-xl-med.ckpt}"
echo "Log: $RUN_LOG"

EEG_GPU="$GPU" WANDB_MODE="$WANDB_MODE" \
  docker compose -f "$COMPOSE_FILE" run --rm --no-deps \
    -e "WANDB_MODE=${WANDB_MODE}" \
    -e "CRISS_CROSS_CHECKPOINT=${CRISS_CROSS_CHECKPOINT:-./checkpoints/baseline/meg-xl-med.ckpt}" \
    "$SERVICE" \
    bash -lc '
      exec uv run --no-sync python -m brainstorm.train_criss_cross_eeg_continuous \
        --config-name=train_criss_cross_eeg_reading_listening_megxl_joint \
        logging.experiment_name="$1" \
        "${@:2}"
    ' bash "$EXPERIMENT" "$@" \
    2>&1 | tee "$RUN_LOG"
