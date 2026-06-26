#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

COMPOSE_FILE="${EEG_MEG_COMPOSE_FILE:-docker-compose.eeg-reading-listening.yml}"
SERVICE="${EEG_MEG_SERVICE:-eeg_train_reading_listening}"
GPU="${EEG_MEG_GPU:-0}"
WANDB_MODE="${WANDB_MODE:-offline}"
CHECKPOINT="${CRISS_CROSS_CHECKPOINT:-./checkpoints/baseline/meg-xl-med.ckpt}"
STAMP="$(date +%Y%m%d_%H%M%S)"
EXPERIMENT="${EEG_MEG_EXPERIMENT:-megxl_to_all_eeg_${STAMP}}"
LOG_DIR="${EEG_MEG_LOG_DIR:-logs/eeg_reading_listening_megxl}"
RUN_LOG="${LOG_DIR}/${EXPERIMENT}.log"
PID_FILE="${LOG_DIR}/${EXPERIMENT}.pid"
LATEST_LOG="${LOG_DIR}/latest.log"
LATEST_PID="${LOG_DIR}/latest.pid"

mkdir -p \
  "$LOG_DIR" \
  data/cache \
  checkpoints \
  logs \
  results \
  wandb \
  embeddings_cache \
  hf_cache

if [[ ! -f "$COMPOSE_FILE" ]]; then
  echo "ERROR: compose file not found: $COMPOSE_FILE" >&2
  exit 1
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "ERROR: MEG-XL checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

# Docker Compose remains attached to the training container, while nohup places
# the complete command in the background and redirects all output to RUN_LOG.
nohup env \
  EEG_GPU="$GPU" \
  WANDB_MODE="$WANDB_MODE" \
  docker compose -f "$COMPOSE_FILE" run --rm --no-deps \
    -e "WANDB_MODE=${WANDB_MODE}" \
    -e "CRISS_CROSS_CHECKPOINT=${CHECKPOINT}" \
    "$SERVICE" \
    bash -lc '
      exec uv run --no-sync python -m brainstorm.train_criss_cross_eeg_continuous \
        --config-name=train_criss_cross_eeg_reading_listening_megxl_joint \
        logging.experiment_name="$1" \
        "${@:2}"
    ' bash "$EXPERIMENT" "$@" \
  >"$RUN_LOG" 2>&1 < /dev/null &

PID=$!
printf '%s\n' "$PID" > "$PID_FILE"
printf '%s\n' "$PID" > "$LATEST_PID"
ln -sfn "$(basename "$RUN_LOG")" "$LATEST_LOG"

echo "Experiment: $EXPERIMENT"
echo "Mode: MEG-XL checkpoint -> EEG only (no LibriBrain/MEG replay)"
echo "GPU: $GPU"
echo "Checkpoint: $CHECKPOINT"
echo "PID: $PID"
echo "Log: $RUN_LOG"
echo
echo "Ver el log en directo:"
echo "  tail -f $LATEST_LOG"
echo
echo "Ver las ultimas 100 lineas:"
echo "  tail -n 100 $LATEST_LOG"
echo
echo "Comprobar el proceso:"
echo "  ps -p \$(cat $LATEST_PID) -o pid,etime,cmd"
echo
echo "Buscar errores:"
echo "  grep -iE 'error|exception|traceback|failed|cuda out of memory' $LATEST_LOG"
