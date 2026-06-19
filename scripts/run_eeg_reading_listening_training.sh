#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPOSE_FILE="${REPO_ROOT}/docker-compose.eeg-reading-listening.yml"
DOCKER_BIN="${DOCKER_BIN:-docker}"
DOCKER_CMD=("${DOCKER_BIN}")

if [[ "${DOCKER_USE_SUDO:-0}" == "1" ]]; then
  DOCKER_CMD=(sudo "${DOCKER_BIN}")
fi

mkdir -p \
  "${REPO_ROOT}/data/cache" \
  "${REPO_ROOT}/logs" \
  "${REPO_ROOT}/checkpoints" \
  "${REPO_ROOT}/promotions" \
  "${REPO_ROOT}/results" \
  "${REPO_ROOT}/wandb" \
  "${REPO_ROOT}/embeddings_cache" \
  "${REPO_ROOT}/hf_cache"

cd "${REPO_ROOT}"

exec "${DOCKER_CMD[@]}" compose \
  -f "${COMPOSE_FILE}" \
  run --rm --no-deps \
  eeg_train_reading_listening \
  bash -lc '
    exec uv run --no-sync python -m brainstorm.train_criss_cross_eeg_continuous \
      --config-name=train_criss_cross_eeg_reading_listening_continuous "$@"
  ' bash "$@"
