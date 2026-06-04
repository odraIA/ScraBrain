#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET_ID="${OPENNEURO_DATASET_ID:-ds007808}"
DEST_DIR="${OPENNEURO_DEST_DIR:-${REPO_ROOT}/datasets/OpenNeuroEEG}"
DOCKER_BIN="${DOCKER_BIN:-docker}"
DOCKER_CMD=("${DOCKER_BIN}")

if [[ "${DOCKER_USE_SUDO:-0}" == "1" ]]; then
  DOCKER_CMD=(sudo "${DOCKER_BIN}")
fi

mkdir -p "${DEST_DIR}"
DEST_ABS="$(cd "${DEST_DIR}" && pwd)"

echo "Downloading OpenNeuro ${DATASET_ID} into ${DEST_ABS}"

exec "${DOCKER_CMD[@]}" run --rm \
  --user "$(id -u):$(id -g)" \
  -v "${DEST_ABS}:/data" \
  amazon/aws-cli \
  s3 sync --no-sign-request "s3://openneuro.org/${DATASET_ID}" /data "$@"
