#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEST_DIR="${SPARRKULEE_DEST_DIR:-${REPO_ROOT}/datasets/sparrkulee}"
REMOTE_NAME="${SPARRKULEE_REMOTE_NAME:-sparrkulee}"
REMOTE_URL="${SPARRKULEE_URL:-}"
DOCKER_BIN="${DOCKER_BIN:-docker}"
DOCKER_CMD=("${DOCKER_BIN}")

if [[ "${DOCKER_USE_SUDO:-0}" == "1" ]]; then
  DOCKER_CMD=(sudo "${DOCKER_BIN}")
fi

if [[ -z "${REMOTE_URL}" ]]; then
  if [[ -z "${SPARRKULEE_USER:-}" || -z "${SPARRKULEE_PASS:-}" ]]; then
    echo "Set SPARRKULEE_USER and SPARRKULEE_PASS, or set SPARRKULEE_URL directly." >&2
    exit 2
  fi
  REMOTE_URL="https://${SPARRKULEE_USER}:${SPARRKULEE_PASS}@homes.esat.kuleuven.be/~spchdata/corpora/auditory_eeg_data/sparrkulee"
fi

mkdir -p "${DEST_DIR}"
DEST_ABS="$(cd "${DEST_DIR}" && pwd)"
CONFIG_DIR="$(mktemp -d "${TMPDIR:-/tmp}/sparrkulee-rclone.XXXXXX")"
trap 'rm -rf "${CONFIG_DIR}"' EXIT

printf '[%s]\ntype = http\nurl = %s\n' "${REMOTE_NAME}" "${REMOTE_URL}" > "${CONFIG_DIR}/rclone.conf"

echo "Copying ${REMOTE_NAME}: into ${DEST_ABS}"

exec "${DOCKER_CMD[@]}" run --rm \
  --user "$(id -u):$(id -g)" \
  -v "${CONFIG_DIR}:/config/rclone:ro" \
  -v "${DEST_ABS}:/data" \
  rclone/rclone copy "${REMOTE_NAME}:" /data --progress "$@"
