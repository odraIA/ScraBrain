#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET_DIR="${OPENNEURO_DATASET_DIR:-${REPO_ROOT}/datasets/OpenNeuroEEG}"
OUTPUT_FILE="${OPENNEURO_TREE_OUTPUT:-${REPO_ROOT}/datasets/OpenNeuroEEG_folder_organization.txt}"
DOCKER_BIN="${DOCKER_BIN:-docker}"
DOCKER_CMD=("${DOCKER_BIN}")

if [[ "${DOCKER_USE_SUDO:-0}" == "1" ]]; then
  DOCKER_CMD=(sudo "${DOCKER_BIN}")
fi

if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "Dataset directory does not exist: ${DATASET_DIR}" >&2
  exit 1
fi

DATASET_ABS="$(cd "${DATASET_DIR}" && pwd)"
OUTPUT_DIR="$(dirname "${OUTPUT_FILE}")"
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR_ABS="$(cd "${OUTPUT_DIR}" && pwd)"
OUTPUT_NAME="$(basename "${OUTPUT_FILE}")"

echo "Writing folder organization for ${DATASET_ABS}"
echo "Output: ${OUTPUT_DIR_ABS}/${OUTPUT_NAME}"

exec "${DOCKER_CMD[@]}" run --rm \
  --user "$(id -u):$(id -g)" \
  -e OUTPUT_NAME="${OUTPUT_NAME}" \
  -v "${DATASET_ABS}:/dataset:ro" \
  -v "${OUTPUT_DIR_ABS}:/out" \
  alpine:3.20 \
  sh -c '
    set -eu
    out="/out/${OUTPUT_NAME}"
    dir_count="$(find /dataset -type d | wc -l | tr -d " ")"
    {
      printf "OpenNeuro EEG folder organization\n"
      printf "Source mount: /dataset\n"
      printf "Generated UTC: %s\n" "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
      printf "Directory count: %s\n\n" "${dir_count}"
      printf ".\n"
      find /dataset -type d | sort | awk "
        {
          sub(\"^/dataset/?\", \"\", \$0)
          if (\$0 == \"\") next
          n = split(\$0, parts, \"/\")
          indent = \"\"
          for (i = 1; i < n; i++) indent = indent \"  \"
          print indent \"- \" parts[n] \"/\"
        }
      "
    } > "${out}"
    printf "Wrote %s\n" "${out}"
  '
