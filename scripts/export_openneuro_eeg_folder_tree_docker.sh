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
    patterns_dir="/tmp/openneuro_file_patterns"
    rm -rf "${patterns_dir}"
    mkdir -p "${patterns_dir}"

    indent_for_rel() {
      rel="$1"
      depth=0
      rest="${rel}"
      while [ "${rest#*/}" != "${rest}" ]; do
        depth=$((depth + 1))
        rest="${rest#*/}"
      done

      indent=""
      while [ "${depth}" -gt 0 ]; do
        indent="${indent}  "
        depth=$((depth - 1))
      done
      printf "%s" "${indent}"
    }

    dir_count="$(find /dataset -type d | wc -l | tr -d " ")"
    file_count="$(find /dataset -type f | wc -l | tr -d " ")"
    {
      printf "OpenNeuro EEG folder and file organization\n"
      printf "Source mount: /dataset\n"
      printf "Generated UTC: %s\n" "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
      printf "Directory count: %s\n" "${dir_count}"
      printf "File count: %s\n\n" "${file_count}"
      printf ".\n"

      find /dataset -type d | sort | while IFS= read -r dir; do
        rel="${dir#/dataset}"
        rel="${rel#/}"

        if [ -n "${rel}" ]; then
          indent="$(indent_for_rel "${rel}")"
          base="${rel##*/}"
          printf "%s- %s/\n" "${indent}" "${base}"
          file_indent="${indent}  "
          display_dir="./${rel}"
        else
          file_indent="  "
          display_dir="."
        fi

        file_list="$(find "${dir}" -maxdepth 1 -type f | sed "s|.*/||" | sort)"
        if [ -z "${file_list}" ]; then
          continue
        fi

        signature="$(printf "%s\n" "${file_list}" | cksum | awk "{print \$1 \"_\" \$2}")"
        pattern_file="${patterns_dir}/${signature}"

        if [ -f "${pattern_file}" ]; then
          first_dir="$(cat "${pattern_file}")"
          printf "%s- [same direct files as %s]\n" "${file_indent}" "${first_dir}"
          continue
        fi

        printf "%s\n" "${display_dir}" > "${pattern_file}"
        printf "%s\n" "${file_list}" | while IFS= read -r file_name; do
          printf "%s- %s\n" "${file_indent}" "${file_name}"
        done
      done
    } > "${out}"
    printf "Wrote %s\n" "${out}"
  '
