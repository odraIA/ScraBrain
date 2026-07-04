#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_DIR="${1:-$HOME/proyectos/meegxl/ScraBrain}"

if [[ ! -d "$REPO_DIR/scripts" ]]; then
  echo "No encuentro el repositorio en: $REPO_DIR" >&2
  echo "Uso: bash scripts/legacy/install.sh /ruta/a/ScraBrain" >&2
  exit 2
fi

mkdir -p "$REPO_DIR/scripts/eeg_full_band_pipeline"

cp -f "$SOURCE_DIR/scripts/run_eeg_full_band_reading_then_listening_worker.sh"       "$REPO_DIR/scripts/"
cp -f "$SOURCE_DIR/scripts/run_eeg_full_band_reading_then_listening_sweep.sh"       "$REPO_DIR/scripts/"
cp -f "$SOURCE_DIR/scripts/run_eeg_full_band_listening_compare_sweep.sh"       "$REPO_DIR/scripts/"
cp -f "$SOURCE_DIR/scripts/eeg_full_band_pipeline/"*.sh       "$REPO_DIR/scripts/eeg_full_band_pipeline/"

chmod +x   "$REPO_DIR/scripts/run_eeg_full_band_reading_then_listening_worker.sh"   "$REPO_DIR/scripts/run_eeg_full_band_reading_then_listening_sweep.sh"   "$REPO_DIR/scripts/run_eeg_full_band_listening_compare_sweep.sh"   "$REPO_DIR/scripts/eeg_full_band_pipeline/"*.sh

bash -n "$REPO_DIR/scripts/run_eeg_full_band_reading_then_listening_worker.sh"
bash -n "$REPO_DIR/scripts/run_eeg_full_band_reading_then_listening_sweep.sh"
bash -n "$REPO_DIR/scripts/run_eeg_full_band_listening_compare_sweep.sh"

echo "Instalación completada en: $REPO_DIR"
echo "Flujo: reading (EEGDash + ZuCo) -> listening"
