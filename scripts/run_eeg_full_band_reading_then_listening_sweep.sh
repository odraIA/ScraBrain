#!/usr/bin/env bash
set -Eeuo pipefail
trap '' HUP
LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/eeg_full_band_pipeline"
source "$LIB/launcher_env.sh"
source "$LIB/launcher_preprocess.sh"
source "$LIB/launcher_launch.sh"
source "$LIB/launcher_run.sh"
