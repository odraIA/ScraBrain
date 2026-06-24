#!/usr/bin/env bash
set -Eeuo pipefail
trap '' HUP
cd /workspace
LIB=/workspace/scripts/eeg_full_band_pipeline
source "$LIB/worker_env.sh"
source "$LIB/worker_helpers.sh"
source "$LIB/worker_stage.sh"
source "$LIB/worker_run.sh"
