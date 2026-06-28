#!/usr/bin/env bash
set -Eeuo pipefail
trap '' HUP
LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/eeg_language_curriculum_pipeline"
source "$LIB/launcher_env.sh"
source "$LIB/launcher_queue.sh"
source "$LIB/launcher_preprocess.sh"
source "$LIB/launcher_launch.sh"
source "$LIB/launcher_run.sh"
