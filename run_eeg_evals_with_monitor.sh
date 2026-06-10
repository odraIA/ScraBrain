#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

STATE_FILE="${EEG_SEQUENCE_STATE_FILE:-.eeg_eval_sequence.json}"
MONITOR_SERVICE="${EEG_MONITOR_SERVICE:-eeg_monitor}"
MONITOR_PORT="${EEG_MONITOR_PORT:-8082}"
MONITOR_URL="${EEG_MONITOR_URL:-http://localhost:${MONITOR_PORT}}"
CONTINUE_ON_ERROR=0
DRY_RUN=0
START_MONITOR=1

SERVICES=(
  "eval_eeg_reading:EEG reading"
  "eval_eeg_listening:EEG listening"
  "eval_eeg_reading_listening:EEG reading+listening"
)

usage() {
  cat <<'EOF'
Usage: bash run_eeg_evals_with_monitor.sh [options]

Starts eeg_monitor, then runs the three EEG eval Docker services in order:
  1. eval_eeg_reading
  2. eval_eeg_listening
  3. eval_eeg_reading_listening

Options:
  --dry-run            Print the docker compose commands and update monitor state only.
  --no-monitor         Do not start eeg_monitor.
  --continue-on-error  Run later evaluations even if an earlier one fails.
  -h, --help           Show this help.

Useful env vars:
  EEG_MONITOR_PORT=8082
  EEG_TARGET_SFREQ=50
  EEG_TOKENIZER_NAME=biocodec
  EEG_TRAIN_FROM_SCRATCH=false
  EEG_USE_PROMOTED_CHECKPOINT=false
  EEG_PROMOTED_CHECKPOINT=...
  CRISS_CROSS_CHECKPOINT=...
  BIOCODEC_CHECKPOINT=...
  WANDB_MODE=offline
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      ;;
    --no-monitor)
      START_MONITOR=0
      ;;
    --continue-on-error)
      CONTINUE_ON_ERROR=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

json_state() {
  local action="$1"
  local service="${2:-}"
  local status="${3:-}"
  local exit_code="${4:-}"
  ACTION="$action" \
  SERVICE="$service" \
  STATUS="$status" \
  EXIT_CODE="$exit_code" \
  STATE_FILE="$STATE_FILE" \
  MONITOR_SERVICE="$MONITOR_SERVICE" \
  MONITOR_URL="$MONITOR_URL" \
  DRY_RUN="$DRY_RUN" \
  PID="$$" \
  SERVICES_JOINED="$(printf '%s\n' "${SERVICES[@]}")" \
  python - <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


state_path = Path(os.environ["STATE_FILE"])
action = os.environ["ACTION"]
service = os.environ.get("SERVICE") or ""
status = os.environ.get("STATUS") or ""
exit_code_raw = os.environ.get("EXIT_CODE") or ""

if state_path.exists():
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(state, dict):
            state = {}
    except Exception:
        state = {}
else:
    state = {}

services = []
for line in os.environ["SERVICES_JOINED"].splitlines():
    if not line.strip():
        continue
    service_name, _, label = line.partition(":")
    services.append({
        "service": service_name,
        "label": label or service_name,
        "status": "pending",
        "started_at": None,
        "finished_at": None,
        "exit_code": None,
    })

if action == "init" or not state:
    state = {
        "kind": "eeg_eval_sequence",
        "status": "dry_run" if os.environ["DRY_RUN"] == "1" else "running",
        "started_at": now(),
        "updated_at": now(),
        "finished_at": None,
        "monitor_service": os.environ["MONITOR_SERVICE"],
        "monitor_url": os.environ["MONITOR_URL"],
        "dry_run": os.environ["DRY_RUN"] == "1",
        "pid": int(os.environ["PID"]),
        "current_service": None,
        "services": services,
    }

by_service = {
    item.get("service"): item
    for item in state.get("services", [])
    if isinstance(item, dict)
}
for item in services:
    by_service.setdefault(item["service"], item)
state["services"] = [by_service[item["service"]] for item in services]
state["updated_at"] = now()
state["monitor_service"] = os.environ["MONITOR_SERVICE"]
state["monitor_url"] = os.environ["MONITOR_URL"]
state["dry_run"] = os.environ["DRY_RUN"] == "1"

if action == "start_service":
    row = by_service[service]
    row["status"] = "running"
    row["started_at"] = now()
    row["finished_at"] = None
    row["exit_code"] = None
    state["status"] = "running"
    state["current_service"] = service
elif action == "finish_service":
    row = by_service[service]
    row["status"] = status
    row["finished_at"] = now()
    row["exit_code"] = int(exit_code_raw) if exit_code_raw else None
    running = [item for item in state["services"] if item.get("status") == "running"]
    state["current_service"] = running[0]["service"] if running else None
elif action == "finish_sequence":
    state["status"] = status
    state["finished_at"] = now()
    state["current_service"] = None

state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

json_state init

if [[ "$START_MONITOR" -eq 1 ]]; then
  echo "[EEG] Starting monitor service: ${MONITOR_SERVICE}"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "docker compose up -d ${MONITOR_SERVICE}"
  else
    docker compose up -d "$MONITOR_SERVICE"
  fi
  echo "[EEG] Monitor URL: ${MONITOR_URL}"
else
  echo "[EEG] Monitor start skipped. Existing monitor can read ${STATE_FILE}."
fi

overall_status="done"

for entry in "${SERVICES[@]}"; do
  service="${entry%%:*}"
  label="${entry#*:}"
  echo "[EEG] Running ${label}: docker compose run --rm ${service}"
  json_state start_service "$service"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "docker compose run --rm ${service}"
    rc=0
  else
    set +e
    docker compose run --rm "$service"
    rc=$?
    set -e
  fi

  if [[ "$rc" -eq 0 ]]; then
    json_state finish_service "$service" done "$rc"
    echo "[EEG] ${label} completed"
  else
    json_state finish_service "$service" failed "$rc"
    echo "[EEG] ${label} failed with exit code ${rc}" >&2
    overall_status="failed"
    if [[ "$CONTINUE_ON_ERROR" -ne 1 ]]; then
      json_state finish_sequence "" failed
      exit "$rc"
    fi
  fi
done

json_state finish_sequence "" "$overall_status"
echo "[EEG] Sequence ${overall_status}. State file: ${STATE_FILE}"
echo "[EEG] Monitor URL: ${MONITOR_URL}"
