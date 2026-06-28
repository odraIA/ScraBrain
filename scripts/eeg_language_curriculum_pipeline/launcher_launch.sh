launch_worker() {
  local gpu="$1"
  local idx="$2"
  local name="scrabrain_curriculum_worker_${idx}_${STAMP}"
  EEG_GPU="$gpu" docker compose -f "$TRAIN_YML" run -d --build --no-deps \
  --env-from-file "$WORKER_ENV_FILE" -e "EEG_WORKER_GPU=$gpu" \
  --name "$name" eeg_train_reading_listening \
  uv run --no-sync python scripts/run_eeg_language_curriculum_three_models_worker.py \
  > "$RUN_ROOT/worker_${idx}.container_id"
  printf '%s\n' "$name" > "$RUN_ROOT/worker_${idx}.container_name"
}
