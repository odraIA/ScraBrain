preprocess_stage() {
  local label="$1" config="$2" name id status
  name="scrabrain_curriculum_pre_${label}_${STAMP}"
  docker rm -f "$name" >/dev/null 2>&1 || true
  id="$(docker compose -f "$PRE_YML" run -d --no-deps --name "$name" eeg_preprocess \
    uv run --no-sync python scripts/preprocess_eeg_language_curriculum.py \
    --config-name "$config" --target-sfreq 50 --l-freq 0.1 --h-freq 40 \
    --cache-dir "$STAGING/$label" --main-cache-dir "$CACHE" | tail -1)"
  status="$(docker wait "$id")"
  docker logs "$id" > "$RUN_ROOT/preprocessing_${label}.log" 2>&1 || true
  docker rm "$id" >/dev/null 2>&1 || true
  [[ "$status" -eq 0 ]]
}

if [[ "${EEG_SKIP_PREPROCESS:-false}" != true ]]; then
  docker compose -f "$PRE_YML" build eeg_preprocess
  preprocess_stage reading train_criss_cross_eeg_reading_continuous
  preprocess_stage language train_criss_cross_eeg_language_listening_continuous
fi
