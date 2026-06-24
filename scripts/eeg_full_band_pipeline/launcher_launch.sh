launch() {
  local gpu="$1" init="$2" name="scrabrain_fullband_${init}_${STAMP}" id
  docker rm -f "$name" >/dev/null 2>&1 || true
  id="$(EEG_GPU="$gpu" EEG_GPU_COUNT=1 WANDB_MODE="${WANDB_MODE:-offline}" \
    docker compose -f "$TRAIN_YML" run -d --no-deps --name "$name" \
    -e "EEG_INIT_MODE=$init" -e "EEG_SEED=$SEED" \
    -e "EEG_READING_BATCH_SIZE=$READ_BATCH" -e "EEG_LISTENING_BATCH_SIZE=$LISTEN_BATCH" \
    -e "EEG_NUM_WORKERS=${EEG_NUM_WORKERS:-6}" -e "EEG_CACHE_DIR=$CACHE" \
    -e "EEG_PIPELINE_RUN_ROOT=$RUN_ROOT" -e "EEG_TRAIN_LOG_ROOT=$LOG_ROOT" \
    -e "EEG_CHECKPOINT_ROOT=$CKPT_ROOT" -e "EEG_RESUME=${EEG_RESUME:-true}" \
    -e "CRISS_CROSS_CHECKPOINT=${CRISS_CROSS_CHECKPOINT:-./checkpoints/baseline/meg-xl-med.ckpt}" \
    eeg_train_reading_listening bash /workspace/scripts/run_eeg_full_band_reading_then_listening_worker.sh \
    | tail -1)"
  printf '%s\n' "$id" > "$RUN_ROOT/${init}.container_id"
  printf '%s\n' "$name" > "$RUN_ROOT/${init}.container_name"
  echo "$init: $name ($id)"
}
