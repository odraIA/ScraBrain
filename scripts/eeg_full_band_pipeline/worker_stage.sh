run_stage() {
  local stage="$1" config="$2" exp="$3" batch="$4" promoted="${5:-}" resume_path=""
  local -a init resume_args cmd
  if completed "$exp"; then
    LAST_CKPT="$(best_ckpt "$exp")"
    echo "REUSING completed $stage: $exp"
    return
  fi
  if [[ "$RESUME" == true ]] && resume_path="$(resume_ckpt "$exp")"; then
    resume_args=('checkpoint.resume=true' "checkpoint.resume_path=$resume_path")
    echo "RESUMING $stage from $resume_path"
  else
    resume_args=('checkpoint.resume=false' 'checkpoint.resume_path=null')
  fi
  if [[ "$stage" == reading && "$INIT" == pretrained ]]; then
    init=('model.train_from_scratch=false' 'model.use_promoted_checkpoint=false' "model.criss_cross_checkpoint=$MEGXL")
  elif [[ "$stage" == reading ]]; then
    init=('model.train_from_scratch=true' 'model.use_promoted_checkpoint=false')
  else
    [[ -s "$promoted" ]] || { echo "Missing reading checkpoint: $promoted" >&2; return 4; }
    init=('model.train_from_scratch=false' 'model.use_promoted_checkpoint=true' "model.promoted_checkpoint=$promoted")
  fi
  cmd=(uv run --no-sync python -m brainstorm.train_criss_cross_eeg_continuous --config-name "$config"
    "${COMMON[@]}" "training.batch_size=$batch" "logging.experiment_name=$exp"
    "logging.save_dir=$LOG_ROOT" "checkpoint.save_dir=$CKPT_ROOT" "${init[@]}" "${resume_args[@]}")
  echo "STARTING $stage: $exp"
  "${cmd[@]}"
  LAST_CKPT="$(best_ckpt "$exp")"
  [[ -s "$LAST_CKPT" ]]
}
