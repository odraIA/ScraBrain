INIT="${EEG_INIT_MODE:?pretrained or from_scratch}"
[[ "$INIT" == pretrained || "$INIT" == from_scratch ]] || exit 2
SEED="${EEG_SEED:-42}"
CACHE="${EEG_CACHE_DIR:-./data/cache/eeg_preprocessed}"
LOG_ROOT="${EEG_TRAIN_LOG_ROOT:?}"
CKPT_ROOT="${EEG_CHECKPOINT_ROOT:?}"
RUN_ROOT="${EEG_PIPELINE_RUN_ROOT:?}"
READ_BATCH="${EEG_READING_BATCH_SIZE:-4}"
LISTEN_BATCH="${EEG_LISTENING_BATCH_SIZE:-4}"
WORKERS="${EEG_NUM_WORKERS:-6}"
VAL_EVERY="${EEG_VAL_CHECK_INTERVAL:-500}"
SAVE_EVERY="${EEG_CHECKPOINT_EVERY_N_TRAIN_STEPS:-5000}"
MEGXL="${CRISS_CROSS_CHECKPOINT:-./checkpoints/baseline/meg-xl-med.ckpt}"
RESUME="${EEG_RESUME:-true}"
READ_EXP="eeg_full_band_0p1_50_fixed50_50hz_biocodec_${INIT}_reading_seed${SEED}"
LISTEN_EXP="eeg_full_band_0p1_50_fixed50_50hz_biocodec_${INIT}_listening_seed${SEED}"
PIPE_DIR="${RUN_ROOT}/pipelines/${INIT}"
mkdir -p "$PIPE_DIR" "$LOG_ROOT" "$CKPT_ROOT"
COMMON=(data.target_sfreq=50.0 model.sampling_rate=50 data.l_freq=0.1 data.h_freq=50.0
  "data.cache_dir=$CACHE" "training.num_workers=$WORKERS" training.persistent_workers=true
  trainer.devices=1 trainer.strategy=auto "trainer.val_check_interval=$VAL_EVERY"
  "checkpoint.every_n_train_steps=$SAVE_EVERY" model.tokenizer_name=biocodec
  model.tokenizer_variant=default model.tokenizer_checkpoint=./brainstorm/neuro_tokenizers/biocodec_ckpt.pt
  model.tokenizer_ckpt=./brainstorm/neuro_tokenizers/biocodec_ckpt.pt model.tokenizer_config_path=null
  model.vocab_size=256 model.num_quantizers=6 model.num_quantizers_used=6
  model.tokenizer_downsample_ratio=12 model.overlap_ratio=0.0 "seed=$SEED")
