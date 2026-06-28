ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
TRAIN_YML="${EEG_COMPOSE_FILE:-$ROOT/docker-compose.eeg-reading-listening.yml}"
PRE_YML="${EEG_PREPROCESS_COMPOSE_FILE:-$ROOT/docker-compose.eeg-preprocess.yml}"
STAMP="${EEG_SWEEP_STAMP:-$(date +%Y%m%d_%H%M%S)}"
SEED="${EEG_SEED:-42}"
CACHE="${EEG_CACHE_DIR:-./data/cache/eeg_preprocessed}"
RUN_ROOT="${EEG_SWEEP_ROOT:-results/eeg_language_curriculum_three_models/$STAMP}"
LOG_ROOT="${EEG_TRAIN_LOG_ROOT:-./logs/eeg_language_curriculum_three_models/$STAMP}"
CKPT_ROOT="${EEG_CHECKPOINT_ROOT:-./checkpoints/eeg_language_curriculum_three_models/$STAMP}"
STAGING="${EEG_STAGING_CACHE_ROOT:-./data/cache/eeg_language_curriculum_staging/$STAMP}"
READ_BATCH="${EEG_READING_BATCH_SIZE:-${EEG_BATCH_SIZE:-4}}"
LANG_BATCH="${EEG_LANGUAGE_BATCH_SIZE:-${EEG_LISTENING_BATCH_SIZE:-1}}"
WANDB_MODE="${WANDB_MODE:-offline}"
MEGXL="${CRISS_CROSS_CHECKPOINT:-./checkpoints/baseline/meg-xl-med.ckpt}"
read -r -a GPUS <<< "${EEG_GPUS:-0 1}"
[[ ${#GPUS[@]} -ge 1 ]] || { echo "EEG_GPUS needs at least one GPU." >&2; exit 2; }
mkdir -p "$RUN_ROOT" "$LOG_ROOT" "$CKPT_ROOT" "$STAGING" "$CACHE"
cat > "$RUN_ROOT/metadata.txt" <<META
Order: reading -> language listening
Reading: EEGDash + ZuCo
Language listening: SparrKULee + Weissbart + Alice EEG + ds007808
Held out: ds004408
Models: from_scratch/eeg2, MEG-XL/eeg2, MEG-XL/eeg1
Physical EEG sensor id remains 2; eeg1 only redirects the type embedding lookup.
GPUs: ${GPUS[*]}
Reading batch: $READ_BATCH
Language batch: $LANG_BATCH
MEG-XL checkpoint: $MEGXL
META
